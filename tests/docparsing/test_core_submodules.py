"""Unit tests covering refactored DocParsing core submodules."""

from __future__ import annotations

import argparse
import io
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import DocsToKG.DocParsing.core.http as core_http
from DocsToKG.DocParsing.config import ConfigLoadError, load_toml_markers, load_yaml_markers
from DocsToKG.DocParsing.core import (
    ResumeController,
    compute_relative_doc_id,
    compute_stable_shard,
    derive_doc_id_and_chunks_path,
    derive_doc_id_and_doctags_path,
    derive_doc_id_and_vectors_path,
    load_structural_marker_profile,
    normalize_http_timeout,
    should_skip_output,
)
from DocsToKG.DocParsing.core import cli as core_cli
from DocsToKG.DocParsing.core.cli_utils import merge_args, preview_list
from DocsToKG.DocParsing.core.http import get_http_session
from DocsToKG.DocParsing.core.planning import (
    display_plan,
    plan_chunk,
    plan_doctags,
    plan_embed,
)


@pytest.fixture
def planning_module_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide lightweight DocParsing submodules for planning tests."""

    import DocsToKG.DocParsing as docparsing_pkg

    stub_doctags = types.ModuleType("DocsToKG.DocParsing.doctags")
    stub_doctags.HTML_MANIFEST_STAGE = "doctags-html"
    stub_doctags.MANIFEST_STAGE = "doctags-pdf"

    def add_data_root_option(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-root", type=Path, default=None)

    def add_resume_force_options(
        parser: argparse.ArgumentParser, *, resume_help: str = "", force_help: str = ""
    ) -> None:
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--force", action="store_true")

    def list_htmls(directory: Path) -> list[Path]:
        return sorted(Path(directory).rglob("*.html"))

    def list_pdfs(directory: Path) -> list[Path]:
        return sorted(Path(directory).rglob("*.pdf"))

    def prepare_data_root(data_root: Path | None, detected: Path) -> Path:
        if data_root is not None:
            return Path(data_root).expanduser().resolve()
        return Path(detected).expanduser().resolve()

    def resolve_pipeline_path(
        *,
        cli_value: Path | None,
        default_path: Path,
        resolved_data_root: Path,
        data_root_overridden: bool,
        resolver,
    ) -> Path:
        if cli_value is not None:
            return Path(cli_value).expanduser().resolve()
        if default_path is not None:
            return Path(default_path).expanduser().resolve()
        resolved_root = Path(resolved_data_root).expanduser().resolve()
        return Path(resolver(resolved_root)).expanduser().resolve()

    stub_doctags.add_data_root_option = add_data_root_option
    stub_doctags.add_resume_force_options = add_resume_force_options
    stub_doctags.list_htmls = list_htmls
    stub_doctags.list_pdfs = list_pdfs
    stub_doctags.prepare_data_root = prepare_data_root
    stub_doctags.resolve_pipeline_path = resolve_pipeline_path

    stub_chunking = types.ModuleType("DocsToKG.DocParsing.chunking")
    stub_chunking.MANIFEST_STAGE = "chunks"

    def build_chunk_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--data-root", type=Path, default=None)
        parser.add_argument("--in-dir", type=Path, default=None)
        parser.add_argument("--out-dir", type=Path, default=None)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--force", action="store_true")
        return parser

    stub_chunking.build_parser = build_chunk_parser

    stub_embedding = types.ModuleType("DocsToKG.DocParsing.embedding")
    stub_embedding.MANIFEST_STAGE = "embeddings"

    def build_embed_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--data-root", type=Path, default=None)
        parser.add_argument("--chunks-dir", type=Path, default=None)
        parser.add_argument("--out-dir", type=Path, default=None)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--validate-only", action="store_true")
        return parser

    stub_embedding.build_parser = build_embed_parser

    monkeypatch.setitem(sys.modules, "DocsToKG.DocParsing.doctags", stub_doctags)
    monkeypatch.setitem(sys.modules, "DocsToKG.DocParsing.chunking", stub_chunking)
    monkeypatch.setitem(sys.modules, "DocsToKG.DocParsing.embedding", stub_embedding)

    monkeypatch.setitem(docparsing_pkg._MODULE_CACHE, "doctags", stub_doctags)
    monkeypatch.setitem(docparsing_pkg._MODULE_CACHE, "chunking", stub_chunking)
    monkeypatch.setitem(docparsing_pkg._MODULE_CACHE, "embedding", stub_embedding)

    monkeypatch.setattr(docparsing_pkg, "doctags", stub_doctags, raising=False)
    monkeypatch.setattr(docparsing_pkg, "chunking", stub_chunking, raising=False)
    monkeypatch.setattr(docparsing_pkg, "embedding", stub_embedding, raising=False)


def test_normalize_http_timeout_scalar_and_iterables() -> None:
    """HTTP timeout coercion handles scalars, strings, and iterables."""

    assert normalize_http_timeout(None) == (5.0, 30.0)
    assert normalize_http_timeout(12) == (5.0, 12.0)
    assert normalize_http_timeout("1.5, 2.5") == (1.5, 2.5)
    assert normalize_http_timeout([2, 3]) == (2.0, 3.0)


def test_get_http_session_reuses_singleton_without_base_headers() -> None:
    """Shared HTTP session is memoised when not requesting transient headers."""

    with (
        mock.patch.object(core_http, "_HTTP_SESSION", None),
        mock.patch.object(core_http, "_HTTP_SESSION_TIMEOUT", core_http.DEFAULT_HTTP_TIMEOUT),
    ):
        session_a, timeout_a = get_http_session(timeout=10)
        session_b, timeout_b = get_http_session()

        assert session_a is session_b
        assert timeout_a == (5.0, 10.0)
        assert timeout_b == (5.0, 30.0)


def test_get_http_session_transient_headers_do_not_leak() -> None:
    """Transient base headers return isolated sessions without contaminating shared state."""

    with (
        mock.patch.object(core_http, "_HTTP_SESSION", None),
        mock.patch.object(core_http, "_HTTP_SESSION_TIMEOUT", core_http.DEFAULT_HTTP_TIMEOUT),
    ):
        session_a, _ = get_http_session(base_headers={"Authorization": "Token A"})
        session_b, _ = get_http_session(base_headers={"Authorization": "Token B"})
        shared_session, _ = get_http_session()

        assert session_a is not session_b
        assert session_a.headers["Authorization"] == "Token A"
        assert session_b.headers["Authorization"] == "Token B"
        assert shared_session is not session_a
        assert shared_session is not session_b
        assert "Authorization" not in shared_session.headers


def test_manifest_resume_controller(tmp_path: Path) -> None:
    """ResumeController mirrors should_skip_output behaviour."""

    output = tmp_path / "out.jsonl"
    manifest_success = {"input_hash": "abc123", "status": "success"}
    manifest_skip = {"input_hash": "abc123", "status": "skip"}
    manifest_failure = {"input_hash": "abc123", "status": "failure"}
    manifest_unknown = {"input_hash": "abc123", "status": "other"}
    manifest_missing_status = {"input_hash": "abc123"}

    assert not should_skip_output(output, manifest_success, "abc123", resume=True, force=True)

    output.write_text("data", encoding="utf-8")
    assert should_skip_output(output, manifest_success, "abc123", resume=True, force=False)
    assert should_skip_output(output, manifest_skip, "abc123", resume=True, force=False)
    assert not should_skip_output(output, manifest_failure, "abc123", resume=True, force=False)
    assert not should_skip_output(output, manifest_unknown, "abc123", resume=True, force=False)
    assert not should_skip_output(output, manifest_missing_status, "abc123", resume=True, force=False)

    controller = ResumeController(
        resume=True,
        force=False,
        manifest_index={"doc1": manifest_success, "doc2": manifest_failure},
    )

    skip, entry = controller.should_skip("doc1", output, "abc123")
    assert skip is True and entry is manifest_success

    skip_failure, entry_failure = controller.should_skip("doc2", output, "abc123")
    assert skip_failure is False and entry_failure is manifest_failure

    process, _ = controller.should_process("doc1", output, "zzzz")
    assert process is True


def test_path_derivation_helpers(tmp_path: Path) -> None:
    """Path derivation utilities maintain stable relative identifiers."""

    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    doctags_root = tmp_path / "doctags"
    doctags_root.mkdir()
    chunks_root = tmp_path / "chunks"
    chunks_root.mkdir()
    vectors_root = tmp_path / "vectors"
    vectors_root.mkdir()

    pdf_path = pdf_root / "report.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")
    doctags_path = doctags_root / "report.doctags"
    doctags_path.write_text("{}", encoding="utf-8")
    chunk_path = chunks_root / "report.chunks.jsonl"
    chunk_path.write_text("{}", encoding="utf-8")

    doc_id, doctags_out = derive_doc_id_and_doctags_path(pdf_path, pdf_root, doctags_root)
    assert doc_id == "report.pdf"
    assert doctags_out.name == "report.doctags"

    doc_id, chunk_out = derive_doc_id_and_chunks_path(doctags_path, doctags_root, chunks_root)
    assert doc_id == "report.doctags"
    assert chunk_out.name == "report.chunks.jsonl"

    doc_id, vector_out = derive_doc_id_and_vectors_path(chunk_path, chunks_root, vectors_root)
    assert doc_id == "report.doctags"
    assert vector_out.name == "report.vectors.jsonl"
    assert compute_relative_doc_id(vector_out, vectors_root) == "report.vectors.jsonl"


def test_compute_stable_shard_distribution() -> None:
    """Stable shard uses hash modulus semantics."""

    shard = compute_stable_shard("example", 8)
    assert 0 <= shard < 8
    assert compute_stable_shard("example", 8) == shard
    with pytest.raises(ValueError):
        compute_stable_shard("oops", 0)


def test_structural_marker_profile(tmp_path: Path) -> None:
    """Structural marker loader accepts JSON format."""

    json_file = tmp_path / "markers.json"
    json_file.write_text('{"headings": ["#"], "captions": ["Figure"]}', encoding="utf-8")
    headings, captions = load_structural_marker_profile(json_file)
    assert headings == ["#"]
    assert captions == ["Figure"]


def test_cli_utils_preview_and_merge() -> None:
    """Preview helpers and argument merging behave deterministically."""

    assert preview_list(["a", "b", "c"]) == ["a", "b", "c"]
    assert preview_list(list("abcdef"), limit=3) == ["a", "b", "c", "... (+3 more)"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--foo")
    parser.add_argument("--bar", type=int, default=1)

    merged = merge_args(parser, {"foo": "value"})
    assert merged.foo == "value"
    assert merged.bar == 1


def test_load_structural_marker_profile_yaml(tmp_path: Path) -> None:
    """YAML structural marker profile is parsed via public helper."""

    yaml_file = tmp_path / "markers.yaml"
    yaml_file.write_text('headings:\n  - "##"\ncaptions: []\n', encoding="utf-8")
    headings, captions = load_structural_marker_profile(yaml_file)
    assert headings == ["##"]
    assert captions == []


def test_load_yaml_markers_success() -> None:
    """Public YAML loader returns parsed objects."""

    data = load_yaml_markers("headings:\n  - '#'")
    assert data == {"headings": ["#"]}


def test_load_toml_markers_success() -> None:
    """TOML loader parses dictionaries."""

    data = load_toml_markers('headings = ["#"]\ncaptions = []')
    assert data == {"headings": ["#"], "captions": []}


def test_load_toml_markers_failure() -> None:
    """Malformed TOML surfaces ConfigLoadError with context."""

    with pytest.raises(ConfigLoadError) as excinfo:
        load_toml_markers("[bad\nvalue = 1")
    assert "TOML" in str(excinfo.value)


def test_chunk_cli_validation_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Chunk CLI surfaces validation errors without tracebacks."""

    exit_code = core_cli.chunk(["--min-tokens", "-1"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "chunk" in captured.err
    assert "--min-tokens" in captured.err
    assert "non-negative" in captured.err


def test_embed_cli_validation_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Embedding CLI reports conflicting flag usage cleanly."""

    exit_code = core_cli.embed(["--plan-only", "--validate-only"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "embed" in captured.err
    assert "cannot be combined" in captured.err


def test_display_plan_stream_output() -> None:
    """Planner display writes to injected stream and returns lines."""

    plans = [
        {
            "stage": "doctags",
            "mode": "html",
            "process": {"count": 1, "preview": ["a"]},
            "skip": {"count": 0, "preview": []},
            "input_dir": "/input/doc",
            "output_dir": "/output/doc",
            "notes": ["Ready"],
        },
        {
            "stage": "embed",
            "action": "validate",
            "validate": {"count": 1, "preview": ["x"]},
            "missing": {"count": 0, "preview": []},
            "chunks_dir": "/chunks",
            "vectors_dir": "/vectors",
            "notes": [],
        },
    ]
    buffer = io.StringIO()
    lines = display_plan(plans, stream=buffer)
    rendered = buffer.getvalue().splitlines()
    assert lines == rendered
    assert lines[-1] == ""
    assert "docparse all plan" in rendered[0]
    assert any("doctags" in line for line in rendered)
    assert any("process 1" in line for line in rendered)
    assert any("embed (validate-only)" in line for line in rendered)


def test_plan_doctags_auto_detection_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """DocTags planner surfaces auto-detection conflicts as friendly notes."""

    data_root = tmp_path / "Data"
    html_dir = data_root / "HTML"
    pdf_dir = data_root / "PDFs"
    html_dir.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)
    (html_dir / "doc.html").write_text("<html></html>", encoding="utf-8")
    (pdf_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    plan = plan_doctags(["--mode", "auto"])

    assert plan["stage"] == "doctags"
    assert plan["process"] == [] and plan["skip"] == []
    assert plan["notes"], "Expected an explanatory note when mode detection fails"
    note = plan["notes"][0]
    assert "Cannot auto-detect mode" in note
    assert "Hint:" in note


def test_plan_doctags_without_resume_skips_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """DocTags planner avoids hashing inputs when resume is disabled."""

    html_dir = tmp_path / "html"
    output_dir = tmp_path / "doctags"
    html_dir.mkdir()
    output_dir.mkdir()
    (html_dir / "doc.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    def _raise_hash(_path: Path) -> str:
        raise AssertionError("compute_content_hash should not run without resume")

    monkeypatch.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash
    )

    plan = plan_doctags(
        [
            "--mode",
            "html",
            "--in-dir",
            str(html_dir),
            "--out-dir",
            str(output_dir),
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0


def test_plan_chunk_resume_missing_manifest_skips_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Chunk planner does not hash when resume entry is absent."""

    data_root = tmp_path / "data"
    in_dir = data_root / "DocTagsFiles"
    out_dir = data_root / "ChunkedDocTagFiles"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "doc1.doctags").write_text("{}", encoding="utf-8")

    def _raise_hash(_path: Path) -> str:
        raise AssertionError("compute_content_hash should not run without manifest entry")

    monkeypatch.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash
    )
    monkeypatch.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index", lambda *_args, **_kwargs: {}
    )

    plan = plan_chunk(
        [
            "--data-root",
            str(data_root),
            "--in-dir",
            str(in_dir),
            "--out-dir",
            str(out_dir),
            "--resume",
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0


def test_plan_embed_resume_missing_manifest_skips_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Embedding planner avoids hashing when resume entry is absent."""

    data_root = tmp_path / "data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)
    (chunks_dir / "doc1.chunks.jsonl").write_text("{}\n", encoding="utf-8")

    def _raise_hash(_path: Path) -> str:
        raise AssertionError("compute_content_hash should not run without manifest entry")

    monkeypatch.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash
    )
    monkeypatch.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index", lambda *_args, **_kwargs: {}
    )

    plan = plan_embed(
        [
            "--data-root",
            str(data_root),
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(vectors_dir),
            "--resume",
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0


def test_plan_embed_generate_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Embedding planner reports generate counts for discovered chunks."""

    data_root = tmp_path / "data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)
    (chunks_dir / "doc1.chunks.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root", lambda *_args, **_kwargs: data_root
    )

    plan = plan_embed(
        [
            "--data-root",
            str(data_root),
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(vectors_dir),
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0
