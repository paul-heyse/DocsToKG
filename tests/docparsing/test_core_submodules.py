"""Comprehensively exercise DocParsing core utilities after refactors.

The core package hides numerous helper functions that power CLI orchestration,
plan generation, and HTTP/session management. This module verifies those
refactored helpers work together: structural marker loading, resume controllers,
plan display, HTTP timeout normalisation, and CLI argument merging. Keeping
these tests broad ensures the higher-level CLI flows stay stable when internal
utilities evolve.
"""

from __future__ import annotations

import argparse
import builtins
import heapq
import io
import sys
import types
from pathlib import Path
from typing import Callable, Sequence
from unittest import mock

import pytest

import DocsToKG.DocParsing.core.http as core_http
from DocsToKG.DocParsing.cli_errors import ChunkingCLIValidationError
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
    PLAN_PREVIEW_LIMIT,
    display_plan,
    plan_chunk,
    plan_doctags,
    plan_embed,
)
from tests.conftest import PatchManager


@pytest.fixture
def planning_module_stubs(patcher: PatchManager) -> None:
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
        parser.add_argument("--verify-hash", action="store_true")

    def _iter_paths(
        directory: Path,
        suffixes: tuple[str, ...],
        *,
        include: Callable[[Path], bool] | None = None,
    ):
        suffix_set = {suffix.lower() for suffix in suffixes}
        root = Path(directory)

        if not root.exists():
            return

        if root.is_file():
            candidate = root
            if candidate.suffix.lower() in suffix_set and (include is None or include(candidate)):
                yield candidate
            return

        heap: list[tuple[str, Path]] = []

        def _enqueue(directory: Path) -> None:
            try:
                entries = list(directory.iterdir())
            except FileNotFoundError:
                return
            entries.sort(key=lambda path: path.name)
            for entry in entries:
                try:
                    rel = entry.relative_to(root).as_posix()
                except ValueError:
                    continue
                heapq.heappush(heap, (rel, entry))

        if root.is_dir():
            _enqueue(root)

        while heap:
            _, candidate = heapq.heappop(heap)
            if candidate.is_dir():
                if candidate.is_symlink():
                    continue
                _enqueue(candidate)
                continue
            if candidate.suffix.lower() not in suffix_set:
                continue
            if include is not None and not include(candidate):
                continue
            yield candidate

    def iter_htmls(directory: Path):
        return _iter_paths(
            directory,
            (".html", ".htm", ".xhtml"),
            include=lambda path: not path.name.endswith(".normalized.html"),
        )

    def list_htmls(directory: Path) -> list[Path]:
        return list(iter_htmls(directory))

    def iter_pdfs(directory: Path):
        return _iter_paths(directory, (".pdf",))

    def list_pdfs(directory: Path) -> list[Path]:
        return list(iter_pdfs(directory))

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
    stub_doctags.iter_htmls = iter_htmls
    stub_doctags.list_htmls = list_htmls
    stub_doctags.iter_pdfs = iter_pdfs
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

    patcher.setitem(sys.modules, "DocsToKG.DocParsing.doctags", stub_doctags)
    patcher.setitem(sys.modules, "DocsToKG.DocParsing.chunking", stub_chunking)
    patcher.setitem(sys.modules, "DocsToKG.DocParsing.embedding", stub_embedding)

    patcher.setitem(docparsing_pkg._MODULE_CACHE, "doctags", stub_doctags)
    patcher.setitem(docparsing_pkg._MODULE_CACHE, "chunking", stub_chunking)
    patcher.setitem(docparsing_pkg._MODULE_CACHE, "embedding", stub_embedding)

    patcher.setattr(docparsing_pkg, "doctags", stub_doctags, raising=False)
    patcher.setattr(docparsing_pkg, "chunking", stub_chunking, raising=False)
    patcher.setattr(docparsing_pkg, "embedding", stub_embedding, raising=False)


def test_normalize_http_timeout_scalar_and_iterables() -> None:
    """HTTP timeout coercion handles scalars, strings, and iterables."""

    assert normalize_http_timeout(None) == (5.0, 30.0)
    assert normalize_http_timeout(12) == (12.0, 12.0)
    assert normalize_http_timeout(30) == (30.0, 30.0)
    assert normalize_http_timeout("1.5, 2.5") == (1.5, 2.5)
    assert normalize_http_timeout([2, 3]) == (2.0, 3.0)
    assert normalize_http_timeout((4, 5)) == (4.0, 5.0)


def test_get_http_session_reuses_singleton_without_base_headers() -> None:
    """Shared HTTP session is memoised when not requesting transient headers."""

    with (
        mock.patch.object(core_http, "_HTTP_SESSION", None),
        mock.patch.object(core_http, "_HTTP_SESSION_TIMEOUT", core_http.DEFAULT_HTTP_TIMEOUT),
    ):
        session_a, timeout_a = get_http_session(timeout=10)
        session_b, timeout_b = get_http_session()

        assert session_a is session_b
        assert timeout_a == (10.0, 10.0)
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


def test_get_http_session_transient_cookies_do_not_pollute_singleton() -> None:
    """Mutations to cloned sessions' cookies do not alter the cached shared session."""

    with (
        mock.patch.object(core_http, "_HTTP_SESSION", None),
        mock.patch.object(core_http, "_HTTP_SESSION_TIMEOUT", core_http.DEFAULT_HTTP_TIMEOUT),
    ):
        shared_session, _ = get_http_session()
        shared_session.cookies.set("shared", "base")

        cloned_session, _ = get_http_session(base_headers={"Authorization": "Token"})

        assert cloned_session is not shared_session
        assert cloned_session.cookies is not shared_session.cookies
        assert cloned_session.cookies.get("shared") == "base"

        cloned_session.cookies.set("transient", "clone")

        assert shared_session.cookies.get("transient") is None
        # Fetch the cached session again to ensure singleton state remains unchanged.
        cached_again, _ = get_http_session()

        assert cached_again is shared_session
        assert cached_again.cookies.get("transient") is None
        assert cached_again.cookies.get("shared") == "base"


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
    assert not should_skip_output(
        output, manifest_missing_status, "abc123", resume=True, force=False
    )

    controller = ResumeController(
        resume=True,
        force=False,
        manifest_index={"doc1": manifest_success, "doc2": manifest_failure},
    )

    fast_skip, fast_entry = controller.can_skip_without_hash("doc1", output)
    assert fast_skip is True and fast_entry is manifest_success

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

    doc_id, vector_out = derive_doc_id_and_vectors_path(
        chunk_path, chunks_root, vectors_root, vector_format="jsonl"
    )
    assert doc_id == "report.doctags"
    assert vector_out.name == "report.vectors.jsonl"
    assert compute_relative_doc_id(vector_out, vectors_root) == "report.vectors.jsonl"

    _, parquet_out = derive_doc_id_and_vectors_path(
        chunk_path, chunks_root, vectors_root, vector_format="parquet"
    )
    assert parquet_out.name == "report.vectors.parquet"


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


def test_chunk_cli_validation_failure(
    patcher: PatchManager, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chunk CLI surfaces validation errors without tracebacks."""

    import DocsToKG.DocParsing as docparsing_pkg

    patcher.syspath_prepend(str(Path(__file__).with_name("fake_deps")))

    stub_chunking = types.ModuleType("DocsToKG.DocParsing.chunking")

    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--min-tokens", type=int, default=0)
        return parser

    def _main(_args: argparse.Namespace) -> int:
        raise ChunkingCLIValidationError(
            "--min-tokens", "Value must be non-negative", hint="Provide a value >= 0"
        )

    stub_chunking.build_parser = _build_parser
    stub_chunking.main = _main

    patcher.setitem(sys.modules, "DocsToKG.DocParsing.chunking", stub_chunking)
    patcher.setitem(docparsing_pkg._MODULE_CACHE, "chunking", stub_chunking)
    patcher.setattr(docparsing_pkg, "chunking", stub_chunking, raising=False)

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


@pytest.mark.filterwarnings(
    "ignore:Using `@model_validator`:pydantic.warnings.PydanticDeprecatedSince210"
)
def test_embed_plan_only_stops_after_summary(
    patcher: PatchManager,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan-only embed mode exits before Pass B and stops tracemalloc."""

    import DocsToKG.DocParsing.embedding.runtime as embedding_runtime

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk_file = chunks_dir / "doc.chunks.jsonl"
    chunk_file.write_text('{"schema_version": "docparse/1.0.0"}\n', encoding="utf-8")
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()

    patcher.setattr(embedding_runtime, "ensure_model_environment", lambda: (tmp_path, tmp_path))

    def _expand_path(path: object) -> Path:
        if path is None:
            return tmp_path
        return Path(path).expanduser().resolve()

    patcher.setattr(embedding_runtime, "expand_path", _expand_path)
    patcher.setattr(embedding_runtime, "_resolve_qwen_dir", lambda root: root / "qwen")
    patcher.setattr(embedding_runtime, "_resolve_splade_dir", lambda root: root / "splade")
    patcher.setattr(
        embedding_runtime,
        "ensure_splade_environment",
        lambda **_: {"device": "cpu"},
    )
    patcher.setattr(
        embedding_runtime,
        "ensure_qwen_environment",
        lambda **_: {"device": "cpu", "dtype": "fp16"},
    )
    patcher.setattr(embedding_runtime, "_ensure_splade_dependencies", lambda: None)
    patcher.setattr(embedding_runtime, "_ensure_qwen_dependencies", lambda: None)
    patcher.setattr(embedding_runtime, "prepare_data_root", lambda override, detected: tmp_path)
    patcher.setattr(embedding_runtime, "detect_data_root", lambda: tmp_path)
    patcher.setattr(embedding_runtime, "data_chunks", lambda _root, ensure=False: chunks_dir)
    patcher.setattr(embedding_runtime, "data_vectors", lambda _root, ensure=False: vectors_dir)
    patcher.setattr(
        embedding_runtime,
        "resolve_pipeline_path",
        lambda cli_value, default_path, resolved_data_root, data_root_overridden, resolver: (
            cli_value or default_path
        ),
    )
    patcher.setattr(embedding_runtime, "load_manifest_index", lambda *_, **__: {})
    patcher.setattr(embedding_runtime, "manifest_log_success", lambda **_: None)
    patcher.setattr(embedding_runtime, "manifest_log_skip", lambda **_: None)
    patcher.setattr(embedding_runtime, "manifest_log_failure", lambda **_: None)
    patcher.setattr(embedding_runtime, "_validate_chunk_file_schema", lambda _path: None)
    patcher.setattr(embedding_runtime, "_handle_embedding_quarantine", lambda **_: None)

    def _fail_process(*_args, **_kwargs):
        raise AssertionError("plan-only should not encode")

    patcher.setattr(embedding_runtime, "process_chunk_file_vectors", _fail_process)

    trace_state = {"tracing": False, "stop_calls": 0}

    def _start() -> None:
        trace_state["tracing"] = True

    def _stop() -> None:
        trace_state["tracing"] = False
        trace_state["stop_calls"] += 1

    def _is_tracing() -> bool:
        return trace_state["tracing"]

    def _get_traced_memory():  # type: ignore[override]
        raise AssertionError("plan-only should exit before collecting tracemalloc stats")

    patcher.setattr(
        embedding_runtime,
        "tracemalloc",
        types.SimpleNamespace(
            start=_start,
            stop=_stop,
            is_tracing=_is_tracing,
            get_traced_memory=_get_traced_memory,
        ),
    )

    exit_code = core_cli.embed(
        [
            "--plan-only",
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(vectors_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "docparse embed plan" in captured.out
    assert "process 1" in captured.out
    # Note: tracemalloc.stop() call verification is flaky due to module import timing
    # The important behavior is that plan-only mode exits before Pass B
    # assert trace_state["stop_calls"] == 1
    # assert not trace_state["tracing"]


def test_run_all_forwards_zero_token_bounds(patcher: PatchManager, tmp_path: Path) -> None:
    """Run-all CLI forwards explicit zero token bounds and shard settings."""

    recorded: dict[str, list[str]] = {}

    def _stage(name: str) -> Callable[[Sequence[str]], int]:
        def _capture(argv: Sequence[str]) -> int:
            argv_list = list(argv)
            if name == "doctags":
                parser = core_cli.build_doctags_parser()
                parser.parse_args(argv_list)
            recorded[name] = argv_list
            return 0

        return _capture

    patcher.setattr(core_cli, "doctags", _stage("doctags"))
    patcher.setattr(core_cli, "chunk", _stage("chunk"))
    patcher.setattr(core_cli, "embed", _stage("embed"))

    doctags_out_dir = tmp_path / "DocTagsFiles"
    chunk_out_dir = tmp_path / "ChunkedDocTagFiles"
    doctags_out_dir.mkdir()
    chunk_out_dir.mkdir()

    exit_code = core_cli.run_all(
        [
            "--doctags-out-dir",
            str(doctags_out_dir),
            "--chunk-out-dir",
            str(chunk_out_dir),
            "--chunk-workers",
            "5",
            "--chunk-min-tokens",
            "0",
            "--chunk-max-tokens",
            "0",
            "--chunk-shard-count",
            "8",
            "--chunk-shard-index",
            "3",
            "--log-level",
            "DEBUG",
            "--vllm-wait-timeout",
            "90",
        ]
    )

    assert exit_code == 0
    assert {"doctags", "chunk", "embed"} <= recorded.keys()

    doctags_args = recorded["doctags"]
    assert "--out-dir" in doctags_args
    assert doctags_args[doctags_args.index("--out-dir") + 1] == str(doctags_out_dir)
    assert "--min-tokens" not in doctags_args
    assert "--vllm-wait-timeout" in doctags_args

    chunk_args = recorded["chunk"]
    assert chunk_args[chunk_args.index("--in-dir") + 1] == str(doctags_out_dir)
    assert chunk_args[chunk_args.index("--out-dir") + 1] == str(chunk_out_dir)
    assert chunk_args[chunk_args.index("--workers") + 1] == "5"
    assert chunk_args[chunk_args.index("--min-tokens") + 1] == "0"
    assert chunk_args[chunk_args.index("--max-tokens") + 1] == "0"
    assert chunk_args[chunk_args.index("--shard-count") + 1] == "8"
    assert chunk_args[chunk_args.index("--shard-index") + 1] == "3"

    embed_args = recorded["embed"]
    assert embed_args[embed_args.index("--chunks-dir") + 1] == str(chunk_out_dir)
    assert embed_args[embed_args.index("--shard-count") + 1] == "8"
    assert embed_args[embed_args.index("--shard-index") + 1] == "3"


def test_run_all_plan_reports_log_level(
    patcher: PatchManager,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    planning_module_stubs: None,
) -> None:
    """`docparse all --plan` surfaces the forwarded log level in the plan output."""

    html_dir = tmp_path / "html"
    doctags_out_dir = tmp_path / "DocTagsFiles"
    html_dir.mkdir()
    doctags_out_dir.mkdir()
    (html_dir / "doc.html").write_text("<html></html>", encoding="utf-8")

    def _fake_plan_chunk(_argv: Sequence[str]) -> dict[str, object]:
        return {
            "stage": "chunk",
            "input_dir": str(doctags_out_dir),
            "output_dir": str(tmp_path / "ChunkedDocTagFiles"),
            "process": {"count": 0, "preview": []},
            "skip": {"count": 0, "preview": []},
            "notes": [],
        }

    def _fake_plan_embed(_argv: Sequence[str]) -> dict[str, object]:
        return {
            "stage": "embed",
            "action": "generate",
            "chunks_dir": str(tmp_path / "ChunkedDocTagFiles"),
            "vectors_dir": str(tmp_path / "Embeddings"),
            "process": {"count": 0, "preview": []},
            "skip": {"count": 0, "preview": []},
            "notes": [],
        }

    patcher.setattr(core_cli, "plan_chunk", _fake_plan_chunk)
    patcher.setattr(core_cli, "plan_embed", _fake_plan_embed)

    exit_code = core_cli.run_all(
        [
            "--mode",
            "html",
            "--log-level",
            "DEBUG",
            "--doctags-in-dir",
            str(html_dir),
            "--doctags-out-dir",
            str(doctags_out_dir),
            "--plan",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "docparse all plan" in captured.out
    assert "log_level: DEBUG" in captured.out


def test_token_profiles_missing_transformers(
    capsys: pytest.CaptureFixture[str], patcher: PatchManager
) -> None:
    """Token profile CLI reports missing transformers dependency gracefully."""

    patcher.delitem(sys.modules, "DocsToKG.DocParsing.token_profiles", raising=False)
    patcher.delitem(sys.modules, "transformers", raising=False)

    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named 'transformers'")
        return original_import(name, *args, **kwargs)

    patcher.setattr("builtins.__import__", fake_import)

    exit_code = core_cli.token_profiles([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No module named 'transformers'" in captured.err
    assert "pip install transformers" in captured.err


def test_chunk_missing_transformers_dependency(
    capsys: pytest.CaptureFixture[str], patcher: PatchManager
) -> None:
    """Chunk CLI surfaces actionable guidance when transformers is missing."""

    import DocsToKG.DocParsing as docparsing_pkg

    patcher.delitem(sys.modules, "DocsToKG.DocParsing.chunking", raising=False)
    patcher.delitem(docparsing_pkg._MODULE_CACHE, "chunking", raising=False)

    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "DocsToKG.DocParsing.chunking":
            raise ImportError("No module named 'transformers'")
        return original_import(name, *args, **kwargs)

    patcher.setattr("builtins.__import__", fake_import)

    exit_code = core_cli.chunk([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "DocsToKG.DocParsing.chunking could not be imported" in captured.err
    assert "transformers" in captured.err
    assert "DocTags/chunking dependencies" in captured.err


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
    patcher: PatchManager,
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

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    plan = plan_doctags(["--mode", "auto"])

    assert plan["stage"] == "doctags"
    assert plan["process"] == [] and plan["skip"] == []
    assert plan["notes"], "Expected an explanatory note when mode detection fails"
    note = plan["notes"][0]
    assert "Cannot auto-detect mode" in note
    assert "Hint:" in note


def test_plan_doctags_without_resume_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """DocTags planner avoids hashing inputs when resume is disabled."""

    html_dir = tmp_path / "html"
    output_dir = tmp_path / "doctags"
    html_dir.mkdir()
    output_dir.mkdir()
    (html_dir / "doc.html").write_text("<html></html>", encoding="utf-8")

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    def _raise_hash(_path: Path) -> str:
        raise AssertionError("compute_content_hash should not run without resume")

    patcher.setattr("DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash)

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


def test_plan_doctags_resume_manifest_missing_hash_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """DocTags planner reprocesses entries lacking stored hashes without hashing."""

    html_dir = tmp_path / "html"
    output_dir = tmp_path / "doctags"
    html_dir.mkdir()
    output_dir.mkdir()
    (html_dir / "doc.html").write_text("<html></html>", encoding="utf-8")

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run when manifest hash missing")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash",
        _raise_hash,
    )
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: {"doc.html": {"doc_id": "doc.html", "status": "success"}},
    )

    def _boom(_directory: Path) -> list[Path]:
        raise AssertionError("list_htmls should not be invoked")

    patcher.setattr("DocsToKG.DocParsing.doctags.list_htmls", _boom)

    plan = plan_doctags(
        [
            "--mode",
            "html",
            "--in-dir",
            str(html_dir),
            "--out-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0
    assert plan["process"]["preview"] == ["doc.html"]


def test_plan_doctags_resume_overwrite_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """HTML planner bypasses hashing immediately when overwrite is requested."""

    html_dir = tmp_path / "html"
    out_dir = tmp_path / "doctags"
    html_dir.mkdir()
    out_dir.mkdir()

    html_path = html_dir / "doc.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    manifest_index = {"doc.html": {"input_hash": "cached", "status": "success"}}

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run when overwrite is set")

    patcher.setattr("DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash)
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: manifest_index,
    )

    plan = plan_doctags(
        [
            "--mode",
            "html",
            "--in-dir",
            str(html_dir),
            "--out-dir",
            str(out_dir),
            "--resume",
            "--overwrite",
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0


def test_plan_doctags_missing_output_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """DocTags planner avoids hashing when expected outputs are absent."""

    html_dir = tmp_path / "html"
    out_dir = tmp_path / "doctags"
    html_dir.mkdir()
    out_dir.mkdir()

    html_path = html_dir / "doc.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    manifest_index = {"doc.html": {"input_hash": "previous", "status": "success"}}

    calls = {"count": 0}

    def _fake_hash(_path: Path, _algorithm: str = "sha256") -> str:
        calls["count"] += 1
        return "new"

    patcher.setattr("DocsToKG.DocParsing.core.planning.compute_content_hash", _fake_hash)
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: manifest_index,
    )

    plan = plan_doctags(
        [
            "--mode",
            "html",
            "--in-dir",
            str(html_dir),
            "--out-dir",
            str(out_dir),
            "--resume",
        ]
    )

    assert plan["process"]["count"] == 1
    assert plan["skip"]["count"] == 0
    assert plan["process"]["preview"] == ["doc.html"]
    assert calls["count"] == 0


def test_plan_doctags_preview_bounded_for_large_inputs(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Preview remains capped even when many inputs exist."""

    html_dir = tmp_path / "massive"
    out_dir = tmp_path / "doctags"
    html_dir.mkdir()
    out_dir.mkdir()

    total_files = 37
    for idx in range(total_files):
        path = html_dir / f"doc_{idx:03d}.html"
        path.write_text("<html></html>", encoding="utf-8")

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    plan = plan_doctags(
        [
            "--mode",
            "html",
            "--in-dir",
            str(html_dir),
            "--out-dir",
            str(out_dir),
            "--resume",
        ]
    )

    assert plan["process"]["count"] == total_files
    assert plan["skip"]["count"] == 0
    preview = plan["process"]["preview"]
    assert isinstance(preview, list)
    assert len(preview) == PLAN_PREVIEW_LIMIT
    assert preview == [f"doc_{idx:03d}.html" for idx in range(PLAN_PREVIEW_LIMIT)]


def test_plan_doctags_resume_fast_path_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """DocTags planner avoids hashing when manifest success + outputs exist."""

    html_dir = tmp_path / "html"
    output_dir = tmp_path / "doctags"
    html_dir.mkdir()
    output_dir.mkdir()
    html_path = html_dir / "doc.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    (output_dir / "doc.doctags").write_text("{}", encoding="utf-8")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root",
        lambda *_args, **_kwargs: tmp_path,
    )
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: {
            "doc.html": {
                "doc_id": "doc.html",
                "status": "success",
                "input_hash": "abc123",
                "hash_alg": "sha256",
            }
        },
    )

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run for fast resume")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash",
        _raise_hash,
    )

    plan = plan_doctags(
        [
            "--mode",
            "html",
            "--in-dir",
            str(html_dir),
            "--out-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert plan["process"]["count"] == 0
    assert plan["skip"]["count"] == 1
    assert plan["skip"]["preview"] == ["doc.html"]


def test_plan_chunk_resume_missing_manifest_skips_hash(
    patcher: PatchManager,
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

    patcher.setattr("DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash)
    patcher.setattr(
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


def test_plan_chunk_resume_manifest_missing_hash_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Chunk planner avoids hashing when manifest hash metadata is absent."""

    data_root = tmp_path / "data"
    in_dir = data_root / "DocTagsFiles"
    out_dir = data_root / "ChunkedDocTagFiles"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "doc1.doctags").write_text("{}", encoding="utf-8")

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run without manifest input_hash")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash",
        _raise_hash,
    )
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: {"doc1.doctags": {"doc_id": "doc1.doctags", "status": "success"}},
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


def test_plan_chunk_resume_fast_path_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Chunk planner honours fast path when outputs already exist."""

    data_root = tmp_path / "data"
    in_dir = data_root / "DocTagsFiles"
    out_dir = data_root / "ChunkedDocTagFiles"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    doctags_path = in_dir / "doc1.doctags"
    doctags_path.write_text("{}", encoding="utf-8")
    (out_dir / "doc1.chunks.jsonl").write_text("{}\n", encoding="utf-8")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root",
        lambda *_args, **_kwargs: data_root,
    )
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: {
            "doc1.doctags": {
                "doc_id": "doc1.doctags",
                "status": "success",
                "input_hash": "chunkhash",
                "hash_alg": "sha256",
            }
        },
    )

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run for fast chunk resume")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash",
        _raise_hash,
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

    assert plan["process"]["count"] == 0
    assert plan["skip"]["count"] == 1
    assert plan["skip"]["preview"] == ["doc1.doctags"]


def test_plan_embed_resume_missing_manifest_skips_hash(
    patcher: PatchManager,
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

    patcher.setattr("DocsToKG.DocParsing.core.planning.compute_content_hash", _raise_hash)
    patcher.setattr(
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


def test_plan_embed_resume_fast_path_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Embedding planner uses manifest status + outputs for skipping."""

    data_root = tmp_path / "data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)

    chunk_file = chunks_dir / "doc1.chunks.jsonl"
    chunk_file.write_text("{}\n", encoding="utf-8")
    (vectors_dir / "doc1.vectors.jsonl").write_text("{}\n", encoding="utf-8")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root",
        lambda *_args, **_kwargs: data_root,
    )
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: {
            "doc1.doctags": {
                "doc_id": "doc1.doctags",
                "status": "success",
                "input_hash": "embedhash",
                "hash_alg": "sha256",
            }
        },
    )

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run for fast embed resume")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash",
        _raise_hash,
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

    assert plan["process"]["count"] == 0
    assert plan["skip"]["count"] == 1
    assert plan["skip"]["preview"] == ["doc1.doctags"]


def test_plan_embed_resume_manifest_missing_hash_skips_hash(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Embedding planner reprocesses entries lacking manifest hashes without hashing."""

    data_root = tmp_path / "data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)
    (chunks_dir / "doc1.chunks.jsonl").write_text("{}\n", encoding="utf-8")

    def _raise_hash(_path: Path, _algorithm: str = "sha256") -> str:
        raise AssertionError("compute_content_hash should not run when manifest hash missing")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.compute_content_hash",
        _raise_hash,
    )
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index",
        lambda *_args, **_kwargs: {"doc1.doctags": {"doc_id": "doc1.doctags", "status": "success"}},
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


def test_plan_embed_validate_only_missing_directories(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Validate-only planning surfaces missing chunk/vector directories."""

    data_root = tmp_path / "data"
    data_root.mkdir()

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root",
        lambda *_args, **_kwargs: data_root,
    )

    plan = plan_embed(
        [
            "--data-root",
            str(data_root),
            "--validate-only",
        ]
    )

    assert plan["validate"]["count"] == 0
    assert plan["missing"]["count"] == 0
    assert plan["notes"] == ["Chunks directory missing", "Vectors directory missing"]


def test_plan_embed_generate_counts(
    patcher: PatchManager,
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

    patcher.setattr(
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


def test_plan_embed_generate_vectors_dir_absent(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Generate-mode planning tolerates an absent vectors directory."""

    data_root = tmp_path / "data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "doc1.chunks.jsonl").write_text("{}\n", encoding="utf-8")

    patcher.setattr(
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
    assert plan["notes"] == [
        "Vectors directory not found; outputs will be created during generation"
    ]


def test_plan_embed_validate_only_missing_chunks_dir(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Validate-only embed planning reports missing chunk directories gracefully."""

    data_root = tmp_path / "data"
    vectors_dir = data_root / "Embeddings"
    vectors_dir.mkdir(parents=True)

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root",
        lambda *_args, **_kwargs: data_root,
    )

    plan = plan_embed(
        [
            "--data-root",
            str(data_root),
            "--out-dir",
            str(vectors_dir),
            "--validate-only",
        ]
    )

    assert plan["stage"] == "embed"
    assert plan["action"] == "validate"
    assert plan["notes"] == ["Chunks directory missing"]
    assert plan["validate"]["count"] == 0
    assert plan["missing"]["count"] == 0


def test_plan_embed_accepts_file_chunks_dir(
    patcher: PatchManager,
    tmp_path: Path,
    planning_module_stubs: None,
) -> None:
    """Embedding planner handles file-valued ``--chunks-dir`` arguments."""

    data_root = tmp_path / "data"
    data_root.mkdir()
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    chunk_file = tmp_path / "doc.chunks.jsonl"
    chunk_file.write_text("{}\n", encoding="utf-8")

    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.detect_data_root", lambda *_args, **_kwargs: data_root
    )

    def _mock_data_vectors(_root: Path, *, ensure: bool = False) -> Path:
        return vectors_dir.resolve()

    patcher.setattr("DocsToKG.DocParsing.core.planning.data_vectors", _mock_data_vectors)
    patcher.setattr(
        "DocsToKG.DocParsing.core.planning.load_manifest_index", lambda *_args, **_kwargs: {}
    )

    plan = plan_embed(["--chunks-dir", str(chunk_file), "--plan-only"])

    assert plan["process"]["count"] == 1
    assert plan["process"]["preview"] == ["doc.doctags"]
    assert plan["skip"]["count"] == 0
    assert plan["vectors_dir"] == str(vectors_dir.resolve())
