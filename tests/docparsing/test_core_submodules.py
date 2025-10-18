"""Unit tests covering refactored DocParsing core submodules."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

import DocsToKG.DocParsing.core.http as core_http
from DocsToKG.DocParsing.cli_errors import (
    ChunkingCLIValidationError,
    EmbeddingCLIValidationError,
)
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
from DocsToKG.DocParsing.core.planning import display_plan
from DocsToKG.DocParsing.config import ConfigLoadError, load_toml_markers, load_yaml_markers


def test_normalize_http_timeout_scalar_and_iterables() -> None:
    """HTTP timeout coercion handles scalars, strings, and iterables."""

    assert normalize_http_timeout(None) == (5.0, 30.0)
    assert normalize_http_timeout(12) == (5.0, 12.0)
    assert normalize_http_timeout("1.5, 2.5") == (1.5, 2.5)
    assert normalize_http_timeout([2, 3]) == (2.0, 3.0)


def test_get_http_session_reuses_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared HTTP session is memoised and merges headers."""

    monkeypatch.setattr(core_http, "_HTTP_SESSION", None)
    monkeypatch.setattr(core_http, "_HTTP_SESSION_TIMEOUT", core_http.DEFAULT_HTTP_TIMEOUT)

    session_a, timeout_a = get_http_session(timeout=10, base_headers={"X-Test": "one"})
    session_b, timeout_b = get_http_session(base_headers={"X-Other": "two"})

    assert session_a is session_b
    assert timeout_a == (5.0, 10.0)
    assert timeout_b == (5.0, 30.0)
    assert session_b.headers["X-Test"] == "one"
    assert session_b.headers["X-Other"] == "two"


def test_manifest_resume_controller(tmp_path: Path) -> None:
    """ResumeController mirrors should_skip_output behaviour."""

    output = tmp_path / "out.jsonl"
    manifest_entry = {"input_hash": "abc123"}

    assert not should_skip_output(output, manifest_entry, "abc123", resume=True, force=True)

    output.write_text("data", encoding="utf-8")
    assert should_skip_output(output, manifest_entry, "abc123", resume=True, force=False)

    controller = ResumeController(resume=True, force=False, manifest_index={"doc1": manifest_entry})
    skip, entry = controller.should_skip("doc1", output, "abc123")
    assert skip is True and entry is manifest_entry

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


def test_load_toml_markers_failure(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "process": ["a"],
            "skip": [],
            "input_dir": "/input/doc",
            "output_dir": "/output/doc",
            "notes": ["Ready"],
        },
        {
            "stage": "embed",
            "action": "validate",
            "validate": ["x"],
            "missing": [],
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
    assert any("embed (validate-only)" in line for line in rendered)
