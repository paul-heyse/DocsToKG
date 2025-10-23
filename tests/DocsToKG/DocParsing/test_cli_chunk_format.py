"""Tests for chunk format propagation through CLI and configuration adapter."""

from __future__ import annotations

from typer.testing import CliRunner

from tests.DocsToKG.DocParsing._stubs import install_all_stubs


install_all_stubs()

from DocsToKG.DocParsing.app_context import build_app_context
from DocsToKG.DocParsing.chunking import runtime as chunking_runtime
from DocsToKG.DocParsing.embedding import runtime as embedding_runtime
from DocsToKG.DocParsing.cli_unified import app
from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
from DocsToKG.DocParsing.settings import Format

runner = CliRunner()


def test_configuration_adapter_to_chunk_respects_format_jsonl(monkeypatch, tmp_path):
    """Ensure the chunk adapter reflects the settings format override."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    app_ctx = build_app_context(chunk_format="jsonl")

    assert app_ctx.settings.chunk.format is Format.JSONL

    cfg = ConfigurationAdapter.to_chunk(app_ctx)

    assert cfg.format == "jsonl"


def test_cli_chunk_format_option_propagates(monkeypatch, tmp_path):
    """Passing --format jsonl should flip the stage configuration."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    input_dir = tmp_path / "DocTagsFiles"
    output_dir = tmp_path / "ChunkedDocTagFiles"
    input_dir.mkdir()
    output_dir.mkdir()

    captured: dict[str, object] = {}

    def fake_main_inner(args=None, config_adapter=None):  # type: ignore[override]
        captured["config"] = config_adapter
        return 0

    monkeypatch.setattr(chunking_runtime, "_main_inner", fake_main_inner)

    result = runner.invoke(
        app,
        [
            "chunk",
            "--in-dir",
            str(input_dir),
            "--out-dir",
            str(output_dir),
            "--format",
            "jsonl",
        ],
    )

    assert "config" in captured

    config = captured["config"]

    assert config.format == "jsonl"
    assert "✅ Chunk stage completed successfully" in result.stdout


def test_cli_chunk_no_resume_flag_propagates(monkeypatch, tmp_path):
    """The --no-resume flag should disable resume in the chunk config."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    input_dir = tmp_path / "DocTagsFiles"
    output_dir = tmp_path / "ChunkedDocTagFiles"
    input_dir.mkdir()
    output_dir.mkdir()

    captured: dict[str, object] = {}

    def fake_main_inner(args=None, config_adapter=None):  # type: ignore[override]
        captured["config"] = config_adapter
        return 0

    monkeypatch.setattr(chunking_runtime, "_main_inner", fake_main_inner)

    result = runner.invoke(
        app,
        [
            "chunk",
            "--in-dir",
            str(input_dir),
            "--out-dir",
            str(output_dir),
            "--no-resume",
        ],
    )

    assert "config" in captured
    assert getattr(captured["config"], "resume") is False
    assert "✅ Chunk stage completed successfully" in result.stdout


def test_cli_embed_no_resume_flag_propagates(monkeypatch, tmp_path):
    """The embed command should pass --no-resume through to the stage config."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    chunks_dir = tmp_path / "ChunkedDocTagFiles"
    vectors_dir = tmp_path / "Embeddings"
    chunks_dir.mkdir()
    vectors_dir.mkdir()

    captured: dict[str, object] = {}

    def fake_embed_main(*, config_adapter=None, **_kwargs):  # type: ignore[override]
        captured["config"] = config_adapter
        return 0

    monkeypatch.setattr(embedding_runtime, "_main_inner", fake_embed_main)

    result = runner.invoke(
        app,
        [
            "embed",
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(vectors_dir),
            "--no-resume",
        ],
    )

    assert "config" in captured
    assert getattr(captured["config"], "resume") is False
    assert "✅ Embed stage completed successfully" in result.stdout
