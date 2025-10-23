"""Tests for embed CLI overrides and configuration propagation."""

from __future__ import annotations

from typer.testing import CliRunner

from tests.DocsToKG.DocParsing._stubs import install_all_stubs


install_all_stubs()

from DocsToKG.DocParsing.app_context import build_app_context
from DocsToKG.DocParsing.cli_unified import app
from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
from DocsToKG.DocParsing.embedding import runtime as embedding_runtime
from DocsToKG.DocParsing.settings import DenseBackend, Format


runner = CliRunner()


def test_configuration_adapter_to_embed_respects_vector_format_jsonl(monkeypatch, tmp_path):
    """Ensure the embed adapter reflects the vector format override."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    app_ctx = build_app_context(embed_vector_format="jsonl")

    assert app_ctx.settings.embed.vectors.format is Format.JSONL

    cfg = ConfigurationAdapter.to_embed(app_ctx)

    assert cfg.vector_format == "jsonl"


def test_configuration_adapter_to_embed_respects_dense_backend_override(monkeypatch, tmp_path):
    """Ensure the embed adapter reflects the dense backend override."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    app_ctx = build_app_context(**{"embed.dense.backend": "sentence_transformers"})

    assert (
        app_ctx.settings.embed.dense.backend is DenseBackend.SENTENCE_TRANSFORMERS
    )

    cfg = ConfigurationAdapter.to_embed(app_ctx)

    assert cfg.dense_backend == "sentence_transformers"


def test_cli_embed_format_option_propagates(monkeypatch, tmp_path):
    """Passing --format jsonl should flip the embed stage configuration."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    chunks_dir = tmp_path / "ChunkedDocTagFiles"
    output_dir = tmp_path / "Embeddings"
    chunks_dir.mkdir()
    output_dir.mkdir()

    captured: dict[str, object] = {}

    def fake_main_inner(args=None, config_adapter=None):  # type: ignore[override]
        captured["config"] = config_adapter
        return 0

    monkeypatch.setattr(embedding_runtime, "_main_inner", fake_main_inner)

    result = runner.invoke(
        app,
        [
            "embed",
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(output_dir),
            "--format",
            "jsonl",
        ],
    )

    assert "config" in captured

    config = captured["config"]

    assert getattr(config, "vector_format") == "jsonl"
    assert "✅ Embed stage completed successfully" in result.stdout


def test_cli_embed_dense_backend_option_propagates(monkeypatch, tmp_path):
    """Passing --dense-backend should update the runtime configuration."""

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))

    chunks_dir = tmp_path / "ChunkedDocTagFiles"
    output_dir = tmp_path / "Embeddings"
    chunks_dir.mkdir()
    output_dir.mkdir()

    captured: dict[str, object] = {}

    def fake_main_inner(args=None, config_adapter=None):  # type: ignore[override]
        captured["config"] = config_adapter
        return 0

    monkeypatch.setattr(embedding_runtime, "_main_inner", fake_main_inner)

    result = runner.invoke(
        app,
        [
            "embed",
            "--chunks-dir",
            str(chunks_dir),
            "--out-dir",
            str(output_dir),
            "--dense-backend",
            "sentence_transformers",
        ],
    )

    assert "config" in captured

    config = captured["config"]

    assert getattr(config, "dense_backend") == "sentence_transformers"
    assert "✅ Embed stage completed successfully" in result.stdout
