"""Tests for DocParsing environment helpers."""

from __future__ import annotations

import os
from pathlib import Path

from DocsToKG.DocParsing import env
from DocsToKG.DocParsing.embedding import runtime


def test_ensure_splade_environment_sets_dir_env(monkeypatch, tmp_path):
    """SPLADE cache directory seeds both supported environment variables."""

    monkeypatch.delenv("DOCSTOKG_SPLADE_DIR", raising=False)
    monkeypatch.delenv("DOCSTOKG_SPLADE_MODEL_DIR", raising=False)

    cache_dir = tmp_path / "splade-cache"
    resolved = env.ensure_splade_environment(cache_dir=cache_dir)

    expected = str(cache_dir.resolve())
    assert os.environ["DOCSTOKG_SPLADE_DIR"] == expected
    assert os.environ["DOCSTOKG_SPLADE_MODEL_DIR"] == expected
    assert resolved["model_dir"] == expected


def test_ensure_splade_environment_override_even_when_populated(monkeypatch, tmp_path):
    """Providing ``cache_dir`` should override any pre-existing values."""

    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(tmp_path / "old"))
    monkeypatch.setenv("DOCSTOKG_SPLADE_MODEL_DIR", str(tmp_path / "legacy"))

    cache_dir = tmp_path / "new-cache"
    resolved = env.ensure_splade_environment(cache_dir=cache_dir)

    expected = str(cache_dir.resolve())
    assert os.environ["DOCSTOKG_SPLADE_DIR"] == expected
    assert os.environ["DOCSTOKG_SPLADE_MODEL_DIR"] == expected
    assert resolved["model_dir"] == expected

    model_root = tmp_path / "model-root"
    resolved_dir = runtime._resolve_splade_dir(model_root)
    assert resolved_dir == cache_dir.resolve()


def test_resolve_splade_dir_prefers_env(monkeypatch, tmp_path):
    """``_resolve_splade_dir`` should prioritise ``DOCSTOKG_SPLADE_DIR``."""

    cache_dir = tmp_path / "configured"
    monkeypatch.setenv("DOCSTOKG_SPLADE_DIR", str(cache_dir))

    model_root = Path(tmp_path / "model-root")
    resolved = runtime._resolve_splade_dir(model_root)

    assert resolved == cache_dir.resolve()
