"""Tests covering lazy optional dependency loading in embedding runtime."""

from __future__ import annotations

import builtins
import sys

import pytest

sys.modules.pop("DocsToKG.DocParsing.embedding", None)

import DocsToKG.DocParsing.embedding.runtime as embedding_runtime


def test_sparse_encoder_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling the loader surfaces an actionable error when dependency absent."""

    monkeypatch.setattr(embedding_runtime, "_SPARSE_ENCODER_CLS", None)
    monkeypatch.setattr(embedding_runtime, "ensure_splade_dependencies", lambda: None)
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("sentence_transformers missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as excinfo:
        embedding_runtime._get_sparse_encoder_cls()
    assert "sentence-transformers" in str(excinfo.value)


def test_vllm_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen helper reports missing vLLM dependencies cleanly."""

    monkeypatch.setattr(embedding_runtime, "_VLLM_COMPONENTS", None)
    monkeypatch.setattr(embedding_runtime, "ensure_qwen_dependencies", lambda: None)
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError("vllm missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as excinfo:
        embedding_runtime._get_vllm_components()
    assert "vLLM" in str(excinfo.value)
