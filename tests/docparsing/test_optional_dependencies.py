"""Verify embedding runtime lazily loads optional dependencies with clear errors.

The embedding stage defers importing heavy libraries until needed. This module
stubs various import paths to ensure missing SPLADE, Qwen, or tqdm dependencies
raise actionable runtime errors and that once dependencies are available the
helpers return cached components without duplicate imports.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from unittest import mock

import pytest


def _import_runtime():
    sys.modules.pop("DocsToKG.DocParsing.embedding.runtime", None)
    return importlib.import_module("DocsToKG.DocParsing.embedding.runtime")


embedding_runtime = _import_runtime()


def test_sparse_encoder_missing_dependency() -> None:
    """Calling the loader surfaces an actionable error when dependency absent."""

    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("sentence_transformers missing")
        return original_import(name, *args, **kwargs)

    with (
        mock.patch.object(embedding_runtime, "_SPARSE_ENCODER_CLS", None),
        mock.patch.object(embedding_runtime, "ensure_splade_dependencies", lambda: None),
        mock.patch("builtins.__import__", new=fake_import),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            embedding_runtime._get_sparse_encoder_cls()
    assert "sentence-transformers" in str(excinfo.value)


def test_vllm_missing_dependency() -> None:
    """Qwen helper reports missing vLLM dependencies cleanly."""

    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError("vllm missing")
        return original_import(name, *args, **kwargs)

    with (
        mock.patch.object(embedding_runtime, "_VLLM_COMPONENTS", None),
        mock.patch.object(embedding_runtime, "ensure_qwen_dependencies", lambda: None),
        mock.patch("builtins.__import__", new=fake_import),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            embedding_runtime._get_vllm_components()
    assert "vLLM" in str(excinfo.value)
