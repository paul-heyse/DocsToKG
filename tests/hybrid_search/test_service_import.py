"""Validate HybridSearch service import initialises FAISS when available."""

import sys

import pytest


def test_hybrid_service_import_initialises_faiss():
    """Ensure service import path exposes the FAISS module when installed."""

    pytest.importorskip("faiss")

    module_name = "DocsToKG.HybridSearch.service"
    sys.modules.pop(module_name, None)

    from DocsToKG.HybridSearch.service import HybridSearchService  # noqa: F401

    service_module = sys.modules[module_name]
    assert service_module.faiss is not None
