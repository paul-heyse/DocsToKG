"""Unit tests for `HybridSearchConfig` validation helpers."""

import pytest

from DocsToKG.HybridSearch.config import HybridSearchConfig, RetrievalConfig


def test_hybrid_search_config_from_dict_rejects_non_mapping() -> None:
    """`HybridSearchConfig.from_dict` should reject non-mapping payloads."""
    with pytest.raises(ValueError) as exc_info:
        HybridSearchConfig.from_dict([])  # type: ignore[arg-type]

    assert "mapping payload" in str(exc_info.value)


def test_retrieval_config_executor_max_workers_must_be_positive() -> None:
    """executor_max_workers must be a positive integer when provided."""

    with pytest.raises(ValueError):
        RetrievalConfig(executor_max_workers=0)


def test_retrieval_config_executor_max_workers_requires_int() -> None:
    """executor_max_workers must reject non-integer values."""

    with pytest.raises(TypeError):
        RetrievalConfig(executor_max_workers=3.5)  # type: ignore[arg-type]
