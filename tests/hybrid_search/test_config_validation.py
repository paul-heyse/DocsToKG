"""Unit tests for `HybridSearchConfig` validation helpers."""

import pytest

from DocsToKG.HybridSearch.config import HybridSearchConfig


def test_hybrid_search_config_from_dict_rejects_non_mapping() -> None:
    """`HybridSearchConfig.from_dict` should reject non-mapping payloads."""
    with pytest.raises(ValueError) as exc_info:
        HybridSearchConfig.from_dict([])  # type: ignore[arg-type]

    assert "mapping payload" in str(exc_info.value)
