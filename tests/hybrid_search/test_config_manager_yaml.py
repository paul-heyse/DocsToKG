"""Smoke test covering YAML config loading for HybridSearch."""

from __future__ import annotations

import textwrap

import pytest

from DocsToKG.HybridSearch.config import HybridSearchConfigManager


def test_hybrid_search_config_manager_loads_yaml(tmp_path):
    """Ensure YAML configs round-trip through ``HybridSearchConfigManager``."""

    config_path = tmp_path / "hybrid_search.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            chunking:
              max_tokens: 512
              overlap: 64
            dense:
              index_type: ivf_flat
              nprobe: 16
            fusion:
              channel_weights:
                bm25: 1.0
                splade: 1.0
                dense: 1.5
            retrieval:
              bm25_top_k: 40
              dense_top_k: 25
            """
        ).strip()
    )

    manager = HybridSearchConfigManager(config_path)
    config = manager.get()

    assert config.chunking.max_tokens == 512
    assert config.chunking.overlap == 64
    assert config.dense.index_type == "ivf_flat"
    assert config.dense.nprobe == 16
    assert config.fusion.channel_weights["dense"] == 1.5
    assert config.retrieval.dense_top_k == 25

    reloaded = manager.reload()
    assert reloaded.dense.nprobe == 16
    assert reloaded.fusion.channel_weights == config.fusion.channel_weights


def test_hybrid_search_config_manager_invalid_yaml(tmp_path):
    """Invalid YAML surfaces a ValueError with file path context."""

    config_path = tmp_path / "hybrid_search.yaml"
    config_path.write_text("chunking: [\n  - max_tokens: 512\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        HybridSearchConfigManager(config_path)

    message = str(excinfo.value)
    assert str(config_path) in message
    assert "Failed to parse YAML configuration" in message
