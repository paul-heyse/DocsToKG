# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_llm_caching",
#   "purpose": "Pytest coverage for docparsing llm caching scenarios",
#   "sections": [
#     {
#       "id": "test_qwen_cache_key_generation",
#       "name": "test_qwen_cache_key_generation",
#       "anchor": "TQCKG",
#       "kind": "function"
#     },
#     {
#       "id": "test_qwen_cache_initialization",
#       "name": "test_qwen_cache_initialization",
#       "anchor": "TQCI",
#       "kind": "function"
#     },
#     {
#       "id": "test_qwen_cache_functionality",
#       "name": "test_qwen_cache_functionality",
#       "anchor": "TQCF",
#       "kind": "function"
#     },
#     {
#       "id": "test_qwen_cache_key_uniqueness",
#       "name": "test_qwen_cache_key_uniqueness",
#       "anchor": "TQCKU",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test Qwen LLM caching effectiveness and output consistency."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from DocsToKG.DocParsing.EmbeddingV2 import _QWEN_LLM_CACHE, _qwen_cache_key


def test_qwen_cache_key_generation():
    """Test that cache keys are generated correctly for different configurations."""
    from DocsToKG.DocParsing.EmbeddingV2 import QwenCfg

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test configurations
        cfg1 = QwenCfg(
            model_dir=tmp_path / "model1",
            dtype="float16",
            tp=1,
            gpu_mem_util=0.5,
            quantization=None,
        )

        cfg2 = QwenCfg(
            model_dir=tmp_path / "model2",
            dtype="float16",
            tp=1,
            gpu_mem_util=0.5,
            quantization=None,
        )

        cfg3 = QwenCfg(
            model_dir=tmp_path / "model1",  # Same as cfg1
            dtype="bfloat16",  # Different dtype
            tp=1,
            gpu_mem_util=0.5,
            quantization=None,
        )

        # Generate cache keys
        key1 = _qwen_cache_key(cfg1)
        key2 = _qwen_cache_key(cfg2)
        key3 = _qwen_cache_key(cfg3)

        # Verify keys are different for different configs
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Verify key structure
        assert len(key1) == 5
        assert isinstance(key1[0], str)  # model_dir
        assert isinstance(key1[1], str)  # dtype
        assert isinstance(key1[2], int)  # tp
        assert isinstance(key1[3], float)  # gpu_mem_util
        assert key1[4] is None  # quantization


def test_qwen_cache_initialization():
    """Test that the Qwen LLM cache is properly initialized."""
    # Clear cache before test
    _QWEN_LLM_CACHE.clear()

    assert len(_QWEN_LLM_CACHE) == 0
    assert isinstance(_QWEN_LLM_CACHE, dict)


def test_qwen_cache_functionality():
    """Test that Qwen cache key generation and storage works correctly."""
    from DocsToKG.DocParsing.EmbeddingV2 import QwenCfg

    # Clear cache before test
    _QWEN_LLM_CACHE.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        cfg = QwenCfg(
            model_dir=tmp_path / "model",
            dtype="float16",
            tp=1,
            gpu_mem_util=0.5,
            quantization=None,
        )

        # Test cache key generation
        cache_key = _qwen_cache_key(cfg)
        assert isinstance(cache_key, tuple)
        assert len(cache_key) == 5

        # Test cache storage
        mock_llm = Mock()
        _QWEN_LLM_CACHE[cache_key] = mock_llm

        # Verify cache retrieval
        retrieved_llm = _QWEN_LLM_CACHE.get(cache_key)
        assert retrieved_llm == mock_llm

        # Test cache miss
        different_cfg = QwenCfg(
            model_dir=tmp_path / "different_model",
            dtype="float16",
            tp=1,
            gpu_mem_util=0.5,
            quantization=None,
        )

        different_key = _qwen_cache_key(different_cfg)
        assert different_key != cache_key

        retrieved_different = _QWEN_LLM_CACHE.get(different_key)
        assert retrieved_different is None


def test_qwen_cache_key_uniqueness():
    """Test that different configurations generate unique cache keys."""
    from DocsToKG.DocParsing.EmbeddingV2 import QwenCfg

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create configurations with different parameters
        configs = [
            QwenCfg(
                model_dir=tmp_path / "model1",
                dtype="float16",
                tp=1,
                gpu_mem_util=0.5,
                quantization=None,
            ),
            QwenCfg(
                model_dir=tmp_path / "model2",
                dtype="float16",
                tp=1,
                gpu_mem_util=0.5,
                quantization=None,
            ),
            QwenCfg(
                model_dir=tmp_path / "model1",
                dtype="bfloat16",
                tp=1,
                gpu_mem_util=0.5,
                quantization=None,
            ),
            QwenCfg(
                model_dir=tmp_path / "model1",
                dtype="float16",
                tp=2,
                gpu_mem_util=0.5,
                quantization=None,
            ),
            QwenCfg(
                model_dir=tmp_path / "model1",
                dtype="float16",
                tp=1,
                gpu_mem_util=0.8,
                quantization=None,
            ),
            QwenCfg(
                model_dir=tmp_path / "model1",
                dtype="float16",
                tp=1,
                gpu_mem_util=0.5,
                quantization="awq",
            ),
        ]

        # Generate cache keys
        keys = [_qwen_cache_key(cfg) for cfg in configs]

        # Verify all keys are unique
        assert len(keys) == len(set(keys))

        # Verify key structure consistency
        for key in keys:
            assert isinstance(key, tuple)
            assert len(key) == 5
            assert isinstance(key[0], str)  # model_dir
            assert isinstance(key[1], str)  # dtype
            assert isinstance(key[2], int)  # tp
            assert isinstance(key[3], float)  # gpu_mem_util
            # quantization can be None or string


if __name__ == "__main__":
    pytest.main([__file__])
