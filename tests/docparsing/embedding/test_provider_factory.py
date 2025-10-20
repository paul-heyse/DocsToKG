from __future__ import annotations

from pathlib import Path

import pytest

from DocsToKG.DocParsing.embedding.backends import (
    NullDenseProvider,
    NullLexicalProvider,
    NullSparseProvider,
    ProviderError,
    ProviderFactory,
)
from DocsToKG.DocParsing.embedding.config import EmbedCfg


def _base_embed_cfg(tmp_path: Path) -> EmbedCfg:
    cfg = EmbedCfg()
    cfg.data_root = tmp_path
    cfg.chunks_dir = tmp_path / "chunks"
    cfg.out_dir = tmp_path / "vectors"
    cfg.chunks_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.embedding_device = "cpu"
    cfg.embedding_dtype = "float32"
    return cfg


def test_factory_dense_fallback_to_tei(tmp_path: Path) -> None:
    cfg = _base_embed_cfg(tmp_path)
    cfg.dense_fallback_backend = "tei"
    cfg.dense_tei_url = "https://example.test/embed"
    cfg.finalize()
    cfg.dense_backend = "qwen_vllm"
    cfg.dense_qwen_vllm_download_dir = None
    cfg.sparse_backend = "none"

    bundle = ProviderFactory.create(cfg)
    assert bundle.dense is not None
    assert bundle.dense.identity.name == "dense.tei"
    assert isinstance(bundle.sparse, NullSparseProvider)


def test_factory_dense_none_returns_null_provider(tmp_path: Path) -> None:
    cfg = _base_embed_cfg(tmp_path)
    cfg.finalize()
    cfg.dense_backend = "none"
    cfg.sparse_backend = "none"
    cfg.lexical_backend = "none"

    bundle = ProviderFactory.create(cfg)
    assert isinstance(bundle.dense, NullDenseProvider)
    assert isinstance(bundle.sparse, NullSparseProvider)
    assert isinstance(bundle.lexical, NullLexicalProvider)


def test_factory_lexical_pyserini_placeholder(tmp_path: Path) -> None:
    cfg = _base_embed_cfg(tmp_path)
    cfg.finalize()
    cfg.dense_backend = "none"
    cfg.sparse_backend = "none"
    cfg.lexical_backend = "pyserini"

    bundle = ProviderFactory.create(cfg)
    assert bundle.lexical.identity.name == "lexical.pyserini"


def test_provider_bundle_raises_with_null_dense(tmp_path: Path) -> None:
    cfg = _base_embed_cfg(tmp_path)
    cfg.dense_backend = "none"
    cfg.sparse_backend = "none"
    cfg.lexical_backend = "local_bm25"
    cfg.finalize()

    bundle = ProviderFactory.create(cfg)
    with pytest.raises(ProviderError) as excinfo:
        with bundle:
            pass
    assert excinfo.value.provider == "dense.none"
