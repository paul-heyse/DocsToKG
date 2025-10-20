"""Logging behaviour regression tests for ``DocsToKG.HybridSearch.store``."""

from __future__ import annotations

import logging

import numpy as np

from DocsToKG.HybridSearch import store as store_module
from tests.hybrid_search.test_gpu_similarity import _make_faiss_stub


def test_cosine_topk_blockwise_logs_info_once_per_block_size(patcher, caplog):
    stub, _ = _make_faiss_stub(enable_knn=True)
    patcher.setattr(store_module, "faiss", stub, raising=False)
    patcher.setattr(store_module, "_COSINE_TOPK_LAST_BLOCK_ROWS", {})

    caplog.set_level(logging.DEBUG, logger=store_module.logger.name)

    q = np.random.default_rng(123).random((2, 4), dtype=np.float32)
    C = np.random.default_rng(321).random((8, 4), dtype=np.float32)

    store_module.cosine_topk_blockwise(
        q,
        C,
        k=3,
        device=0,
        resources=object(),
        block_rows=4,
    )
    store_module.cosine_topk_blockwise(
        q,
        C,
        k=3,
        device=0,
        resources=object(),
        block_rows=4,
    )

    matching_records = [
        record
        for record in caplog.records
        if record.name == store_module.logger.name
        and record.message == "cosine-topk-block-config"
    ]
    info_records = [rec for rec in matching_records if rec.levelno == logging.INFO]
    debug_records = [rec for rec in matching_records if rec.levelno == logging.DEBUG]

    assert len(info_records) == 1, "expected a single info log for the first block sizing"
    assert len(debug_records) == 1, "subsequent identical sizing should fall back to debug"
    assert info_records[0].event == debug_records[0].event == {
        "action": "cosine_topk_blockwise",
        "auto_block_rows": False,
        "block_rows": 4,
        "device": 0,
        "dim": 4,
        "requested_block_rows": 4,
    }
