from DocsToKG.HybridSearch.config import DenseIndexConfig, RetrievalConfig
from DocsToKG.HybridSearch.service import DenseSearchStrategy


def test_cached_dense_plan_respects_increased_dense_top_k() -> None:
    strategy = DenseSearchStrategy()
    signature = ("demo", "dense")
    dense_cfg = DenseIndexConfig(oversample=1)

    retrieval_low = RetrievalConfig(
        dense_top_k=5,
        dense_oversample=1.0,
        dense_overfetch_factor=1.0,
    )
    retrieval_high = RetrievalConfig(
        dense_top_k=12,
        dense_oversample=1.0,
        dense_overfetch_factor=1.0,
    )

    strategy.remember(signature, 8)

    target_low, oversample_low, overfetch_low = strategy.plan(
        signature,
        page_size=5,
        retrieval_cfg=retrieval_low,
        dense_cfg=dense_cfg,
        min_k=0,
    )
    assert target_low == 8
    assert oversample_low >= 1.0
    assert overfetch_low >= 1.0

    target_high, oversample_high, overfetch_high = strategy.plan(
        signature,
        page_size=5,
        retrieval_cfg=retrieval_high,
        dense_cfg=dense_cfg,
        min_k=0,
    )
    assert target_high == 12
    assert oversample_high == oversample_low
    assert overfetch_high == overfetch_low
