"""
Hybrid Search Scale Tests

This module runs the large-scale hybrid search validation suite against
real vector fixtures to guarantee embedding dimensionality, indexing
performance, and retrieval quality match production expectations.

Key Scenarios:
- Builds ingestion stack with on-disk dataset fixtures
- Executes validator scale suite to assert accuracy and latency targets
- Confirms metrics artifacts are generated for CI inspection

Dependencies:
- pytest: Fixture management and markers for selective execution
- DocsToKG.HybridSearch: Scale validation tooling

Usage:
    pytest tests/test_hybrid_search_scale.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    FeatureGenerator,
    HybridSearchConfigManager,
    HybridSearchService,
    HybridSearchValidator,
    Observability,
)
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.storage import ChunkRegistry, OpenSearchSimulator
from DocsToKG.HybridSearch.types import DocumentInput
from DocsToKG.HybridSearch.validation import infer_embedding_dim, load_dataset

DATASET_PATH = Path("Data/HybridScaleFixture/dataset.jsonl")


@pytest.fixture(scope="session")
def scale_dataset() -> Sequence[Mapping[str, object]]:
    if not DATASET_PATH.exists():
        pytest.skip("Large-scale real vector fixture not generated")
    return load_dataset(DATASET_PATH)


def _build_config(tmp_path: Path) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": 3},
        "fusion": {
            "k0": 60.0,
            "mmr_lambda": 0.6,
            "cosine_dedupe_threshold": 0.95,
            "max_chunks_per_doc": 3,
        },
        "retrieval": {"bm25_top_k": 50, "splade_top_k": 50, "dense_top_k": 50},
    }
    config_path = tmp_path / "scale_hybrid_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(config_path)


@pytest.fixture
def scale_stack(tmp_path: Path, scale_dataset: Sequence[Mapping[str, object]]) -> Callable[
    [],
    tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FaissIndexManager,
        OpenSearchSimulator,
    ],
]:
    def factory() -> tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FaissIndexManager,
        OpenSearchSimulator,
    ]:
        manager = _build_config(tmp_path)
        config = manager.get()
        embedding_dim = infer_embedding_dim(scale_dataset)
        feature_generator = FeatureGenerator(embedding_dim=embedding_dim)
        faiss_index = FaissIndexManager(dim=embedding_dim, config=config.dense)
        opensearch = OpenSearchSimulator()
        registry = ChunkRegistry()
        observability = Observability()
        ingestion = ChunkIngestionPipeline(
            faiss_index=faiss_index,
            opensearch=opensearch,
            registry=registry,
            observability=observability,
        )
        service = HybridSearchService(
            config_manager=manager,
            feature_generator=feature_generator,
            faiss_index=faiss_index,
            opensearch=opensearch,
            registry=registry,
            observability=observability,
        )
        validator = HybridSearchValidator(
            ingestion=ingestion,
            service=service,
            registry=registry,
            opensearch=opensearch,
        )
        return ingestion, service, registry, validator, faiss_index, opensearch

    return factory


@pytest.mark.real_vectors
@pytest.mark.scale_vectors
def test_hybrid_scale_suite(
    scale_stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FaissIndexManager,
            OpenSearchSimulator,
        ],
    ],
    scale_dataset: Sequence[Mapping[str, object]],
    tmp_path: Path,
) -> None:
    ingestion, service, registry, validator, _, _ = scale_stack()

    # Ensure ingest idempotency by running once before the scale suite to warm caches.
    documents = [
        DocumentInput(
            doc_id=str(entry["document"]["doc_id"]),
            namespace=str(entry["document"]["namespace"]),
            chunk_path=Path(entry["document"]["chunk_file"]),
            vector_path=Path(entry["document"]["vector_file"]),
            metadata=dict(entry["document"].get("metadata", {})),
        )
        for entry in scale_dataset
    ]
    ingestion.upsert_documents(documents)

    summary = validator.run_scale(
        scale_dataset,
        output_root=tmp_path,
        query_sample_size=120,
    )
    assert summary.passed, "Scale validation suite reported failures"

    metrics_map = {report.name: report.details for report in summary.reports}
    dense_metrics = metrics_map.get("scale_dense_metrics", {})
    assert dense_metrics.get("self_hit_rate", 0.0) >= 0.95
    performance_metrics = metrics_map.get("scale_performance", {})
    assert "latency_p95_ms" in performance_metrics

    generated_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert generated_dirs, "Expected validator to write metrics"
    metrics_file = generated_dirs[0] / "metrics.json"
    assert metrics_file.exists()
    metrics_payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "scale_dense_metrics" in metrics_payload
