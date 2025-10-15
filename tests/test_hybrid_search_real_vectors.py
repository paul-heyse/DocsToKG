"""
Hybrid Search Real Vector Tests

This module runs hybrid search validation against the real vector fixture
produced by CI, covering ingestion idempotency, retrieval correctness,
metrics reporting, and snapshot lifecycle management.

Key Scenarios:
- Ingests real chunk datasets and executes semantic queries
- Exercises re-ingestion pathways including GPU fallback handling
- Generates validator reports and verifies serialization helpers

Dependencies:
- pytest/numpy: Numeric comparisons and fixture orchestration
- DocsToKG.HybridSearch: Full hybrid search stack under test

Usage:
    pytest tests/test_hybrid_search_real_vectors.py
"""

from __future__ import annotations

import json
from http import HTTPStatus
import os
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import pytest

from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    FeatureGenerator,
    HybridSearchAPI,
    HybridSearchConfigManager,
    HybridSearchRequest,
    HybridSearchService,
    HybridSearchValidator,
    Observability,
    build_stats_snapshot,
    restore_state,
    serialize_state,
    should_rebuild_index,
    verify_pagination,
)
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.storage import ChunkRegistry, OpenSearchSimulator
from DocsToKG.HybridSearch.validation import infer_embedding_dim, load_dataset
from DocsToKG.HybridSearch.types import DocumentInput


DATASET_PATH = Path("Data/HybridScaleFixture/dataset.jsonl")

pytestmark = pytest.mark.real_vectors


def _build_config(tmp_path: Path, *, oversample: int = 3) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": oversample},
        "fusion": {"k0": 50.0, "mmr_lambda": 0.6, "cosine_dedupe_threshold": 0.95, "max_chunks_per_doc": 3},
        "retrieval": {"bm25_top_k": 40, "splade_top_k": 40, "dense_top_k": 40},
    }
    config_path = tmp_path / "real_hybrid_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(config_path)


@pytest.fixture(scope="session")
def real_dataset() -> Sequence[Mapping[str, object]]:
    if not DATASET_PATH.exists():
        pytest.skip("Real vector dataset not generated")
    return load_dataset(DATASET_PATH)


def _to_documents(entries: Sequence[Mapping[str, object]]) -> List[DocumentInput]:
    documents: List[DocumentInput] = []
    for entry in entries:
        document = entry["document"]
        documents.append(
            DocumentInput(
                doc_id=str(document["doc_id"]),
                namespace=str(document["namespace"]),
                chunk_path=Path(str(document["chunk_file"])),
                vector_path=Path(str(document["vector_file"])),
                metadata=dict(document.get("metadata", {})),
            )
        )
    return documents


@pytest.fixture
def stack(
    tmp_path: Path, real_dataset: Sequence[Mapping[str, object]]
) -> Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FaissIndexManager, OpenSearchSimulator]]:
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
        embedding_dim = infer_embedding_dim(real_dataset)
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


def test_real_fixture_ingest_and_search(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FaissIndexManager, OpenSearchSimulator]],
    real_dataset: Sequence[Mapping[str, object]],
) -> None:
    ingestion, service, registry, validator, _, _ = stack()
    documents = _to_documents(real_dataset)
    ingested = ingestion.upsert_documents(documents)
    assert ingested, "Expected chunks to ingest from real vector fixture"
    assert registry.count() == len(ingested)

    for entry in real_dataset:
        for query in entry.get("queries", []):
            request = HybridSearchRequest(
                query=str(query["query"]),
                namespace=query.get("namespace"),
                filters={},
                page_size=10,
                diagnostics=True,
            )
            response = service.search(request)
            assert response.results, f"Expected results for query {query['query']}"
            top_ids = [result.doc_id for result in response.results[:10]]
            assert query["expected_doc_id"] in top_ids
            assert response.results[0].diagnostics is not None


def test_real_fixture_reingest_and_reports(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FaissIndexManager, OpenSearchSimulator]],
    real_dataset: Sequence[Mapping[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion, service, registry, validator, faiss_index, opensearch = stack()
    documents = _to_documents(real_dataset)
    ingestion.upsert_documents(documents)
    baseline_total = registry.count()

    if faiss_index._use_native and faiss_index._index is not None:
        original_remove_ids = faiss_index._index.remove_ids

        def side_effect(selector: object) -> None:
            side_effect.calls += 1
            if side_effect.calls == 1:
                raise RuntimeError("remove_ids not implemented for this type of index")
            original_remove_ids(selector)

        side_effect.calls = 0
        monkeypatch.setattr(faiss_index._index, "remove_ids", side_effect, raising=False)

    ingestion.upsert_documents(documents)
    assert registry.count() == baseline_total

    stats = faiss_index.stats()
    if faiss_index._use_native and faiss_index._index is not None:
        assert stats["gpu_remove_fallbacks"] >= 1
    else:
        assert stats["gpu_remove_fallbacks"] == 0

    env_output = os.environ.get("REAL_VECTOR_REPORT_DIR")
    if env_output:
        report_root = Path(env_output)
        report_root.mkdir(parents=True, exist_ok=True)
        for child in report_root.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for descendant in sorted(child.rglob("*"), reverse=True):
                    if descendant.is_file():
                        descendant.unlink()
                    else:
                        descendant.rmdir()
                child.rmdir()
    else:
        report_root = tmp_path / "reports"
    summary = validator.run(real_dataset, output_root=report_root)
    report_dirs = [path for path in report_root.iterdir() if path.is_dir()]
    assert report_dirs, "Expected validator to emit reports"
    summary_file = report_dirs[0] / "summary.json"
    assert summary_file.exists()
    reports_by_name = {report.name: report for report in summary.reports}
    ingest_report = reports_by_name.get("ingest_integrity")
    assert ingest_report and ingest_report.details.get("total_chunks") == registry.count()
    backup_report = reports_by_name.get("backup_restore")
    assert backup_report and backup_report.passed

    state = serialize_state(faiss_index, registry)
    restore_state(faiss_index, state)
    stats_snapshot = build_stats_snapshot(faiss_index, opensearch, registry)
    assert stats_snapshot["faiss"]["ntotal"] >= registry.count()

    request = HybridSearchRequest(query="caregiving burden", namespace="real-fixture", filters={}, page_size=2)
    pagination = verify_pagination(service, request)
    assert not pagination.duplicate_detected

    assert not should_rebuild_index(registry, deleted_since_snapshot=0, threshold=0.5)


def test_real_fixture_api_roundtrip(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FaissIndexManager, OpenSearchSimulator]],
    real_dataset: Sequence[Mapping[str, object]],
) -> None:
    ingestion, service, registry, _, _, _ = stack()
    documents = _to_documents(real_dataset)
    ingestion.upsert_documents(documents)
    api = HybridSearchAPI(service)
    first_entry = real_dataset[0]
    query = first_entry["queries"][0]
    status, body = api.post_hybrid_search(
        {
            "query": query["query"],
            "namespace": query["namespace"],
            "page_size": 3,
        }
    )
    assert status == HTTPStatus.OK
    assert body["results"], "Expected API to return results"
    assert body["results"][0]["doc_id"] == query["expected_doc_id"]


def test_remove_ids_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("faiss")
    manager = FaissIndexManager(dim=8, config=DenseIndexConfig())
    vector = np.ones(8, dtype=np.float32)
    manager.add([vector], ["00000000-0000-4000-8000-000000000001"])

    rebuilds: Dict[str, int] = {"count": 0}
    original_create_index = manager._create_index

    def tracking_create_index() -> object:
        rebuilds["count"] += 1
        return original_create_index()

    monkeypatch.setattr(manager, "_create_index", tracking_create_index)

    original_remove_ids = manager._index.remove_ids

    def failing_remove_ids(selector: object) -> None:
        raise RuntimeError("remove_ids not implemented for this type of index")

    monkeypatch.setattr(manager._index, "remove_ids", failing_remove_ids, raising=False)
    manager.remove(["00000000-0000-4000-8000-000000000001"])

    assert manager._remove_fallbacks == 1
    assert rebuilds["count"] == 1
    assert manager.ntotal == 0
