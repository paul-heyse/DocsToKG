from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Mapping, Sequence

import numpy as np
import pytest

from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    ChunkRegistry,
    FeatureGenerator,
    HybridSearchAPI,
    HybridSearchConfigManager,
    HybridSearchRequest,
    HybridSearchService,
    HybridSearchValidator,
    Observability,
    OpenSearchSimulator,
    build_stats_snapshot,
    restore_state,
    serialize_state,
    should_rebuild_index,
    verify_pagination,
)
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.validation import infer_embedding_dim, load_dataset
from DocsToKG.HybridSearch.types import DocumentInput


DATASET_PATH = Path("tests/data/real_hybrid_dataset/dataset.jsonl")

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
                page_size=5,
                diagnostics=True,
            )
            response = service.search(request)
            assert response.results, f"Expected results for query {query['query']}"
            assert response.results[0].doc_id == query["expected_doc_id"]
            assert response.results[0].diagnostics is not None

    summary = validator.run(real_dataset, output_root=None)
    assert summary.passed


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

    report_root = tmp_path / "reports"
    summary = validator.run(real_dataset, output_root=report_root)
    assert summary.passed
    report_dirs = list(report_root.iterdir())
    assert report_dirs, "Expected validator to emit reports"
    summary_file = report_dirs[0] / "summary.json"
    assert summary_file.exists()

    state = serialize_state(faiss_index, registry)
    restore_state(faiss_index, state)
    stats_snapshot = build_stats_snapshot(faiss_index, opensearch, registry)
    assert stats_snapshot["faiss"]["ntotal"] >= registry.count()

    request = HybridSearchRequest(query="caregiving burden", namespace="real-fixture", filters={}, page_size=2)
    pagination = verify_pagination(service, request)
    assert not pagination.duplicate_detected

    assert not should_rebuild_index(registry, deleted_since_snapshot=0, threshold=0.5)


def test_remove_ids_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = FaissIndexManager(dim=8, config=DenseIndexConfig())
    manager._use_native = True
    manager._id_lookup = {1: "chunk-1", 2: "chunk-2"}

    class FailingIndex:
        def __init__(self) -> None:
            self.ntotal = 2

        def remove_ids(self, selector: object) -> None:  # pragma: no cover - forced failure
            raise RuntimeError("remove_ids not implemented for this type of index")

    class CPUIndex:
        def __init__(self) -> None:
            self.ntotal = 2
            self.removed = 0

        def remove_ids(self, selector: object) -> None:
            self.removed += 1
            self.ntotal = 0

    failing_index = FailingIndex()
    cpu_index = CPUIndex()

    monkeypatch.setattr(manager, "_index", failing_index, raising=False)
    monkeypatch.setattr(manager, "_to_cpu", lambda index: cpu_index)
    monkeypatch.setattr(manager, "_maybe_to_gpu", lambda index: index)

    manager._remove_ids(np.array([1, 2], dtype=np.int64))
    assert manager._remove_fallbacks == 1
    assert cpu_index.removed == 1
    assert manager._index is cpu_index
