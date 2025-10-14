from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from typing import Callable, List, Mapping, Sequence

import numpy as np
import pytest

from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    HybridSearchAPI,
    DocumentInput,
    FeatureGenerator,
    HybridSearchConfigManager,
    HybridSearchRequest,
    HybridSearchService,
    HybridSearchValidator,
    Observability,
    OpenSearchSchemaManager,
    build_stats_snapshot,
    restore_state,
    serialize_state,
    should_rebuild_index,
    verify_pagination,
)
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.storage import ChunkRegistry, OpenSearchSimulator
from DocsToKG.HybridSearch.validation import load_dataset
from DocsToKG.HybridSearch.tokenization import tokenize
from DocsToKG.HybridSearch.ingest import IngestError
from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload
from uuid import NAMESPACE_URL, uuid5, uuid4


def _build_config(tmp_path: Path) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": 3},
        "fusion": {"k0": 50.0, "mmr_lambda": 0.7, "cosine_dedupe_threshold": 0.95, "max_chunks_per_doc": 2},
        "retrieval": {"bm25_top_k": 20, "splade_top_k": 20, "dense_top_k": 20},
    }
    path = tmp_path / "hybrid_config.json"
    path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(path)


@pytest.fixture
def dataset() -> Sequence[Mapping[str, object]]:
    return load_dataset(Path("tests/data/hybrid_dataset.jsonl"))


@pytest.fixture
def stack(
    tmp_path: Path,
) -> Callable[[], tuple[
    ChunkIngestionPipeline,
    HybridSearchService,
    ChunkRegistry,
    HybridSearchValidator,
    FeatureGenerator,
    OpenSearchSimulator,
]]:
    def factory() -> tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FeatureGenerator,
        OpenSearchSimulator,
    ]:
        manager = _build_config(tmp_path)
        config = manager.get()
        feature_generator = FeatureGenerator(embedding_dim=16)
        faiss_index = FaissIndexManager(dim=feature_generator.embedding_dim, config=config.dense)
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
        return ingestion, service, registry, validator, feature_generator, opensearch

    return factory


def _to_documents(entries: Sequence[Mapping[str, object]]) -> List[DocumentInput]:
    documents: List[DocumentInput] = []
    for entry in entries:
        doc = entry["document"]
        documents.append(
            DocumentInput(
                doc_id=str(doc["doc_id"]),
                namespace=str(doc["namespace"]),
                chunk_path=Path(str(doc["chunk_file"])),
                vector_path=Path(str(doc["vector_file"])),
                metadata=dict(doc.get("metadata", {})),
            )
        )
    return documents


def _write_document_artifacts(
    base_dir: Path,
    *,
    doc_id: str,
    namespace: str,
    text: str,
    metadata: Mapping[str, object],
    feature_generator: FeatureGenerator,
) -> DocumentInput:
    chunk_dir = base_dir / "chunks"
    vector_dir = base_dir / "vectors"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    chunk_uuid = str(uuid5(NAMESPACE_URL, f"{doc_id}:{text}"))
    tokens = tokenize(text)
    chunk_payload = {
        "doc_id": doc_id,
        "source_path": f"/tmp/{doc_id}.doctags",
        "chunk_id": 0,
        "source_chunk_idxs": [0],
        "num_tokens": len(tokens),
        "text": text,
        "doc_items_refs": ["#/texts/0"],
        "page_nos": [],
        "uuid": chunk_uuid,
    }
    chunk_path = chunk_dir / f"{doc_id}.chunks.jsonl"
    chunk_path.write_text(json.dumps(chunk_payload) + "\n", encoding="utf-8")

    features = feature_generator.compute_features(text)
    sorted_bm25 = sorted(features.bm25_terms.items())
    sorted_splade = sorted(features.splade_weights.items())
    vector_entry = {
        "UUID": chunk_uuid,
        "BM25": {
            "terms": [token for token, _ in sorted_bm25],
            "weights": [float(weight) for _, weight in sorted_bm25],
            "k1": 1.5,
            "b": 0.75,
            "avgdl": 120.0,
            "N": 1,
        },
        "SpladeV3": {
            "model_id": "naver/splade-v3",
            "tokens": [token for token, _ in sorted_splade],
            "weights": [float(weight) for _, weight in sorted_splade],
        },
        "Qwen3-4B": {
            "model_id": "Qwen/Qwen3-Embedding-4B",
            "vector": [float(x) for x in features.embedding.tolist()],
        },
    }
    vector_path = vector_dir / f"{doc_id}.vectors.jsonl"
    vector_path.write_text(json.dumps(vector_entry) + "\n", encoding="utf-8")

    return DocumentInput(
        doc_id=doc_id,
        namespace=namespace,
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata=dict(metadata),
    )


def test_hybrid_retrieval_end_to_end(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FeatureGenerator, OpenSearchSimulator]],
    dataset: Sequence[Mapping[str, object]],
) -> None:
    ingestion, service, registry, _, _, _ = stack()
    documents = _to_documents(dataset)
    ingestion.upsert_documents(documents)

    request = HybridSearchRequest(
        query="hybrid retrieval faiss",
        namespace="research",
        filters={},
        page_size=5,
    )
    response = service.search(request)

    assert response.results, "Expected hybrid search results"
    assert response.results[0].doc_id == "doc-1"
    assert response.results[0].diagnostics.dense_score is not None
    assert response.timings_ms["total_ms"] > 0.0


def test_reingest_updates_dense_and_sparse_channels(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FeatureGenerator, OpenSearchSimulator]],
    tmp_path: Path,
) -> None:
    ingestion, service, registry, _, feature_generator, _ = stack()
    artifacts_dir = tmp_path / "docs"
    doc = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-10",
        namespace="research",
        text="Original dense retrieval guidance for FAISS Flat indexes.",
        metadata={"author": "Kai"},
        feature_generator=feature_generator,
    )
    ingestion.upsert_documents([doc])

    request = HybridSearchRequest(query="dense retrieval guidance", namespace="research", filters={}, page_size=3)
    first = service.search(request)
    assert first.results and "Original" in first.results[0].text

    updated = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-10",
        namespace="research",
        text="Updated dense retrieval guidance including IVFPQ calibration notes.",
        metadata={"author": "Kai"},
        feature_generator=feature_generator,
    )
    ingestion.upsert_documents([updated])

    second = service.search(request)
    assert second.results and "Updated" in second.results[0].text
    assert ingestion.metrics.chunks_upserted >= 2


def test_validation_harness_reports(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FeatureGenerator, OpenSearchSimulator]],
    dataset: Sequence[Mapping[str, object]],
    tmp_path: Path,
) -> None:
    ingestion, service, registry, validator, _, _ = stack()
    documents = _to_documents(dataset)
    ingestion.upsert_documents(documents)

    summary = validator.run(dataset, output_root=tmp_path)

    assert summary.passed
    report_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert report_dirs, "Validation reports were not written"
    summary_file = report_dirs[0] / "summary.json"
    assert summary_file.exists()


def test_schema_manager_bootstrap_and_registration() -> None:
    manager = OpenSearchSchemaManager()
    template = manager.bootstrap_template("research")
    assert template.body["mappings"]["properties"]["splade"]["type"] == "rank_features"
    assert template.chunking.max_tokens > 0
    simulator = OpenSearchSimulator()
    simulator.register_template(template)
    stored = simulator.template_for("research")
    assert stored is template
    metadata_props = template.body["mappings"]["properties"]["metadata"]["properties"]
    assert "author" in metadata_props and metadata_props["tags"]["type"] == "keyword"


def test_api_post_hybrid_search_success_and_validation(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FeatureGenerator, OpenSearchSimulator]],
    dataset: Sequence[Mapping[str, object]],
) -> None:
    ingestion, service, _, _, _, _ = stack()
    documents = _to_documents(dataset)
    ingestion.upsert_documents(documents)
    api = HybridSearchAPI(service)

    status, body = api.post_hybrid_search(
        {
            "query": "hybrid retrieval faiss",
            "namespace": "research",
            "page_size": 3,
            "filters": {"tags": ["retrieval"]},
        }
    )
    assert status == HTTPStatus.OK
    assert body["results"] and body["results"][0]["doc_id"] == "doc-1"

    error_status, error_body = api.post_hybrid_search({"query": "", "page_size": -1})
    assert error_status == HTTPStatus.BAD_REQUEST
    assert "error" in error_body


def test_operations_snapshot_and_restore_roundtrip(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FeatureGenerator, OpenSearchSimulator]],
    dataset: Sequence[Mapping[str, object]],
) -> None:
    ingestion, service, registry, _, _, opensearch = stack()
    documents = _to_documents(dataset)
    ingestion.upsert_documents(documents)

    stats = build_stats_snapshot(ingestion.faiss_index, opensearch, registry)
    assert stats["faiss"]["ntotal"] >= registry.count()
    state = serialize_state(ingestion.faiss_index, registry)
    restore_state(ingestion.faiss_index, state)

    request = HybridSearchRequest(query="faiss", namespace="research", filters={}, page_size=2)
    pagination_result = verify_pagination(service, request)
    assert not pagination_result.duplicate_detected

    assert not should_rebuild_index(registry, deleted_since_snapshot=0, threshold=0.5)
    assert should_rebuild_index(registry, deleted_since_snapshot=max(1, registry.count() // 2), threshold=0.25)


def test_ingest_missing_vector_raises(
    stack: Callable[[], tuple[ChunkIngestionPipeline, HybridSearchService, ChunkRegistry, HybridSearchValidator, FeatureGenerator, OpenSearchSimulator]],
    tmp_path: Path,
) -> None:
    ingestion, _, _, _, feature_generator, _ = stack()
    artifacts_dir = tmp_path / "docs"
    doc = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-missing",
        namespace="research",
        text="Chunk without matching vector entry",
        metadata={},
        feature_generator=feature_generator,
    )
    vector_entries = [json.loads(line) for line in doc.vector_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vector_entries[0]["UUID"] = "00000000-0000-0000-0000-000000000000"
    doc.vector_path.write_text("\n".join(json.dumps(entry) for entry in vector_entries) + "\n", encoding="utf-8")

    with pytest.raises(IngestError):
        ingestion.upsert_documents([doc])


def test_faiss_index_uses_registry_bridge(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    config = DenseIndexConfig(index_type="flat")
    manager = FaissIndexManager(dim=4, config=config)
    registry = ChunkRegistry()
    manager.set_id_resolver(registry.resolve_faiss_id)

    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    features = ChunkFeatures(bm25_terms={}, splade_weights={}, embedding=embedding)
    chunk = ChunkPayload(
        doc_id="doc-bridge",
        chunk_id="0",
        vector_id=str(uuid4()),
        namespace="bridge",
        text="example chunk",
        metadata={},
        features=features,
        token_count=int(embedding.size),
        source_chunk_idxs=(0,),
        doc_items_refs=(),
        char_offset=(0, len("example chunk")),
    )

    manager.add([features.embedding], [chunk.vector_id])
    registry.upsert([chunk])

    hits = manager.search(embedding, 1)
    assert hits and hits[0].vector_id == chunk.vector_id

