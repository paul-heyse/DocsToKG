# === NAVMAP v1 ===
# {
#   "module": "tests.hybrid_search.test_suite",
#   "purpose": "Pytest coverage for hybrid search suite scenarios",
#   "sections": [
#     {
#       "id": "_build_config",
#       "name": "_build_config",
#       "anchor": "BC",
#       "kind": "function"
#     },
#     {
#       "id": "dataset",
#       "name": "dataset",
#       "anchor": "DATA",
#       "kind": "function"
#     },
#     {
#       "id": "stack",
#       "name": "stack",
#       "anchor": "STAC",
#       "kind": "function"
#     },
#     {
#       "id": "_to_documents",
#       "name": "_to_documents",
#       "anchor": "TD",
#       "kind": "function"
#     },
#     {
#       "id": "_write_document_artifacts",
#       "name": "_write_document_artifacts",
#       "anchor": "WDA",
#       "kind": "function"
#     },
#     {
#       "id": "test_hybrid_retrieval_end_to_end",
#       "name": "test_hybrid_retrieval_end_to_end",
#       "anchor": "THRET",
#       "kind": "function"
#     },
#     {
#       "id": "test_reingest_updates_dense_and_sparse_channels",
#       "name": "test_reingest_updates_dense_and_sparse_channels",
#       "anchor": "TRUDA",
#       "kind": "function"
#     },
#     {
#       "id": "test_validation_harness_reports",
#       "name": "test_validation_harness_reports",
#       "anchor": "TVHR",
#       "kind": "function"
#     },
#     {
#       "id": "test_schema_manager_bootstrap_and_registration",
#       "name": "test_schema_manager_bootstrap_and_registration",
#       "anchor": "TSMBA",
#       "kind": "function"
#     },
#     {
#       "id": "test_api_post_hybrid_search_success_and_validation",
#       "name": "test_api_post_hybrid_search_success_and_validation",
#       "anchor": "TAPHS",
#       "kind": "function"
#     },
#     {
#       "id": "test_operations_snapshot_and_restore_roundtrip",
#       "name": "test_operations_snapshot_and_restore_roundtrip",
#       "anchor": "TOSAR",
#       "kind": "function"
#     },
#     {
#       "id": "test_ingest_missing_vector_raises",
#       "name": "test_ingest_missing_vector_raises",
#       "anchor": "TIMVR",
#       "kind": "function"
#     },
#     {
#       "id": "test_faiss_index_uses_registry_bridge",
#       "name": "test_faiss_index_uses_registry_bridge",
#       "anchor": "TFIUR",
#       "kind": "function"
#     },
#     {
#       "id": "_build_config",
#       "name": "_build_config",
#       "anchor": "BC1",
#       "kind": "function"
#     },
#     {
#       "id": "real_dataset",
#       "name": "real_dataset",
#       "anchor": "RD",
#       "kind": "function"
#     },
#     {
#       "id": "_to_documents",
#       "name": "_to_documents",
#       "anchor": "TD1",
#       "kind": "function"
#     },
#     {
#       "id": "stack",
#       "name": "stack",
#       "anchor": "STAC1",
#       "kind": "function"
#     },
#     {
#       "id": "test_real_fixture_ingest_and_search",
#       "name": "test_real_fixture_ingest_and_search",
#       "anchor": "TRFIA",
#       "kind": "function"
#     },
#     {
#       "id": "test_real_fixture_reingest_and_reports",
#       "name": "test_real_fixture_reingest_and_reports",
#       "anchor": "TRFRA",
#       "kind": "function"
#     },
#     {
#       "id": "test_real_fixture_api_roundtrip",
#       "name": "test_real_fixture_api_roundtrip",
#       "anchor": "TRFAR",
#       "kind": "function"
#     },
#     {
#       "id": "test_remove_ids_cpu_fallback",
#       "name": "test_remove_ids_cpu_fallback",
#       "anchor": "TRICF",
#       "kind": "function"
#     },
#     {
#       "id": "scale_dataset",
#       "name": "scale_dataset",
#       "anchor": "SD",
#       "kind": "function"
#     },
#     {
#       "id": "_build_config",
#       "name": "_build_config",
#       "anchor": "BC2",
#       "kind": "function"
#     },
#     {
#       "id": "scale_stack",
#       "name": "scale_stack",
#       "anchor": "SS",
#       "kind": "function"
#     },
#     {
#       "id": "test_hybrid_scale_suite",
#       "name": "test_hybrid_scale_suite",
#       "anchor": "THSS",
#       "kind": "function"
#     },
#     {
#       "id": "_toy_data",
#       "name": "_toy_data",
#       "anchor": "TD2",
#       "kind": "function"
#     },
#     {
#       "id": "_target_device",
#       "name": "_target_device",
#       "anchor": "TD3",
#       "kind": "function"
#     },
#     {
#       "id": "_make_id_resolver",
#       "name": "_make_id_resolver",
#       "anchor": "MIR",
#       "kind": "function"
#     },
#     {
#       "id": "_emit_vectors",
#       "name": "_emit_vectors",
#       "anchor": "EV",
#       "kind": "function"
#     },
#     {
#       "id": "_assert_gpu_index",
#       "name": "_assert_gpu_index",
#       "anchor": "AGI",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_flat_end_to_end",
#       "name": "test_gpu_flat_end_to_end",
#       "anchor": "TGFET",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_ivf_flat_build_and_search",
#       "name": "test_gpu_ivf_flat_build_and_search",
#       "anchor": "TGIFB",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_ivfpq_build_and_search",
#       "name": "test_gpu_ivfpq_build_and_search",
#       "anchor": "TGIBA",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_cosine_against_corpus",
#       "name": "test_gpu_cosine_against_corpus",
#       "anchor": "TGCAC",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_clone_strict_coarse_quantizer",
#       "name": "test_gpu_clone_strict_coarse_quantizer",
#       "anchor": "TGCSC",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_near_duplicate_detection_filters_duplicates",
#       "name": "test_gpu_near_duplicate_detection_filters_duplicates",
#       "anchor": "TGNDD",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_nprobe_applied_during_search",
#       "name": "test_gpu_nprobe_applied_during_search",
#       "anchor": "TGNAD",
#       "kind": "function"
#     },
#     {
#       "id": "test_gpu_similarity_uses_supplied_device",
#       "name": "test_gpu_similarity_uses_supplied_device",
#       "anchor": "TGSUS",
#       "kind": "function"
#     },
#     {
#       "id": "test_operations_shim_emits_warning_and_reexports",
#       "name": "test_operations_shim_emits_warning_and_reexports",
#       "anchor": "TOSEW",
#       "kind": "function"
#     },
#     {
#       "id": "test_results_shim_emits_warning_and_reexports",
#       "name": "test_results_shim_emits_warning_and_reexports",
#       "anchor": "TRSEW",
#       "kind": "function"
#     },
#     {
#       "id": "test_similarity_shim_emits_warning_and_reexports",
#       "name": "test_similarity_shim_emits_warning_and_reexports",
#       "anchor": "TSSEW",
#       "kind": "function"
#     },
#     {
#       "id": "test_retrieval_shim_emits_warning_and_reexports",
#       "name": "test_retrieval_shim_emits_warning_and_reexports",
#       "anchor": "RSEW1",
#       "kind": "function"
#     },
#     {
#       "id": "test_schema_shim_emits_warning_and_reexports",
#       "name": "test_schema_shim_emits_warning_and_reexports",
#       "anchor": "SSEW1",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Consolidated hybrid search test suite."""

from __future__ import annotations

import importlib
import json
import os
import sys
import uuid
import warnings
from http import HTTPStatus
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
import pytest

from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    DocumentInput,
    FeatureGenerator,
    HybridSearchAPI,
    HybridSearchConfigManager,
    HybridSearchRequest,
    HybridSearchService,
    HybridSearchValidator,
    Observability,
    OpenSearchIndexTemplate,
    OpenSearchSchemaManager,
)
from DocsToKG.HybridSearch.config import DenseIndexConfig, FusionConfig
from DocsToKG.HybridSearch.features import tokenize
from DocsToKG.HybridSearch.ingest import IngestError
from DocsToKG.HybridSearch.ranking import ResultShaper
from DocsToKG.HybridSearch.service import (
    ChannelResults,
    RequestValidationError,
    build_stats_snapshot,
    should_rebuild_index,
    verify_pagination,
)
from DocsToKG.HybridSearch.storage import ChunkRegistry, OpenSearchSimulator
from DocsToKG.HybridSearch.types import (
    ChunkFeatures,
    ChunkPayload,
    vector_uuid_to_faiss_int,
)
from DocsToKG.HybridSearch.validation import infer_embedding_dim, load_dataset
from DocsToKG.HybridSearch.vectorstore import (
    FaissIndexManager,
    cosine_against_corpus_gpu,
    max_inner_product,
    normalize_rows,
    pairwise_inner_products,
    restore_state,
    serialize_state,
)

faiss = pytest.importorskip("faiss")
if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
    pytest.skip(
        "Hybrid search integration suite requires CUDA-enabled faiss", allow_module_level=True
    )

REAL_VECTOR_MARK = pytest.mark.real_vectors

GPU_MARK = pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="FAISS GPU device required")


# ---- test_hybrid_search.py -----------------------------
def _build_config(tmp_path: Path) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": 3},
        "fusion": {
            "k0": 50.0,
            "mmr_lambda": 0.7,
            "cosine_dedupe_threshold": 0.95,
            "max_chunks_per_doc": 2,
        },
        "retrieval": {"bm25_top_k": 20, "splade_top_k": 20, "dense_top_k": 20},
    }
    path = tmp_path / "hybrid_config.json"
    path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(path)


# ---- test_hybrid_search.py -----------------------------
@pytest.fixture
def dataset() -> Sequence[Mapping[str, object]]:
    return load_dataset(Path("tests/data/hybrid_dataset.jsonl"))


# ---- test_hybrid_search.py -----------------------------
@pytest.fixture
def stack(
    tmp_path: Path,
) -> Callable[
    [],
    tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FeatureGenerator,
        OpenSearchSimulator,
    ],
]:
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
        feature_generator = FeatureGenerator()
        faiss_index = FaissIndexManager(dim=feature_generator.embedding_dim, config=config.dense)
        assert (
            faiss_index.dim == feature_generator.embedding_dim
        ), "Faiss index dimensionality must match feature generator"
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


# ---- test_hybrid_search.py -----------------------------
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


# ---- test_hybrid_search.py -----------------------------
def _write_document_artifacts(
    base_dir: Path,
    *,
    doc_id: str,
    namespace: str,
    text: str,
    metadata: Mapping[str, object],
    feature_generator: FeatureGenerator,
) -> DocumentInput:
    if not hasattr(feature_generator, "compute_features"):
        feature_generator = FeatureGenerator()
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
        "schema_version": "docparse/1.1.0",
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
        "schema_version": "embeddings/1.0.0",
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


# ---- test_hybrid_search.py -----------------------------
def test_hybrid_retrieval_end_to_end(
    stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FeatureGenerator,
            OpenSearchSimulator,
        ],
    ],
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


# ---- test_hybrid_search.py -----------------------------
def test_reingest_updates_dense_and_sparse_channels(
    stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FeatureGenerator,
            OpenSearchSimulator,
        ],
    ],
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

    request = HybridSearchRequest(
        query="dense retrieval guidance", namespace="research", filters={}, page_size=3
    )
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


# ---- test_hybrid_search.py -----------------------------
def test_validation_harness_reports(
    stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FeatureGenerator,
            OpenSearchSimulator,
        ],
    ],
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


# ---- test_hybrid_search.py -----------------------------
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


# ---- test_hybrid_search.py -----------------------------
def test_api_post_hybrid_search_success_and_validation(
    stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FeatureGenerator,
            OpenSearchSimulator,
        ],
    ],
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


# ---- test_hybrid_search.py -----------------------------
def test_operations_snapshot_and_restore_roundtrip(
    stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FeatureGenerator,
            OpenSearchSimulator,
        ],
    ],
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
    assert should_rebuild_index(
        registry, deleted_since_snapshot=max(1, registry.count() // 2), threshold=0.25
    )


# ---- test_hybrid_search.py -----------------------------
def test_ingest_missing_vector_raises(
    stack: Callable[
        [],
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FeatureGenerator,
            OpenSearchSimulator,
        ],
    ],
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
    vector_entries = [
        json.loads(line)
        for line in doc.vector_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vector_entries[0]["UUID"] = "00000000-0000-0000-0000-000000000000"
    doc.vector_path.write_text(
        "\n".join(json.dumps(entry) for entry in vector_entries) + "\n", encoding="utf-8"
    )

    with pytest.raises(IngestError):
        ingestion.upsert_documents([doc])


# ---- test_hybrid_search.py -----------------------------
def test_faiss_index_uses_registry_bridge(tmp_path: Path) -> None:
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


# ---- test_hybrid_search_real_vectors.py -----------------------------
DATASET_PATH = Path("Data/HybridScaleFixture/dataset.jsonl")


# ---- test_hybrid_search_real_vectors.py -----------------------------
def _build_config(tmp_path: Path, *, oversample: int = 3) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": oversample},
        "fusion": {
            "k0": 50.0,
            "mmr_lambda": 0.6,
            "cosine_dedupe_threshold": 0.95,
            "max_chunks_per_doc": 3,
        },
        "retrieval": {"bm25_top_k": 40, "splade_top_k": 40, "dense_top_k": 40},
    }
    config_path = tmp_path / "real_hybrid_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(config_path)


# ---- test_hybrid_search_real_vectors.py -----------------------------
@REAL_VECTOR_MARK
@pytest.fixture(scope="session")
def real_dataset() -> Sequence[Mapping[str, object]]:
    if not DATASET_PATH.exists():
        pytest.skip("Real vector dataset not generated")
    return load_dataset(DATASET_PATH)


# ---- test_hybrid_search_real_vectors.py -----------------------------
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


# ---- test_hybrid_search_real_vectors.py -----------------------------
@REAL_VECTOR_MARK
@pytest.fixture
def stack(tmp_path: Path, real_dataset: Sequence[Mapping[str, object]]) -> Callable[  # noqa: F811
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


# ---- test_hybrid_search_real_vectors.py -----------------------------
@REAL_VECTOR_MARK
def test_real_fixture_ingest_and_search(
    stack: Callable[
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


# ---- test_hybrid_search_real_vectors.py -----------------------------
@REAL_VECTOR_MARK
def test_real_fixture_reingest_and_reports(
    stack: Callable[
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
    real_dataset: Sequence[Mapping[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion, service, registry, validator, faiss_index, opensearch = stack()
    documents = _to_documents(real_dataset)
    ingestion.upsert_documents(documents)
    baseline_total = registry.count()

    if faiss_index._index is not None:
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
    if faiss_index._index is not None:
        assert stats["gpu_remove_fallbacks"] >= 1

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

    request = HybridSearchRequest(
        query="caregiving burden", namespace="real-fixture", filters={}, page_size=2
    )
    pagination = verify_pagination(service, request)
    assert not pagination.duplicate_detected

    assert not should_rebuild_index(registry, deleted_since_snapshot=0, threshold=0.5)


# ---- test_hybrid_search_real_vectors.py -----------------------------
@REAL_VECTOR_MARK
def test_real_fixture_api_roundtrip(
    stack: Callable[
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


# ---- test_hybrid_search_real_vectors.py -----------------------------
def test_remove_ids_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = FaissIndexManager(dim=8, config=DenseIndexConfig())
    vector = np.ones(8, dtype=np.float32)
    manager.add([vector], ["00000000-0000-4000-8000-000000000001"])

    rebuilds: Dict[str, int] = {"count": 0}
    original_create_index = manager._create_index

    def tracking_create_index() -> object:
        rebuilds["count"] += 1
        return original_create_index()

    monkeypatch.setattr(manager, "_create_index", tracking_create_index)

    def failing_remove_ids(selector: object) -> None:
        raise RuntimeError("remove_ids not implemented for this type of index")

    monkeypatch.setattr(manager._index, "remove_ids", failing_remove_ids, raising=False)
    manager.remove(["00000000-0000-4000-8000-000000000001"])

    assert manager._remove_fallbacks == 1
    assert rebuilds["count"] == 1
    assert manager.ntotal == 0


# ---- test_hybrid_search_scale.py -----------------------------
DATASET_PATH = Path("Data/HybridScaleFixture/dataset.jsonl")


# ---- test_hybrid_search_scale.py -----------------------------
@pytest.fixture(scope="session")
def scale_dataset() -> Sequence[Mapping[str, object]]:
    if not DATASET_PATH.exists():
        pytest.skip("Large-scale real vector fixture not generated")
    return load_dataset(DATASET_PATH)


# ---- test_hybrid_search_scale.py -----------------------------
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


# ---- test_hybrid_search_scale.py -----------------------------
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


# ---- test_hybrid_search_scale.py -----------------------------
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


# ---- test_hybridsearch_gpu_only.py -----------------------------
try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("faiss is required for these tests", allow_module_level=True)


# ---- test_hybridsearch_gpu_only.py -----------------------------
def _toy_data(n: int = 2048, d: int = 128) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((64, d), dtype=np.float32)
    return xb, xq


# ---- test_hybridsearch_gpu_only.py -----------------------------
def _target_device() -> int:
    return int(os.getenv("HYBRIDSEARCH_FAISS_DEVICE", "0"))


# ---- test_hybridsearch_gpu_only.py -----------------------------
def _make_id_resolver(vector_ids: list[str]) -> Callable[[int], str | None]:
    bridge = {vector_uuid_to_faiss_int(vid): vid for vid in vector_ids}
    return bridge.get


# ---- test_hybridsearch_gpu_only.py -----------------------------
def _emit_vectors(xb: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    vectors = [row.copy() for row in xb]
    vector_ids = [str(uuid.uuid4()) for _ in vectors]
    return vectors, vector_ids


# ---- test_hybridsearch_gpu_only.py -----------------------------
def _assert_gpu_index(manager: FaissIndexManager) -> None:
    stats = manager.stats()
    device = stats.get("device")
    assert device not in (None, "*"), f"Expected GPU device assignment, stats={stats}"
    assert int(device) == _target_device(), f"Index promoted to unexpected device: {stats}"
    base = manager._index
    if hasattr(base, "index"):
        base = base.index
    if hasattr(faiss, "downcast_index"):
        try:
            base = faiss.downcast_index(base)
        except Exception:
            pass
    assert "Gpu" in type(base).__name__, f"Expected GPU index type, got {type(base)}"
    assert stats.get("gpu_base") is True
    assert float(stats.get("gpu_remove_fallbacks", 0.0)) == 0.0


# ---- test_hybridsearch_gpu_only.py -----------------------------
@GPU_MARK
def test_gpu_flat_end_to_end() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(index_type="flat", nprobe=1, device=_target_device())
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


# ---- test_hybridsearch_gpu_only.py -----------------------------
@GPU_MARK
def test_gpu_ivf_flat_build_and_search() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(index_type="ivf_flat", nlist=256, nprobe=8, device=_target_device())
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


# ---- test_hybridsearch_gpu_only.py -----------------------------
@GPU_MARK
def test_gpu_ivfpq_build_and_search() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(
        index_type="ivf_pq",
        nlist=256,
        nprobe=8,
        pq_m=16,
        pq_bits=8,
        device=_target_device(),
    )
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


# ---- test_hybridsearch_gpu_only.py -----------------------------
@GPU_MARK
def test_gpu_cosine_against_corpus() -> None:
    xb, xq = _toy_data(n=512)
    query = xq[0]
    resources = faiss.StandardGpuResources()
    sims = cosine_against_corpus_gpu(query, xb, device=_target_device(), resources=resources)
    assert sims.shape == (1, xb.shape[0])
    self_sim = float(
        cosine_against_corpus_gpu(
            query,
            query.reshape(1, -1),
            device=_target_device(),
            resources=resources,
        )[0, 0]
    )
    assert 0.98 <= self_sim <= 1.001


# ---- test_hybridsearch_gpu_only.py -----------------------------
def test_gpu_clone_strict_coarse_quantizer() -> None:
    cfg = DenseIndexConfig(index_type="flat", device=_target_device())
    manager = FaissIndexManager(dim=32, config=cfg)
    cpu_index = faiss.IndexFlatIP(32)
    mapped = faiss.IndexIDMap2(cpu_index)
    gpu_index = manager._maybe_to_gpu(mapped)
    base = gpu_index.index if hasattr(gpu_index, "index") else gpu_index
    if hasattr(faiss, "downcast_index"):
        base = faiss.downcast_index(base)
    assert "Gpu" in type(base).__name__, "Expected GPU index after strict cloning"


# ---- test_hybridsearch_gpu_only.py -----------------------------
def test_gpu_near_duplicate_detection_filters_duplicates() -> None:
    embedding = np.ones(16, dtype=np.float32)
    features = ChunkFeatures({}, {}, embedding)
    chunk_a = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        vector_id="vec-1",
        namespace="default",
        text="Hybrid search test chunk",
        metadata={},
        features=features,
        token_count=4,
        source_chunk_idxs=[0],
        doc_items_refs=[],
    )
    chunk_b = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-2",
        vector_id="vec-2",
        namespace="default",
        text="Hybrid search test chunk",
        metadata={},
        features=ChunkFeatures({}, {}, embedding.copy()),
        token_count=4,
        source_chunk_idxs=[1],
        doc_items_refs=[],
    )
    opensearch = OpenSearchSimulator()
    opensearch.bulk_upsert([chunk_a, chunk_b])
    shaper = ResultShaper(
        opensearch,
        FusionConfig(cosine_dedupe_threshold=0.9),
        device=_target_device(),
        resources=faiss.StandardGpuResources(),
    )
    request = HybridSearchRequest(query="hybrid search", namespace=None, filters={}, page_size=5)
    fused_scores = {chunk_a.vector_id: 1.0, chunk_b.vector_id: 0.95}
    channel_scores = {"dense": fused_scores}
    results = shaper.shape([chunk_a, chunk_b], fused_scores, request, channel_scores)
    assert len(results) == 1, "GPU near-duplicate detection should filter duplicates"


# ---- test_hybridsearch_gpu_only.py -----------------------------
def test_gpu_nprobe_applied_during_search() -> None:
    xb, xq = _toy_data(n=512, d=64)
    cfg = DenseIndexConfig(index_type="ivf_flat", nlist=64, nprobe=32, device=_target_device())
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    manager.search(xq[0], top_k=5)
    base = manager._index.index if hasattr(manager._index, "index") else manager._index
    if hasattr(faiss, "downcast_index"):
        base = faiss.downcast_index(base)
    assert hasattr(base, "nprobe")
    assert int(base.nprobe) == cfg.nprobe


# ---- test_hybridsearch_gpu_only.py -----------------------------
def test_gpu_similarity_uses_supplied_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DenseIndexConfig(index_type="flat", device=_target_device())
    manager = FaissIndexManager(dim=32, config=cfg)

    captured: dict[str, object] = {}

    def fake_pairwise(resources, A, B, metric, device):  # type: ignore[no-untyped-def]
        captured["resources"] = resources
        captured["device"] = device
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)

    monkeypatch.setattr(faiss, "pairwise_distance_gpu", fake_pairwise)

    q = np.ones(32, dtype=np.float32)
    corpus = np.ones((3, 32), dtype=np.float32)
    cosine_against_corpus_gpu(q, corpus, device=manager.device, resources=manager.gpu_resources)

    assert captured.get("device") == manager.device
    assert captured.get("resources") is manager.gpu_resources


def test_operations_shim_emits_warning_and_reexports() -> None:
    module_name = "DocsToKG.HybridSearch.operations"
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(
        "deprecated" in str(w.message).lower() and w.category is DeprecationWarning for w in caught
    ), "Importing the shim should emit a deprecation warning"
    assert module.serialize_state is serialize_state
    assert module.restore_state is restore_state
    assert module.verify_pagination is verify_pagination


def test_results_shim_emits_warning_and_reexports() -> None:
    module_name = "DocsToKG.HybridSearch.results"
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(
        "deprecated" in str(w.message).lower() and w.category is DeprecationWarning for w in caught
    ), "Importing the shim should emit a deprecation warning"
    assert module.ResultShaper is ResultShaper


def test_similarity_shim_emits_warning_and_reexports() -> None:
    module_name = "DocsToKG.HybridSearch.similarity"
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(
        "deprecated" in str(w.message).lower() and w.category is DeprecationWarning for w in caught
    ), "Importing the shim should emit a deprecation warning"
    assert module.normalize_rows is normalize_rows
    assert module.cosine_against_corpus_gpu is cosine_against_corpus_gpu
    assert module.pairwise_inner_products is pairwise_inner_products
    assert module.max_inner_product is max_inner_product


def test_retrieval_shim_emits_warning_and_reexports() -> None:
    module_name = "DocsToKG.HybridSearch.retrieval"
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(
        "deprecated" in str(w.message).lower() and w.category is DeprecationWarning for w in caught
    ), "Importing the shim should emit a deprecation warning"
    assert module.HybridSearchService is HybridSearchService
    assert module.HybridSearchAPI is HybridSearchAPI
    assert module.RequestValidationError is RequestValidationError
    assert module.ChannelResults is ChannelResults
    assert module.verify_pagination is verify_pagination


def test_schema_shim_emits_warning_and_reexports() -> None:
    module_name = "DocsToKG.HybridSearch.schema"
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(
        "deprecated" in str(w.message).lower() and w.category is DeprecationWarning for w in caught
    ), "Importing the shim should emit a deprecation warning"
    assert module.OpenSearchSchemaManager is OpenSearchSchemaManager
    assert module.OpenSearchIndexTemplate is OpenSearchIndexTemplate
