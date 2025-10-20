# === NAVMAP v1 ===
"""End-to-end hybrid search suite validating ingestion, FAISS GPU usage, API, and snapshots.

Stitches together config management, DocParsing fixtures, FAISS GPU ingestion,
API requests, validator resource budgets, and snapshot/restore flows. Mirrors
the README quickstart and ensures the custom `faiss-1.12.0` wheel (CUDA 12 +
OpenBLAS) works across ingestion, query, and scale scenarios."""

# {
#   "module": "tests.hybrid_search.test_suite",
#   "purpose": "Pytest coverage for hybrid search suite scenarios",
#   "sections": [
#     {
#       "id": "build-config",
#       "name": "_build_config",
#       "anchor": "function-build-config",
#       "kind": "function"
#     },
#     {
#       "id": "dataset",
#       "name": "dataset",
#       "anchor": "function-dataset",
#       "kind": "function"
#     },
#     {
#       "id": "stack",
#       "name": "stack",
#       "anchor": "function-stack",
#       "kind": "function"
#     },
#     {
#       "id": "to-documents",
#       "name": "_to_documents",
#       "anchor": "function-to-documents",
#       "kind": "function"
#     },
#     {
#       "id": "write-document-artifacts",
#       "name": "_write_document_artifacts",
#       "anchor": "function-write-document-artifacts",
#       "kind": "function"
#     },
#     {
#       "id": "test-hybrid-retrieval-end-to-end",
#       "name": "test_hybrid_retrieval_end_to_end",
#       "anchor": "function-test-hybrid-retrieval-end-to-end",
#       "kind": "function"
#     },
#     {
#       "id": "test-reingest-updates-dense-and-sparse-channels",
#       "name": "test_reingest_updates_dense_and_sparse_channels",
#       "anchor": "function-test-reingest-updates-dense-and-sparse-channels",
#       "kind": "function"
#     },
#     {
#       "id": "test-validation-harness-reports",
#       "name": "test_validation_harness_reports",
#       "anchor": "function-test-validation-harness-reports",
#       "kind": "function"
#     },
#     {
#       "id": "test-validator-validation-resources-honor-null-stream-flags",
#       "name": "test_validator_validation_resources_honor_null_stream_flags",
#       "anchor": "function-test-validator-validation-resources-honor-null-stream-flags",
#       "kind": "function"
#     },
#     {
#       "id": "test-schema-manager-bootstrap-and-registration",
#       "name": "test_schema_manager_bootstrap_and_registration",
#       "anchor": "function-test-schema-manager-bootstrap-and-registration",
#       "kind": "function"
#     },
#     {
#       "id": "test-api-post-hybrid-search-success-and-validation",
#       "name": "test_api_post_hybrid_search_success_and_validation",
#       "anchor": "function-test-api-post-hybrid-search-success-and-validation",
#       "kind": "function"
#     },
#     {
#       "id": "test-operations-snapshot-and-restore-roundtrip",
#       "name": "test_operations_snapshot_and_restore_roundtrip",
#       "anchor": "function-test-operations-snapshot-and-restore-roundtrip",
#       "kind": "function"
#     },
#     {
#       "id": "test-ingest-missing-vector-raises",
#       "name": "test_ingest_missing_vector_raises",
#       "anchor": "function-test-ingest-missing-vector-raises",
#       "kind": "function"
#     },
#     {
#       "id": "test-faiss-index-uses-registry-bridge",
#       "name": "test_faiss_index_uses_registry_bridge",
#       "anchor": "function-test-faiss-index-uses-registry-bridge",
#       "kind": "function"
#     },
#     {
#       "id": "build-config",
#       "name": "_build_config",
#       "anchor": "function-build-config",
#       "kind": "function"
#     },
#     {
#       "id": "real-dataset",
#       "name": "real_dataset",
#       "anchor": "function-real-dataset",
#       "kind": "function"
#     },
#     {
#       "id": "to-documents",
#       "name": "_to_documents",
#       "anchor": "function-to-documents",
#       "kind": "function"
#     },
#     {
#       "id": "stack",
#       "name": "stack",
#       "anchor": "function-stack",
#       "kind": "function"
#     },
#     {
#       "id": "test-real-fixture-ingest-and-search",
#       "name": "test_real_fixture_ingest_and_search",
#       "anchor": "function-test-real-fixture-ingest-and-search",
#       "kind": "function"
#     },
#     {
#       "id": "test-real-fixture-reingest-and-reports",
#       "name": "test_real_fixture_reingest_and_reports",
#       "anchor": "function-test-real-fixture-reingest-and-reports",
#       "kind": "function"
#     },
#     {
#       "id": "test-real-fixture-api-roundtrip",
#       "name": "test_real_fixture_api_roundtrip",
#       "anchor": "function-test-real-fixture-api-roundtrip",
#       "kind": "function"
#     },
#     {
#       "id": "test-remove-ids-cpu-fallback",
#       "name": "test_remove_ids_cpu_fallback",
#       "anchor": "function-test-remove-ids-cpu-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "scale-dataset",
#       "name": "scale_dataset",
#       "anchor": "function-scale-dataset",
#       "kind": "function"
#     },
#     {
#       "id": "build-config",
#       "name": "_build_config",
#       "anchor": "function-build-config",
#       "kind": "function"
#     },
#     {
#       "id": "scale-stack",
#       "name": "scale_stack",
#       "anchor": "function-scale-stack",
#       "kind": "function"
#     },
#     {
#       "id": "test-hybrid-scale-suite",
#       "name": "test_hybrid_scale_suite",
#       "anchor": "function-test-hybrid-scale-suite",
#       "kind": "function"
#     },
#     {
#       "id": "toy-data",
#       "name": "_toy_data",
#       "anchor": "function-toy-data",
#       "kind": "function"
#     },
#     {
#       "id": "target-device",
#       "name": "_target_device",
#       "anchor": "function-target-device",
#       "kind": "function"
#     },
#     {
#       "id": "make-id-resolver",
#       "name": "_make_id_resolver",
#       "anchor": "function-make-id-resolver",
#       "kind": "function"
#     },
#     {
#       "id": "emit-vectors",
#       "name": "_emit_vectors",
#       "anchor": "function-emit-vectors",
#       "kind": "function"
#     },
#     {
#       "id": "assert-gpu-index",
#       "name": "_assert_gpu_index",
#       "anchor": "function-assert-gpu-index",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-flat-end-to-end",
#       "name": "test_gpu_flat_end_to_end",
#       "anchor": "function-test-gpu-flat-end-to-end",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-ivf-flat-build-and-search",
#       "name": "test_gpu_ivf_flat_build_and_search",
#       "anchor": "function-test-gpu-ivf-flat-build-and-search",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-ivfpq-build-and-search",
#       "name": "test_gpu_ivfpq_build_and_search",
#       "anchor": "function-test-gpu-ivfpq-build-and-search",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-cosine-against-corpus",
#       "name": "test_gpu_cosine_against_corpus",
#       "anchor": "function-test-gpu-cosine-against-corpus",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-clone-strict-coarse-quantizer",
#       "name": "test_gpu_clone_strict_coarse_quantizer",
#       "anchor": "function-test-gpu-clone-strict-coarse-quantizer",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-near-duplicate-detection-filters-duplicates",
#       "name": "test_gpu_near_duplicate_detection_filters_duplicates",
#       "anchor": "function-test-gpu-near-duplicate-detection-filters-duplicates",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-nprobe-applied-during-search",
#       "name": "test_gpu_nprobe_applied_during_search",
#       "anchor": "function-test-gpu-nprobe-applied-during-search",
#       "kind": "function"
#     },
#     {
#       "id": "test-gpu-similarity-uses-supplied-device",
#       "name": "test_gpu_similarity_uses_supplied_device",
#       "anchor": "function-test-gpu-similarity-uses-supplied-device",
#       "kind": "function"
#     },
#     {
#       "id": "test-operations-shim-emits-warning-and-reexports",
#       "name": "test_operations_shim_emits_warning_and_reexports",
#       "anchor": "function-test-operations-shim-emits-warning-and-reexports",
#       "kind": "function"
#     },
#     {
#       "id": "test-results-shim-emits-warning-and-reexports",
#       "name": "test_results_shim_emits_warning_and_reexports",
#       "anchor": "function-test-results-shim-emits-warning-and-reexports",
#       "kind": "function"
#     },
#     {
#       "id": "test-similarity-shim-emits-warning-and-reexports",
#       "name": "test_similarity_shim_emits_warning_and_reexports",
#       "anchor": "function-test-similarity-shim-emits-warning-and-reexports",
#       "kind": "function"
#     },
#     {
#       "id": "test-retrieval-shim-emits-warning-and-reexports",
#       "name": "test_retrieval_shim_emits_warning_and_reexports",
#       "anchor": "function-test-retrieval-shim-emits-warning-and-reexports",
#       "kind": "function"
#     },
#     {
#       "id": "test-schema-shim-emits-warning-and-reexports",
#       "name": "test_schema_shim_emits_warning_and_reexports",
#       "anchor": "function-test-schema-shim-emits-warning-and-reexports",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import random
import sys
import uuid
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
import pytest

import DocsToKG.HybridSearch.service as service_module
from DocsToKG.DocParsing.embedding.runtime import (
    VECTOR_SCHEMA_VERSION,
    create_vector_writer,
)
from DocsToKG.HybridSearch import (
    ChunkIngestionPipeline,
    DocumentInput,
    HybridSearchAPI,
    HybridSearchConfigManager,
    HybridSearchService,
    HybridSearchValidator,
    Observability,
)
from DocsToKG.HybridSearch.config import (
    DenseIndexConfig,
    FusionConfig,
    HybridSearchConfig,
    RetrievalConfig,
)
from DocsToKG.HybridSearch.devtools.features import FeatureGenerator, tokenize
from DocsToKG.HybridSearch.devtools.opensearch_simulator import (
    OpenSearchSchemaManager,
    OpenSearchSimulator,
)
from DocsToKG.HybridSearch.pipeline import IngestError, RetryableIngestError
from DocsToKG.HybridSearch.service import (
    AdaptiveDensePlanner,
    RequestValidationError,
    ResultShaper,
    build_stats_snapshot,
    infer_embedding_dim,
    load_dataset,
    should_rebuild_index,
    verify_pagination,
)
from DocsToKG.HybridSearch.store import (
    ChunkRegistry,
    FaissSearchResult,
    FaissVectorStore,
    ManagedFaissAdapter,
    cosine_against_corpus_gpu,
    pairwise_inner_products,
    restore_state,
    serialize_state,
)
from DocsToKG.HybridSearch.types import (
    ChunkFeatures,
    ChunkPayload,
    EmbeddingProxy,
    HybridSearchDiagnostics,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
)
from tests.conftest import PatchManager

faiss = pytest.importorskip("faiss")
if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
    pytest.skip(
        "Hybrid search integration suite requires CUDA-enabled faiss", allow_module_level=True
    )

REAL_VECTOR_MARK = pytest.mark.real_vectors

GPU_MARK = pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="FAISS GPU device required")


# --- test_hybrid_search.py ---


def _build_config(tmp_path: Path) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": 3},
        "fusion": {
            "k0": 50.0,
            "mmr_lambda": 0.7,
            "cosine_dedupe_threshold": 0.95,
            "max_chunks_per_doc": 2,
            "strict_highlights": False,
        },
        "retrieval": {"bm25_top_k": 20, "splade_top_k": 20, "dense_top_k": 20},
    }
    path = tmp_path / "hybrid_config.json"
    path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(path)


# --- test_hybrid_search.py ---


@pytest.fixture
def dataset() -> Sequence[Mapping[str, object]]:
    return load_dataset(Path("tests/data/hybrid_dataset.jsonl"))


def test_infer_embedding_dim_returns_after_first_valid_vector(tmp_path: Path) -> None:
    vector_path = tmp_path / "mock_vectors.jsonl"
    with vector_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"Qwen3-4B": {"vector": [0.0, 1.0, 2.0]}}) + "\n")
        # Subsequent lines are intentionally invalid JSON to prove we exit early.
        for _ in range(1000):
            handle.write("not-json\n")

    dataset = [{"document": {"vector_file": str(vector_path)}}]

    assert infer_embedding_dim(dataset) == 3


def test_infer_embedding_dim_handles_underscore_key(tmp_path: Path) -> None:
    vector_path = tmp_path / "mock_vectors_underscore.jsonl"
    with vector_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"Qwen3_4B": {"vector": [0.0, 1.0, 2.0, 3.0]}}) + "\n")

    dataset = [{"document": {"vector_file": str(vector_path)}}]

    assert infer_embedding_dim(dataset) == 4


# --- test_hybrid_search.py ---


@pytest.fixture
def stack(
    tmp_path: Path,
) -> Callable[
    ...,
    tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FeatureGenerator,
        OpenSearchSimulator,
    ],
]:
    def factory(
        *,
        force_remove_ids_fallback: bool = False,
    ) -> tuple[
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
        dense_config = replace(config.dense, force_remove_ids_fallback=force_remove_ids_fallback)
        faiss_index = FaissVectorStore(dim=feature_generator.embedding_dim, config=dense_config)
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
            faiss_index=ManagedFaissAdapter(faiss_index),
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


# --- test_hybrid_search.py ---


def _to_documents(
    entries: Sequence[Mapping[str, object]], vector_format: str | None = None
) -> List[DocumentInput]:
    documents: List[DocumentInput] = []
    format_override = str(vector_format or "").lower()
    for entry in entries:
        doc = entry["document"]
        entry_format = str(doc.get("vector_format") or "").lower()
        vector_files = doc.get("vector_files")
        effective_format = (
            format_override
            if format_override in {"jsonl", "parquet"}
            else entry_format if entry_format in {"jsonl", "parquet"} else ""
        )
        vector_value = doc.get("vector_file")
        if isinstance(vector_files, Mapping):
            candidate = vector_files.get(effective_format) or (
                vector_files.get("jsonl") if effective_format == "" else None
            )
            if candidate is not None:
                vector_value = candidate
        vector_path = Path(str(vector_value))
        if not isinstance(vector_files, Mapping):
            if effective_format == "jsonl":
                vector_path = vector_path.with_suffix(".jsonl")
            elif effective_format == "parquet":
                vector_path = vector_path.with_suffix(".parquet")
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


# --- test_hybrid_search.py ---


def _write_document_artifacts(
    base_dir: Path,
    *,
    doc_id: str,
    namespace: str,
    text: str,
    metadata: Mapping[str, object],
    feature_generator: FeatureGenerator,
    vector_format: str = "jsonl",
    chunk_span: Optional[Tuple[int, int]] = None,
    chunk_span_field: str = "char_offset",
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
    if chunk_span is not None:
        chunk_payload[chunk_span_field] = list(chunk_span)
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
        "SPLADEv3": {
            "model_id": "naver/splade-v3",
            "tokens": [token for token, _ in sorted_splade],
            "weights": [float(weight) for _, weight in sorted_splade],
        },
        "Qwen3-4B": {
            "model_id": "Qwen/Qwen3-Embedding-4B",
            "vector": [float(x) for x in features.embedding.tolist()],
            "dimension": len(features.embedding.tolist()),
        },
        "model_metadata": {},
        "schema_version": VECTOR_SCHEMA_VERSION,
    }
    suffix = "parquet" if vector_format == "parquet" else "jsonl"
    vector_path = vector_dir / f"{doc_id}.vectors.{suffix}"
    if vector_format == "parquet":
        writer = create_vector_writer(vector_path, vector_format)
        with writer:
            writer.write_rows([vector_entry])
    else:
        vector_path.write_text(json.dumps(vector_entry) + "\n", encoding="utf-8")

    return DocumentInput(
        doc_id=doc_id,
        namespace=namespace,
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata=dict(metadata),
    )


def test_load_precomputed_chunks_preserves_char_span(
    stack: Callable[
        ...,
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
    artifacts_dir = tmp_path / "document_with_span"
    span = (5, 42)
    document = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-span",
        namespace="research",
        text="Chunk spans should persist.",
        metadata={},
        feature_generator=feature_generator,
        chunk_span=span,
        chunk_span_field="char_range",
    )

    loaded = ingestion._load_precomputed_chunks(document)

    assert len(loaded) == 1
    assert loaded[0].char_offset == span


# --- test_hybrid_search.py ---


def test_commit_batch_handles_empty_document(
    stack: Callable[
        ...,
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
    ingestion, _, _, _, _, _ = stack()
    artifacts_dir = tmp_path / "empty_document"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = artifacts_dir / "doc-0.chunks.jsonl"
    chunk_path.write_text("", encoding="utf-8")
    vector_path = artifacts_dir / "doc-0.vectors.jsonl"
    vector_path.write_text("", encoding="utf-8")
    document = DocumentInput(
        doc_id="doc-0",
        namespace="research",
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata={},
    )

    loaded = ingestion._load_precomputed_chunks(document)
    assert isinstance(loaded, list)
    assert loaded == []

    result = ingestion._commit_batch(loaded, collect_vector_ids=True)
    assert result.chunk_count == 0
    assert result.namespaces == ()
    assert result.vector_ids == ()


def test_commit_batch_handles_populated_document(
    stack: Callable[
        ...,
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
    artifacts_dir = tmp_path / "populated_document"
    document = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-1",
        namespace="research",
        text="FAISS ingestion smoke test.",
        metadata={"author": "Test"},
        feature_generator=feature_generator,
    )

    loaded = ingestion._load_precomputed_chunks(document)
    assert isinstance(loaded, list)
    assert len(loaded) == 1

    result = ingestion._commit_batch(loaded, collect_vector_ids=True)
    assert result.chunk_count == 1
    assert result.namespaces == ("research",)
    assert result.vector_ids == (loaded[0].vector_id,)


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_hybrid_retrieval_end_to_end(
    stack: Callable[
        ...,
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
    vector_format: str,
) -> None:
    ingestion, service, registry, _, _, _ = stack()
    documents = _to_documents(dataset, vector_format)
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


# --- test_hybrid_search.py ---


def test_reingest_updates_dense_and_sparse_channels(
    stack: Callable[
        ...,
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


# --- test_hybrid_search.py ---


def test_validation_harness_reports(
    stack: Callable[
        ...,
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


def test_validator_validation_resources_honor_null_stream_flags(
    patcher: PatchManager, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="DocsToKG.HybridSearch")

    class StubConfigManager:
        def __init__(self, config: object) -> None:
            self._config = config

        def get(self) -> object:
            return self._config

    class StubRouter:
        def iter_stores(self) -> Sequence[tuple[str, object]]:
            return []

    dense_config = replace(
        DenseIndexConfig(),
        gpu_use_default_null_stream_all_devices=True,
        gpu_temp_memory_bytes=None,
        gpu_pinned_memory_bytes=None,
    )
    config = SimpleNamespace(dense=dense_config)
    service = SimpleNamespace(
        _config_manager=StubConfigManager(config),
        _observability=Observability(),
        _faiss_router=StubRouter(),
    )
    validator = HybridSearchValidator(
        ingestion=SimpleNamespace(),
        service=service,
        registry=SimpleNamespace(),
        opensearch=SimpleNamespace(),
    )

    class RecordingResource:
        def __init__(self) -> None:
            self.temp_memory_calls: list[int] = []
            self.pinned_memory_calls: list[int] = []
            self.null_stream_calls: list[object | None] = []
            self.null_stream_all_calls = 0

        def setTempMemory(self, value: int) -> None:  # pragma: no cover - stub
            self.temp_memory_calls.append(value)

        def setPinnedMemory(self, value: int) -> None:  # pragma: no cover - stub
            self.pinned_memory_calls.append(value)

        def setDefaultNullStreamAllDevices(self) -> None:  # pragma: no cover - stub
            self.null_stream_all_calls += 1

        def setDefaultNullStream(
            self, device: object | None = None
        ) -> None:  # pragma: no cover - stub
            self.null_stream_calls.append(device)

    stub_faiss = SimpleNamespace(StandardGpuResources=RecordingResource)
    patcher.setattr(service_module, "faiss", stub_faiss, raising=False)

    resource = validator._ensure_validation_resources()
    assert isinstance(resource, RecordingResource)
    assert resource.null_stream_all_calls == 1
    assert resource.null_stream_calls == []

    gauges = {
        sample.name: sample.value for sample in service._observability.metrics.export_gauges()
    }
    assert gauges.get("faiss_gpu_default_null_stream_all_devices") == 1.0
    assert gauges.get("faiss_gpu_default_null_stream") == 0.0

    assert any(record.message == "faiss-gpu-resource-configured" for record in caplog.records)


# --- test_hybrid_search.py ---


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


# --- test_hybrid_search.py ---


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_api_post_hybrid_search_success_and_validation(
    stack: Callable[
        ...,
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
    vector_format: str,
) -> None:
    ingestion, service, _, _, _, _ = stack()
    documents = _to_documents(dataset, vector_format)
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


# --- test_hybrid_search.py ---


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_api_post_hybrid_search_allows_null_filters(
    stack: Callable[
        ...,
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
    vector_format: str,
) -> None:
    ingestion, service, _, _, _, _ = stack()
    documents = _to_documents(dataset, vector_format)
    ingestion.upsert_documents(documents)
    api = HybridSearchAPI(service)

    status, body = api.post_hybrid_search(
        {
            "query": "hybrid retrieval faiss",
            "namespace": "research",
            "page_size": 3,
            "filters": None,
        }
    )

    assert status == HTTPStatus.OK
    assert body["results"]


# --- test_hybrid_search.py ---


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_api_post_hybrid_search_omits_diagnostics_when_disabled(
    stack: Callable[
        ...,
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
    vector_format: str,
) -> None:
    ingestion, service, _, _, _, _ = stack()
    documents = _to_documents(dataset, vector_format)
    ingestion.upsert_documents(documents)
    api = HybridSearchAPI(service)

    status, body = api.post_hybrid_search(
        {
            "query": "hybrid retrieval faiss",
            "namespace": "research",
            "page_size": 3,
            "diagnostics": False,
        }
    )

    assert status == HTTPStatus.OK
    assert body["results"]
    assert all("diagnostics" not in result for result in body["results"])


# --- test_hybrid_search.py ---


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_operations_snapshot_and_restore_roundtrip(
    stack: Callable[
        ...,
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
    vector_format: str,
) -> None:
    ingestion, service, registry, _, _, opensearch = stack()
    documents = _to_documents(dataset, vector_format)
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


def test_verify_pagination_preserves_recall_first() -> None:
    responses = [
        HybridSearchResponse(
            results=[
                HybridSearchResult(
                    doc_id="doc-0",
                    chunk_id="chunk-0",
                    vector_id="vec-0",
                    namespace="test",
                    score=1.0,
                    fused_rank=0,
                    text="page 0",
                    highlights=(),
                    provenance_offsets=(),
                    diagnostics=HybridSearchDiagnostics(),
                    metadata={},
                )
            ],
            next_cursor="cursor-1",
            total_candidates=1,
            timings_ms={},
        ),
        HybridSearchResponse(
            results=[
                HybridSearchResult(
                    doc_id="doc-1",
                    chunk_id="chunk-1",
                    vector_id="vec-1",
                    namespace="test",
                    score=1.0,
                    fused_rank=1,
                    text="page 1",
                    highlights=(),
                    provenance_offsets=(),
                    diagnostics=HybridSearchDiagnostics(),
                    metadata={},
                )
            ],
            next_cursor="cursor-2",
            total_candidates=1,
            timings_ms={},
        ),
        HybridSearchResponse(
            results=[
                HybridSearchResult(
                    doc_id="doc-2",
                    chunk_id="chunk-2",
                    vector_id="vec-2",
                    namespace="test",
                    score=1.0,
                    fused_rank=2,
                    text="page 2",
                    highlights=(),
                    provenance_offsets=(),
                    diagnostics=HybridSearchDiagnostics(),
                    metadata={},
                )
            ],
            next_cursor=None,
            total_candidates=1,
            timings_ms={},
        ),
    ]

    class RecordingService:
        def __init__(self, pages: Sequence[HybridSearchResponse]) -> None:
            self._pages = list(pages)
            self._index = 0
            self.requests: list[HybridSearchRequest] = []

        def search(self, search_request: HybridSearchRequest) -> HybridSearchResponse:
            self.requests.append(search_request)
            page = self._pages[self._index]
            self._index += 1
            return page

    service = RecordingService(responses)
    request = HybridSearchRequest(
        query="recall",
        namespace="demo",
        filters={},
        page_size=1,
        recall_first=True,
    )

    pagination = verify_pagination(service, request)

    assert pagination.cursor_chain == ["cursor-1", "cursor-2"]
    assert not pagination.duplicate_detected
    assert len(service.requests) == 3
    assert all(call.recall_first for call in service.requests)


def test_recall_first_dense_signature_isolated(
    stack: Callable[
        ...,
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

    base_request = HybridSearchRequest(
        query="hybrid retrieval",
        namespace="research",
        filters={},
        page_size=3,
    )
    recall_request = replace(base_request, recall_first=True)

    def signature_for(request: HybridSearchRequest) -> tuple[object, ...]:
        active_filters: MutableMapping[str, object] = dict(request.filters)
        if request.namespace:
            active_filters["namespace"] = request.namespace
        return service._dense_request_signature(request, active_filters)

    service.search(recall_request)

    recall_signature = signature_for(recall_request)
    normal_signature = signature_for(base_request)

    assert service._dense_strategy.has_cache(recall_signature)
    assert not service._dense_strategy.has_cache(normal_signature)
    assert normal_signature not in service._dense_strategy._signature_pass

    service.search(base_request)

    assert service._dense_strategy.has_cache(normal_signature)
    assert normal_signature in service._dense_strategy._signature_pass
    assert normal_signature != recall_signature


def test_recall_first_does_not_mutate_global_pass_rate(
    stack: Callable[
        ...,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion, service, _, _, _, _ = stack()
    documents = _to_documents(dataset)
    ingestion.upsert_documents(documents)

    strategy = service._dense_strategy
    initial_pass = strategy.current_pass_rate()

    recall_request = HybridSearchRequest(
        query="hybrid retrieval faiss",
        namespace="research",
        filters={"author": "Unknown"},
        page_size=3,
        recall_first=True,
    )

    service.search(recall_request)

    assert strategy.current_pass_rate() == pytest.approx(initial_pass)

    standard_request = replace(recall_request, filters={}, recall_first=False)
    observed_pass_rates: List[float] = []
    original_plan = strategy.plan

    def capture_plan(
        signature: tuple[object, ...],
        *,
        page_size: int,
        retrieval_cfg: "RetrievalConfig",
        dense_cfg: "DenseIndexConfig",
        min_k: int = 0,
    ) -> tuple[int, float, float]:
        observed_pass_rates.append(strategy.current_pass_rate())
        return original_plan(
            signature,
            page_size=page_size,
            retrieval_cfg=retrieval_cfg,
            dense_cfg=dense_cfg,
            min_k=min_k,
        )

    monkeypatch.setattr(strategy, "plan", capture_plan)
    service.search(standard_request)

    assert observed_pass_rates, "planner should execute for standard request"
    assert observed_pass_rates[0] == pytest.approx(initial_pass)


def test_cursor_rejects_when_recall_first_mismatch() -> None:
    service = object.__new__(HybridSearchService)
    page_size = 2
    base_results = [
        HybridSearchResult(
            doc_id=f"doc-{idx}",
            chunk_id=f"chunk-{idx}",
            vector_id=f"vec-{idx}",
            namespace="demo",
            score=float(10 - idx),
            fused_rank=idx,
            text=f"chunk {idx}",
            highlights=(),
            provenance_offsets=(),
            diagnostics=HybridSearchDiagnostics(),
            metadata={},
        )
        for idx in range(3)
    ]

    recall_request = HybridSearchRequest(
        query="demo query",
        namespace="demo",
        filters={},
        page_size=page_size,
        recall_first=True,
    )
    filters = {"namespace": recall_request.namespace}
    recall_fingerprint = service._cursor_fingerprint(recall_request, filters)
    cursor = service._build_cursor(base_results, page_size, recall_fingerprint, True)
    assert cursor is not None

    sliced = service._slice_from_cursor(
        base_results,
        cursor,
        page_size,
        recall_fingerprint,
        True,
    )
    assert [result.vector_id for result in sliced] == ["vec-2"]

    followup_request = HybridSearchRequest(
        query="demo query",
        namespace="demo",
        filters={},
        page_size=page_size,
        cursor=cursor,
        recall_first=False,
    )
    followup_fingerprint = service._cursor_fingerprint(followup_request, filters)
    assert followup_fingerprint != recall_fingerprint

    with pytest.raises(RequestValidationError):
        service._slice_from_cursor(
            base_results,
            cursor,
            page_size,
            followup_fingerprint,
            False,
        )


# --- test_hybrid_search.py ---


def test_execute_dense_uses_range_search_for_recall_first_without_score_floor() -> None:
    request = HybridSearchRequest(
        query="dense recall",
        namespace="demo",
        filters={},
        page_size=2,
        recall_first=True,
    )
    config = HybridSearchConfig()
    query_features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.array([0.1, 0.2], dtype=np.float32),
    )
    chunk_features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.array([0.3, 0.4], dtype=np.float32),
    )
    chunk = ChunkPayload(
        doc_id="doc-0",
        chunk_id="chunk-0",
        vector_id="vec-0",
        namespace="demo",
        text="dense chunk",
        metadata={"tags": ["dense"]},
        features=chunk_features,
        token_count=10,
        source_chunk_idxs=[0],
        doc_items_refs=["doc:0"],
    )

    class RecordingRegistry:
        def __init__(self, payload: ChunkPayload) -> None:
            self._payloads = {payload.vector_id: payload}
            self._dim = int(payload.features.embedding.shape[-1])
            self.resolve_embeddings_calls: List[Tuple[str, ...]] = []

        def bulk_get(self, vector_ids: Sequence[str]) -> Sequence[ChunkPayload]:
            return [
                self._payloads[vector_id]
                for vector_id in vector_ids
                if vector_id in self._payloads
            ]

        def resolve_embeddings(
            self,
            vector_ids: Sequence[str],
            *,
            cache: Optional[Dict[str, np.ndarray]] = None,
            dtype: np.dtype = np.float32,
        ) -> np.ndarray:
            self.resolve_embeddings_calls.append(tuple(vector_ids))
            rows: List[np.ndarray] = []
            for vector_id in vector_ids:
                embedding = np.asarray(
                    self._payloads[vector_id].features.embedding, dtype=np.float32
                )
                rows.append(embedding)
                if cache is not None:
                    cache[vector_id] = embedding
            if not rows:
                return np.empty((0, self._dim), dtype=dtype)
            return np.ascontiguousarray(np.stack(rows), dtype=dtype)

        def resolve_embedding(
            self,
            vector_id: str,
            *,
            cache: Optional[Dict[str, np.ndarray]] = None,
            dtype: np.dtype = np.float32,
        ) -> np.ndarray:
            matrix = self.resolve_embeddings([vector_id], cache=cache, dtype=dtype)
            return matrix[0]

    class RecordingDenseStore:
        def __init__(self, hits: Sequence[FaissSearchResult]) -> None:
            self._hits = list(hits)
            self.range_calls: List[Tuple[np.ndarray, float, Optional[int]]] = []
            self.search_batch_calls: List[Tuple[np.ndarray, int]] = []
            self.adapter_stats = SimpleNamespace(nprobe=1, fp16_enabled=False)

        def range_search(
            self, query: np.ndarray, score_floor: float, limit: Optional[int] = None
        ) -> Sequence[FaissSearchResult]:
            self.range_calls.append((np.asarray(query), float(score_floor), limit))
            return list(self._hits)

        def search_batch(self, queries: np.ndarray, depth: int) -> List[List[FaissSearchResult]]:
            self.search_batch_calls.append((np.asarray(queries), int(depth)))
            return [self._hits[:depth]]

    registry = RecordingRegistry(chunk)
    store = RecordingDenseStore([FaissSearchResult(vector_id="vec-0", score=0.42)])
    service = object.__new__(HybridSearchService)
    service._dense_strategy = service_module.DenseSearchStrategy()  # type: ignore[attr-defined]
    service._observability = Observability()  # type: ignore[attr-defined]
    service._registry = registry  # type: ignore[attr-defined]

    timings: Dict[str, float] = {}
    result = service._execute_dense(
        request=request,
        filters={},
        config=config,
        query_features=query_features,
        timings=timings,
        store=store,
    )

    assert store.range_calls, "range_search should be invoked when recall_first is set"
    assert not store.search_batch_calls
    assert isinstance(result, service_module.ChannelResults)
    assert [candidate.chunk.vector_id for candidate in result.candidates] == ["vec-0"]
    assert timings["dense_ms"] >= 0.0
    assert registry.resolve_embeddings_calls == [("vec-0",)]
    assert result.embeddings is not None
    np.testing.assert_allclose(result.embeddings, chunk_features.embedding[None, :])


def test_recall_first_pass_rate_isolated_from_default_planner() -> None:
    class _MetricsStub:
        def observe(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_gauge(self, *_args: object, **_kwargs: object) -> None:
            return None

        def increment(self, *_args: object, **_kwargs: object) -> None:
            return None

        def percentile(self, *_args: object, **_kwargs: object) -> Optional[float]:
            return None

    class _RegistryStub:
        def __init__(self, payload: ChunkPayload) -> None:
            self._payload = payload
            self._embedding = payload.features.embedding

        def bulk_get(self, vector_ids: Sequence[str]) -> Sequence[ChunkPayload]:
            return [
                self._payload
                for vector_id in vector_ids
                if vector_id == self._payload.vector_id
            ]

        def resolve_embeddings(
            self, vector_ids: Sequence[str], *, cache: Optional[Dict[str, np.ndarray]] = None
        ) -> np.ndarray:
            rows = [
                self._embedding
                for vector_id in vector_ids
                if vector_id == self._payload.vector_id
            ]
            if not rows:
                return np.empty((0, self._embedding.shape[-1]), dtype=np.float32)
            return np.vstack(rows)

        def resolve_embedding(
            self, vector_id: str, *, cache: Optional[Dict[str, np.ndarray]] = None
        ) -> np.ndarray:
            return self._embedding

    class _StoreStub:
        def __init__(
            self,
            range_hits: Sequence[FaissSearchResult],
            batch_hits: Sequence[FaissSearchResult],
        ) -> None:
            self._range_hits = list(range_hits)
            self._batch_hits = list(batch_hits)
            self.range_calls = 0
            self.batch_depths: List[int] = []

        def range_search(
            self, _query: np.ndarray, _score_floor: float, *, limit: Optional[int] = None
        ) -> Sequence[FaissSearchResult]:
            self.range_calls += 1
            return list(self._range_hits)

        def search_batch(
            self, _queries: np.ndarray, depth: int
        ) -> Sequence[Sequence[FaissSearchResult]]:
            self.batch_depths.append(depth)
            return [list(self._batch_hits)]

    service = object.__new__(HybridSearchService)
    planner = AdaptiveDensePlanner(alpha=0.4, initial_pass_rate=0.5)
    service._dense_strategy = planner  # type: ignore[attr-defined]
    service._observability = SimpleNamespace(metrics=_MetricsStub())  # type: ignore[attr-defined]

    chunk_features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    chunk = ChunkPayload(
        doc_id="doc-0",
        chunk_id="chunk-0",
        vector_id="vec-keep",
        namespace="demo",
        text="dense chunk",
        metadata={},
        features=chunk_features,
        token_count=5,
        source_chunk_idxs=[0],
        doc_items_refs=["doc:0"],
    )
    service._registry = _RegistryStub(chunk)  # type: ignore[attr-defined]

    config = HybridSearchConfig(
        retrieval=RetrievalConfig(
            dense_top_k=3,
            dense_overfetch_factor=1.0,
            dense_oversample=1.0,
        )
    )
    recall_request = HybridSearchRequest(
        query="dense recall",
        namespace="demo",
        filters={},
        page_size=2,
        recall_first=True,
    )
    standard_request = replace(recall_request, recall_first=False)
    filters: Dict[str, object] = {}
    query_features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.array([0.2, 0.4, 0.6], dtype=np.float32),
    )

    range_hits = [
        FaissSearchResult(vector_id="vec-keep", score=0.95)
    ] + [
        FaissSearchResult(vector_id=f"vec-miss-{idx}", score=0.05)
        for idx in range(19)
    ]
    batch_hits = [
        FaissSearchResult(vector_id="vec-keep", score=0.9),
        FaissSearchResult(vector_id="vec-miss", score=0.2),
    ]
    store = _StoreStub(range_hits, batch_hits)

    normal_signature = service._dense_request_signature(standard_request, filters)
    planned_before = planner.plan(
        normal_signature,
        page_size=standard_request.page_size,
        retrieval_cfg=config.retrieval,
        dense_cfg=config.dense,
    )[0]
    initial_pass = planner.current_pass_rate()

    service._execute_dense(
        recall_request,
        filters,
        config,
        query_features,
        timings={},
        store=store,
    )

    assert store.range_calls == 1
    assert planner.current_pass_rate() == pytest.approx(initial_pass, rel=1e-6)

    planned_after = planner.plan(
        normal_signature,
        page_size=standard_request.page_size,
        retrieval_cfg=config.retrieval,
        dense_cfg=config.dense,
    )[0]
    assert planned_after == planned_before

    service._execute_dense(
        standard_request,
        filters,
        config,
        query_features,
        timings={},
        store=store,
    )

    assert store.batch_depths[0] == planned_before

def test_recall_first_range_search_respects_initial_k_budget() -> None:
    request = HybridSearchRequest(
        query="dense recall budget",
        namespace="demo",
        filters={},
        page_size=8,
        recall_first=True,
    )
    config = HybridSearchConfig()
    query_features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.array([0.5, 0.25], dtype=np.float32),
    )

    class RecordingRegistry:
        def __init__(self, payloads: Mapping[str, ChunkPayload]) -> None:
            self._payloads = dict(payloads)
            self._dim = next(iter(payloads.values())).features.embedding.shape[-1]
            self.resolve_embeddings_calls: List[Tuple[str, ...]] = []

        def bulk_get(self, vector_ids: Sequence[str]) -> Sequence[ChunkPayload]:
            return [
                self._payloads[vector_id]
                for vector_id in vector_ids
                if vector_id in self._payloads
            ]

        def resolve_embeddings(
            self,
            vector_ids: Sequence[str],
            *,
            cache: Optional[Dict[str, np.ndarray]] = None,
            dtype: np.dtype = np.float32,
        ) -> np.ndarray:
            self.resolve_embeddings_calls.append(tuple(vector_ids))
            rows: List[np.ndarray] = []
            for vector_id in vector_ids:
                embedding = np.asarray(
                    self._payloads[vector_id].features.embedding, dtype=dtype
                )
                rows.append(embedding)
                if cache is not None:
                    cache[vector_id] = embedding
            if not rows:
                return np.empty((0, self._dim), dtype=dtype)
            return np.ascontiguousarray(np.stack(rows), dtype=dtype)

    class RecordingDenseStore:
        def __init__(self, hits: Sequence[FaissSearchResult]) -> None:
            self._hits = list(hits)
            self.range_calls: List[Tuple[np.ndarray, float, Optional[int]]] = []
            self.search_batch_calls: List[Tuple[np.ndarray, int]] = []
            self.adapter_stats = SimpleNamespace(nprobe=1, fp16_enabled=False)

        def range_search(
            self, query: np.ndarray, score_floor: float, limit: Optional[int] = None
        ) -> Sequence[FaissSearchResult]:
            self.range_calls.append((np.asarray(query), float(score_floor), limit))
            return list(self._hits)

        def search_batch(self, queries: np.ndarray, depth: int) -> List[List[FaissSearchResult]]:
            self.search_batch_calls.append((np.asarray(queries), int(depth)))
            return [self._hits[:depth]]

    vector_count = service_module.DenseSearchStrategy._MAX_K * 2
    hits = [
        FaissSearchResult(vector_id=f"vec-{idx}", score=1.0 - idx * 1e-4)
        for idx in range(vector_count)
    ]
    payloads: Dict[str, ChunkPayload] = {}
    for idx, hit in enumerate(hits):
        features = ChunkFeatures(
            bm25_terms={},
            splade_weights={},
            embedding=np.full(query_features.embedding.shape, float(idx + 1), dtype=np.float32),
        )
        payloads[hit.vector_id] = ChunkPayload(
            doc_id=f"doc-{idx}",
            chunk_id=f"chunk-{idx}",
            vector_id=hit.vector_id,
            namespace=request.namespace or "",
            text=f"chunk {idx}",
            metadata={},
            features=features,
            token_count=10,
            source_chunk_idxs=[0],
            doc_items_refs=[f"doc:{idx}"],
        )

    registry = RecordingRegistry(payloads)
    store = RecordingDenseStore(hits)
    service = object.__new__(HybridSearchService)
    service._registry = registry  # type: ignore[attr-defined]
    service._dense_strategy = service_module.DenseSearchStrategy()  # type: ignore[attr-defined]
    service._observability = Observability()  # type: ignore[attr-defined]

    filters: Dict[str, object] = {}
    signature = service._dense_request_signature(request, filters)
    expected_budget, _, _ = service._dense_strategy.plan(  # type: ignore[attr-defined]
        signature,
        page_size=max(1, request.page_size),
        retrieval_cfg=config.retrieval,
        dense_cfg=config.dense,
    )

    timings: Dict[str, float] = {}
    result = service._execute_dense(
        request=request,
        filters=filters,
        config=config,
        query_features=query_features,
        timings=timings,
        store=store,
    )

    assert store.range_calls, "range_search should be invoked with a bounded budget"
    assert store.range_calls[0][2] == expected_budget
    assert not store.search_batch_calls
    assert len(result.candidates) == expected_budget
    assert registry.resolve_embeddings_calls
    assert len(registry.resolve_embeddings_calls[0]) == expected_budget

# --- test_hybrid_search.py ---


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_ingest_missing_vector_raises(
    stack: Callable[
        ...,
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
    vector_format: str,
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
        vector_format=vector_format,
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

    with pytest.raises(IngestError) as excinfo:
        ingestion.upsert_documents([doc])

    assert not isinstance(excinfo.value, RetryableIngestError)


# --- test_hybrid_search.py ---


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_ingest_retryable_errors_propagate(
    stack: Callable[
        ...,
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
    vector_format: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion, _, _, _, feature_generator, _ = stack()
    doc = _write_document_artifacts(
        tmp_path / "docs",
        doc_id="doc-transient",
        namespace="research",
        text="Chunk raising retryable error",
        metadata={},
        feature_generator=feature_generator,
        vector_format=vector_format,
    )
    error = RetryableIngestError("transient failure")

    def raise_retryable(*_: object, **__: object) -> None:
        raise error

    monkeypatch.setattr(ingestion, "_commit_batch", raise_retryable)

    with pytest.raises(RetryableIngestError) as excinfo:
        ingestion.upsert_documents([doc])

    assert excinfo.value is error


# --- test_hybrid_search.py ---


def test_ingest_mixed_vector_formats_error(
    stack: Callable[
        ...,
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
    artifacts_dir = tmp_path / "mixed"
    doc_jsonl = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-jsonl",
        namespace="research",
        text="JSONL vector entry",
        metadata={},
        feature_generator=feature_generator,
        vector_format="jsonl",
    )
    doc_parquet = _write_document_artifacts(
        artifacts_dir,
        doc_id="doc-parquet",
        namespace="research",
        text="Parquet vector entry",
        metadata={},
        feature_generator=feature_generator,
        vector_format="parquet",
    )

    with pytest.raises(IngestError) as exc:
        ingestion.upsert_documents([doc_jsonl, doc_parquet])

    assert "Mixed DocParsing vector formats" in str(exc.value)


def test_managed_adapter_supports_ingestion_training_sample() -> None:
    """Managed adapter should expose config/dim for ingestion helpers."""

    class _StubLexicalIndex:
        def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:  # pragma: no cover - stub
            self._last_upsert = list(chunks)

        def bulk_delete(self, vector_ids: Sequence[str]) -> None:  # pragma: no cover - stub
            self._last_delete = list(vector_ids)

        def search_bm25(self, *args, **kwargs):  # pragma: no cover - stub
            return [], None

        def search_splade(self, *args, **kwargs):  # pragma: no cover - stub
            return [], None

        def search_bm25_true(self, *args, **kwargs):  # pragma: no cover - stub
            return [], None

        def highlight(
            self, chunk: ChunkPayload, query_tokens: Sequence[str]
        ) -> List[str]:  # pragma: no cover - stub
            return []

        def stats(self):  # pragma: no cover - stub
            return {}

    class _StubFaissStore:
        def __init__(self, *, dim: int, config: DenseIndexConfig) -> None:
            self._dim = dim
            self._config = config
            self._needs_training = True
            self.config_accesses = 0
            self.adapter_stats = SimpleNamespace(device=0, ntotal=0)
            self._store: Dict[str, np.ndarray] = {}

        @property
        def config(self) -> DenseIndexConfig:
            self.config_accesses += 1
            return self._config

        @property
        def dim(self) -> int:
            return self._dim

        @property
        def device(self) -> int:  # pragma: no cover - unused fallback
            return 0

        @property
        def ntotal(self) -> int:  # pragma: no cover - unused fallback
            return 0

        def set_id_resolver(self, resolver):  # pragma: no cover - stub
            self._resolver = resolver

        def needs_training(self) -> bool:
            return self._needs_training

        def train(self, vectors: Sequence[np.ndarray]) -> None:
            self._needs_training = False
            self.trained = list(vectors)

        def add(self, vectors, vector_ids):  # pragma: no cover - stub
            self._last_add = (vectors, vector_ids)
            for vec, vid in zip(vectors, vector_ids):
                self._store[str(vid)] = np.asarray(vec, dtype=np.float32)

        def remove(self, vector_ids):  # pragma: no cover - stub
            self._last_remove = list(vector_ids)
            for vid in vector_ids:
                self._store.pop(str(vid), None)

        def search(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def search_many(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def search_batch(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def range_search(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def reconstruct_batch(self, vector_ids: Sequence[str]) -> np.ndarray:
            return np.stack([self._store[str(vid)] for vid in vector_ids], dtype=np.float32)

        def serialize(self) -> bytes:  # pragma: no cover - stub
            return b""

        def restore(self, payload: bytes) -> None:  # pragma: no cover - stub
            self._last_restore = payload

        def flush_snapshot(self, *, reason: str = "flush") -> None:  # pragma: no cover - stub
            self._last_flush_reason = reason

    def _make_chunk(vector_id: str, *, dim: int, value: float) -> ChunkPayload:
        embedding = np.full((dim,), value, dtype=np.float32)
        features = ChunkFeatures(
            bm25_terms={"token": value},
            splade_weights={"token": value * 2},
            embedding=embedding,
        )
        return ChunkPayload(
            doc_id="doc",
            chunk_id=vector_id,
            vector_id=vector_id,
            namespace="ns",
            text="",
            metadata={},
            features=features,
            token_count=0,
            source_chunk_idxs=(),
            doc_items_refs=(),
        )

    dim = 3
    dense_config = DenseIndexConfig(nlist=4, ivf_train_factor=2)
    stub_faiss = _StubFaissStore(dim=dim, config=dense_config)
    adapter = ManagedFaissAdapter(stub_faiss)
    registry = ChunkRegistry()
    observability = Observability()
    pipeline = ChunkIngestionPipeline(
        faiss_index=adapter,
        opensearch=_StubLexicalIndex(),
        registry=registry,
        observability=observability,
    )

    existing = [_make_chunk(f"existing-{i}", dim=dim, value=float(i)) for i in range(2)]
    pipeline.faiss_index.add(
        [chunk.features.embedding for chunk in existing],
        [chunk.vector_id for chunk in existing],
    )
    registry.upsert(existing)
    assert all(
        isinstance(chunk.features.embedding, EmbeddingProxy) for chunk in existing
    )
    new_chunks = [_make_chunk(f"new-{i}", dim=dim, value=float(i)) for i in range(2)]

    assert pipeline.faiss_index.config is dense_config
    assert pipeline.faiss_index.dim == dim

    sample = pipeline._training_sample(new_chunks)
    assert stub_faiss.config_accesses >= 1
    assert len(sample) == len(existing) + len(new_chunks)
    assert all(vec.shape == (dim,) for vec in sample)

    payload = {
        "BM25": {"terms": ["token"], "weights": [0.1]},
        "SpladeV3": {"tokens": ["token"], "weights": [0.2]},
        "Qwen3-4B": {"vector": [0.5, 1.5, 2.5]},
    }
    features = pipeline._features_from_vector(payload)
    np.testing.assert_allclose(features.embedding, np.array([0.5, 1.5, 2.5], dtype=np.float32))
    assert features.bm25_terms == {"token": 0.1}
    assert features.splade_weights == {"token": 0.2}


# --- test_hybrid_search.py ---


def test_commit_batch_rolls_back_on_lexical_failure() -> None:
    class _FailingLexicalIndex:
        def __init__(self) -> None:
            self.deleted: Optional[List[str]] = None

        def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:
            raise RuntimeError("lexical failure")

        def bulk_delete(self, vector_ids: Sequence[str]) -> None:
            self.deleted = list(vector_ids)

    class _TrackingFaissStore:
        def __init__(self, *, dim: int) -> None:
            self._dim = dim
            self._config = DenseIndexConfig(index_type="flat")
            self._needs_training = False
            self.adapter_stats = SimpleNamespace(device=0, ntotal=0)
            self._store: Dict[str, np.ndarray] = {}
            self._last_remove: List[str] = []

        @property
        def config(self) -> DenseIndexConfig:
            return self._config

        @property
        def dim(self) -> int:
            return self._dim

        @property
        def device(self) -> int:  # pragma: no cover - unused fallback
            return 0

        @property
        def ntotal(self) -> int:  # pragma: no cover - unused fallback
            return 0

        def set_id_resolver(self, resolver) -> None:
            self._resolver = resolver

        def needs_training(self) -> bool:
            return self._needs_training

        def train(self, vectors: Sequence[np.ndarray]) -> None:  # pragma: no cover - stub
            self._needs_training = False
            self.trained = list(vectors)

        def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
            self._last_add = (list(vectors), list(vector_ids))
            for vec, vid in zip(vectors, vector_ids):
                self._store[str(vid)] = np.asarray(vec, dtype=np.float32)

        def remove(self, vector_ids: Sequence[str]) -> None:
            self._last_remove = list(vector_ids)
            for vid in vector_ids:
                self._store.pop(str(vid), None)

        def search(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def search_many(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def search_batch(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def range_search(self, *args, **kwargs):  # pragma: no cover - stub
            return []

        def reconstruct_batch(self, vector_ids: Sequence[str]) -> np.ndarray:
            if not vector_ids:
                return np.empty((0, self._dim), dtype=np.float32)
            return np.stack([self._store[str(vid)] for vid in vector_ids], dtype=np.float32)

        def serialize(self) -> bytes:  # pragma: no cover - stub
            return b""

        def restore(self, payload: bytes) -> None:  # pragma: no cover - stub
            self._last_restore = payload

        def flush_snapshot(self, *, reason: str = "flush") -> None:  # pragma: no cover - stub
            self._last_flush_reason = reason

    dim = 4
    chunk = ChunkPayload(
        doc_id="doc",
        chunk_id="chunk-0",
        vector_id="vec-0",
        namespace="research",
        text="",
        metadata={},
        features=ChunkFeatures(
            bm25_terms={},
            splade_weights={},
            embedding=np.ones((dim,), dtype=np.float32),
        ),
        token_count=0,
        source_chunk_idxs=(),
        doc_items_refs=(),
    )

    faiss = _TrackingFaissStore(dim=dim)
    adapter = ManagedFaissAdapter(faiss)
    registry = ChunkRegistry()
    observability = Observability()
    lexical = _FailingLexicalIndex()
    pipeline = ChunkIngestionPipeline(
        faiss_index=adapter,
        opensearch=lexical,
        registry=registry,
        observability=observability,
    )

    with pytest.raises(RuntimeError, match="lexical failure"):
        pipeline._commit_batch([chunk])

    assert faiss._store == {}
    assert faiss._last_remove == [chunk.vector_id]
    assert registry.count() == 0
    assert lexical.deleted is None


# --- test_hybrid_search.py ---


def test_faiss_index_uses_registry_bridge(tmp_path: Path) -> None:
    config = DenseIndexConfig(index_type="flat")
    manager = FaissVectorStore(dim=4, config=config)
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


# --- test_hybrid_search_real_vectors.py ---


def test_chunk_registry_lazy_embeddings_roundtrip(tmp_path: Path) -> None:
    dim = 8
    config = DenseIndexConfig(index_type="flat")
    faiss_index = FaissVectorStore(dim=dim, config=config)
    registry = ChunkRegistry()
    observability = Observability()
    pipeline = ChunkIngestionPipeline(
        faiss_index=faiss_index,
        opensearch=OpenSearchSimulator(),
        registry=registry,
        observability=observability,
    )

    chunks = [_make_chunk(f"registry-{i}", dim=dim, value=float(i)) for i in range(6)]
    originals = {chunk.vector_id: chunk.features.embedding.copy() for chunk in chunks}
    pipeline.faiss_index.add(
        [chunk.features.embedding for chunk in chunks],
        [chunk.vector_id for chunk in chunks],
    )

    registry.upsert(chunks)
    assert registry.count() == len(chunks)
    assert all(isinstance(chunk.features.embedding, EmbeddingProxy) for chunk in registry.all())

    pre_bytes = sum(vector.nbytes for vector in originals.values())
    post_bytes = sum(
        getattr(chunk.features.embedding, "nbytes", 0)
        for chunk in registry.all()
        if isinstance(chunk.features.embedding, np.ndarray)
    )
    assert pre_bytes > 0 and post_bytes == 0

    ordered_ids = list(originals.keys())
    reconstructed = registry.resolve_embeddings(ordered_ids)
    for idx, vector_id in enumerate(ordered_ids):
        np.testing.assert_allclose(
            reconstructed[idx], originals[vector_id], rtol=1e-5, atol=1e-6
        )

    cache: Dict[str, np.ndarray] = {}
    sample_id = ordered_ids[0]
    first = registry.resolve_embedding(sample_id, cache=cache)
    second = registry.resolve_embedding(sample_id, cache=cache)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)


@pytest.fixture
def large_registry_fixture() -> SimpleNamespace:
    dim = 8
    target_doc = "doc-target"
    target_namespace = "ns-heavy"
    target_specs = [
        SimpleNamespace(
            vector_id=f"{target_doc}-vec-{idx}",
            chunk_id=str(idx),
            base=float(idx + 1),
            position=idx,
            text=f"target chunk {idx}",
        )
        for idx in range(96)
    ]

    def make_chunk(
        doc_id: str,
        namespace: str,
        *,
        vector_id: str,
        chunk_id: str,
        base: float,
        position: int,
        text: str,
    ) -> ChunkPayload:
        embedding = np.full((dim,), (base % 11) + 1.0, dtype=np.float32)
        features = ChunkFeatures(
            bm25_terms={"token": base},
            splade_weights={"token": base * 2.0},
            embedding=embedding,
        )
        return ChunkPayload(
            doc_id=doc_id,
            chunk_id=chunk_id,
            vector_id=vector_id,
            namespace=namespace,
            text=text,
            metadata={},
            features=features,
            token_count=position,
            source_chunk_idxs=(position,),
            doc_items_refs=(f"{doc_id}:{chunk_id}",),
        )

    background: Dict[tuple[str, str], List[ChunkPayload]] = {}
    base = 1.0
    for doc_idx in range(28):
        doc_id = f"doc-{doc_idx}"
        namespace = f"ns-{doc_idx % 7}"
        entries: List[ChunkPayload] = []
        for chunk_idx in range(18):
            entries.append(
                make_chunk(
                    doc_id,
                    namespace,
                    vector_id=f"{doc_id}-vec-{chunk_idx}",
                    chunk_id=str(chunk_idx),
                    base=base + chunk_idx,
                    position=chunk_idx,
                    text=f"chunk {chunk_idx} of {doc_id}",
                )
            )
        background[(doc_id, namespace)] = entries
        base += 19.0

    def make_target_chunks() -> List[ChunkPayload]:
        return [
            make_chunk(
                target_doc,
                target_namespace,
                vector_id=spec.vector_id,
                chunk_id=spec.chunk_id,
                base=spec.base,
                position=spec.position,
                text=spec.text,
            )
            for spec in target_specs
        ]

    return SimpleNamespace(
        dim=dim,
        background=background,
        target_doc=target_doc,
        target_namespace=target_namespace,
        make_target_chunks=make_target_chunks,
    )


def test_delete_existing_for_doc_uses_registry_index(
    large_registry_fixture: SimpleNamespace, patcher: PatchManager
) -> None:
    class _RecordingLexicalIndex:
        def __init__(self) -> None:
            self.deleted_batches: List[Tuple[str, ...]] = []

        def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:  # pragma: no cover - stub
            self._last_upsert = [chunk.vector_id for chunk in chunks]

        def bulk_delete(self, vector_ids: Sequence[str]) -> None:
            self.deleted_batches.append(tuple(vector_ids))

    class _RecordingDenseStore:
        def __init__(self, dim: int) -> None:
            self._dim = dim
            self._store: Dict[str, np.ndarray] = {}
            self.removed_batches: List[Tuple[str, ...]] = []

        @property
        def dim(self) -> int:
            return self._dim

        def set_id_resolver(self, resolver) -> None:  # pragma: no cover - stub
            self._resolver = resolver

        def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
            for vector, vector_id in zip(vectors, vector_ids):
                self._store[str(vector_id)] = np.asarray(vector, dtype=np.float32)

        def remove(self, vector_ids: Sequence[str]) -> None:
            batch = tuple(vector_ids)
            self.removed_batches.append(batch)
            for vector_id in vector_ids:
                self._store.pop(str(vector_id), None)

        def reconstruct_batch(self, vector_ids: Sequence[str]) -> np.ndarray:  # pragma: no cover - stub
            return np.stack([self._store[str(vid)] for vid in vector_ids], dtype=np.float32)

    data = large_registry_fixture
    faiss_store = _RecordingDenseStore(data.dim)
    opensearch = _RecordingLexicalIndex()
    registry = ChunkRegistry()
    pipeline = ChunkIngestionPipeline(
        faiss_index=faiss_store,
        opensearch=opensearch,
        registry=registry,
        observability=Observability(),
    )

    background_total = 0
    for (doc_id, namespace), chunks in data.background.items():
        pipeline.faiss_index.add(
            [chunk.features.embedding for chunk in chunks],
            [chunk.vector_id for chunk in chunks],
        )
        registry.upsert(chunks)
        background_total += len(chunks)

    target_chunks = data.make_target_chunks()
    pipeline.faiss_index.add(
        [chunk.features.embedding for chunk in target_chunks],
        [chunk.vector_id for chunk in target_chunks],
    )
    registry.upsert(target_chunks)

    expected_vector_ids = {chunk.vector_id for chunk in target_chunks}

    original_vector_ids_for = registry.vector_ids_for
    call_count = 0

    def spy_vector_ids_for(doc_id: str, namespace: str):
        nonlocal call_count
        call_count += 1
        return original_vector_ids_for(doc_id, namespace)

    def fail_all():  # pragma: no cover - defensive guard
        raise AssertionError("ChunkRegistry.all() should not be used for targeted deletions")

    patcher.setattr(registry, "vector_ids_for", spy_vector_ids_for)
    patcher.setattr(registry, "all", fail_all)

    baseline_ids = tuple(original_vector_ids_for(data.target_doc, data.target_namespace))

    for _ in range(3):
        before = call_count
        staged = pipeline._delete_existing_for_doc(data.target_doc, data.target_namespace)
        assert call_count == before + 1
        assert set(staged) == expected_vector_ids
        assert staged == baseline_ids
        assert not opensearch.deleted_batches
        assert not faiss_store.removed_batches
        assert tuple(original_vector_ids_for(data.target_doc, data.target_namespace)) == baseline_ids

    assert registry.count() == background_total + len(target_chunks)
    assert not faiss_store.removed_batches
    assert not opensearch.deleted_batches
    sample_doc, sample_namespace = next(iter(data.background))
    assert tuple(original_vector_ids_for(sample_doc, sample_namespace))


# --- Regression coverage for deferred deletions ---


def test_upsert_documents_skips_deletes_on_commit_failure(
    patcher: PatchManager, tmp_path: Path
) -> None:
    class _StubLexicalIndex:
        def __init__(self) -> None:
            self.upserts: List[Tuple[str, ...]] = []
            self.deletes: List[Tuple[str, ...]] = []

        def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:
            self.upserts.append(tuple(chunk.vector_id for chunk in chunks))

        def bulk_delete(self, vector_ids: Sequence[str]) -> None:
            self.deletes.append(tuple(vector_ids))

    class _StubDenseStore:
        def __init__(self) -> None:
            self.added: List[Tuple[str, ...]] = []
            self.removed: List[Tuple[str, ...]] = []
            self._dim = 3
            self.config = SimpleNamespace(nlist=1, ivf_train_factor=1)

        @property
        def dim(self) -> int:
            return self._dim

        def set_id_resolver(self, resolver) -> None:  # pragma: no cover - stub
            self._resolver = resolver

        def needs_training(self) -> bool:
            return False

        def train(self, _: Sequence[np.ndarray]) -> None:  # pragma: no cover - stub
            raise AssertionError("train should not be invoked when needs_training() is False")

        def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
            self.added.append(tuple(vector_ids))

        def remove(self, vector_ids: Sequence[str]) -> None:
            self.removed.append(tuple(vector_ids))

    faiss_store = _StubDenseStore()
    opensearch = _StubLexicalIndex()
    registry = ChunkRegistry()
    pipeline = ChunkIngestionPipeline(
        faiss_index=faiss_store,
        opensearch=opensearch,
        registry=registry,
        observability=Observability(),
    )

    features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.zeros(3, dtype=np.float32),
    )
    existing_chunk = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-old",
        vector_id="vec-old",
        namespace="ns-1",
        text="existing",
        metadata={},
        features=features,
        token_count=0,
        source_chunk_idxs=(),
        doc_items_refs=(),
    )
    registry.upsert([existing_chunk])

    new_features = ChunkFeatures(
        bm25_terms={},
        splade_weights={},
        embedding=np.ones(3, dtype=np.float32),
    )
    new_chunk = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-new",
        vector_id="vec-new",
        namespace="ns-1",
        text="replacement",
        metadata={},
        features=new_features,
        token_count=0,
        source_chunk_idxs=(),
        doc_items_refs=(),
    )

    chunk_path = tmp_path / "chunks.jsonl"
    vector_path = tmp_path / "vectors.jsonl"
    chunk_path.write_text("{}\n", encoding="utf-8")
    vector_path.write_text("{}\n", encoding="utf-8")

    document = DocumentInput(
        doc_id="doc-1",
        namespace="ns-1",
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata={},
    )

    patcher.setattr(
        pipeline,
        "_load_precomputed_chunks",
        lambda doc: [new_chunk] if doc.doc_id == "doc-1" else [],
    )

    delete_calls: List[Tuple[str, ...]] = []

    def record_delete(vector_ids: Sequence[str]) -> None:
        delete_calls.append(tuple(vector_ids))

    patcher.setattr(pipeline, "delete_chunks", record_delete)

    def failing_commit(
        batch: Sequence[ChunkPayload], *, collect_vector_ids: bool = False
    ) -> None:
        raise RetryableIngestError("commit failure")

    patcher.setattr(pipeline, "_commit_batch", failing_commit)

    with pytest.raises(RetryableIngestError):
        pipeline.upsert_documents([document])

    assert delete_calls == []
    assert tuple(registry.vector_ids_for("doc-1", "ns-1")) == ("vec-old",)
    assert not faiss_store.removed
    assert not opensearch.deletes


# --- test_hybrid_search_real_vectors.py ---

DATASET_PATH = Path("Data/HybridScaleFixture/dataset.jsonl")


# --- test_hybrid_search_real_vectors.py ---


def _build_config(tmp_path: Path, *, oversample: int = 3) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": oversample},
        "fusion": {
            "k0": 50.0,
            "mmr_lambda": 0.6,
            "cosine_dedupe_threshold": 0.95,
            "max_chunks_per_doc": 3,
            "strict_highlights": False,
        },
        "retrieval": {"bm25_top_k": 40, "splade_top_k": 40, "dense_top_k": 40},
    }
    config_path = tmp_path / "real_hybrid_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(config_path)


# --- test_hybrid_search_real_vectors.py ---


@REAL_VECTOR_MARK
@pytest.fixture(scope="session")
def real_dataset() -> Sequence[Mapping[str, object]]:
    if not DATASET_PATH.exists():
        pytest.skip("Real vector dataset not generated")
    return load_dataset(DATASET_PATH)


# --- test_hybrid_search_real_vectors.py ---


def _to_documents(
    entries: Sequence[Mapping[str, object]], vector_format: str | None = None
) -> List[DocumentInput]:
    documents: List[DocumentInput] = []
    format_override = str(vector_format or "").lower()
    for entry in entries:
        document = entry["document"]
        entry_format = str(document.get("vector_format") or "").lower()
        vector_files = document.get("vector_files")
        effective_format = (
            format_override
            if format_override in {"jsonl", "parquet"}
            else entry_format if entry_format in {"jsonl", "parquet"} else ""
        )
        vector_value = document.get("vector_file")
        if isinstance(vector_files, Mapping):
            candidate = vector_files.get(effective_format) or (
                vector_files.get("jsonl") if effective_format == "" else None
            )
            if candidate is not None:
                vector_value = candidate
        vector_path = Path(str(vector_value))
        if not isinstance(vector_files, Mapping):
            if effective_format == "jsonl":
                vector_path = vector_path.with_suffix(".jsonl")
            elif effective_format == "parquet":
                vector_path = vector_path.with_suffix(".parquet")
        documents.append(
            DocumentInput(
                doc_id=str(document["doc_id"]),
                namespace=str(document["namespace"]),
                chunk_path=Path(str(document["chunk_file"])),
                vector_path=vector_path,
                metadata=dict(document.get("metadata", {})),
            )
        )
    return documents


# --- test_hybrid_search_real_vectors.py ---


@REAL_VECTOR_MARK
@pytest.fixture
def stack(tmp_path: Path, real_dataset: Sequence[Mapping[str, object]]) -> Callable[  # noqa: F811
    ...,
    tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FaissVectorStore,
        OpenSearchSimulator,
    ],
]:
    def factory(
        *,
        force_remove_ids_fallback: bool = False,
    ) -> tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FaissVectorStore,
        OpenSearchSimulator,
    ]:
        manager = _build_config(tmp_path)
        config = manager.get()
        embedding_dim = infer_embedding_dim(real_dataset)
        feature_generator = FeatureGenerator(embedding_dim=embedding_dim)
        dense_config = replace(config.dense, force_remove_ids_fallback=force_remove_ids_fallback)
        faiss_index = FaissVectorStore(dim=embedding_dim, config=dense_config)
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
            faiss_index=ManagedFaissAdapter(faiss_index),
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


# --- test_hybrid_search_real_vectors.py ---


@REAL_VECTOR_MARK
def test_real_fixture_ingest_and_search(
    stack: Callable[
        ...,
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FaissVectorStore,
            OpenSearchSimulator,
        ],
    ],
    real_dataset: Sequence[Mapping[str, object]],
) -> None:
    ingestion, service, registry, validator, _, _ = stack()
    documents = _to_documents(real_dataset)
    summary = ingestion.upsert_documents(documents)
    assert summary.chunk_count > 0, "Expected chunks to ingest from real vector fixture"
    assert registry.count() == summary.chunk_count

    for entry in real_dataset:
        for query in entry.get("queries", []):
            diagnostics_enabled = bool(query.get("diagnostics", True))
            request = HybridSearchRequest(
                query=str(query["query"]),
                namespace=query.get("namespace"),
                filters={},
                page_size=10,
                diagnostics=diagnostics_enabled,
            )
            response = service.search(request)
            assert response.results, f"Expected results for query {query['query']}"
            top_ids = [result.doc_id for result in response.results[:10]]
            assert query["expected_doc_id"] in top_ids
            if diagnostics_enabled:
                assert response.results[0].diagnostics is not None
            else:
                assert response.results[0].diagnostics is None


# --- test_hybrid_search_real_vectors.py ---


@REAL_VECTOR_MARK
def test_real_fixture_reingest_and_reports(
    stack: Callable[
        ...,
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FaissVectorStore,
            OpenSearchSimulator,
        ],
    ],
    real_dataset: Sequence[Mapping[str, object]],
    tmp_path: Path,
) -> None:
    ingestion, service, registry, validator, faiss_index, opensearch = stack(
        force_remove_ids_fallback=True
    )
    documents = _to_documents(real_dataset)
    ingestion.upsert_documents(documents)
    baseline_total = registry.count()

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


# --- test_hybrid_search_real_vectors.py ---


@REAL_VECTOR_MARK
def test_real_fixture_api_roundtrip(
    stack: Callable[
        ...,
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FaissVectorStore,
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

    status, body = api.post_hybrid_search(
        {
            "query": query["query"],
            "namespace": query["namespace"],
            "page_size": 3,
            "diagnostics": False,
        }
    )
    assert status == HTTPStatus.OK
    assert body["results"], "Expected API to return results when diagnostics disabled"
    assert "diagnostics" not in body["results"][0]


# --- test_hybrid_search_real_vectors.py ---


def test_remove_ids_cpu_fallback() -> None:
    manager = FaissVectorStore(dim=8, config=DenseIndexConfig(force_remove_ids_fallback=True))
    vector = np.ones(8, dtype=np.float32)
    manager.add([vector], ["00000000-0000-4000-8000-000000000001"])

    baseline_stats = manager.stats()
    manager.remove(["00000000-0000-4000-8000-000000000001"])

    stats = manager.stats()
    assert int(stats["gpu_remove_fallbacks"]) == int(baseline_stats["gpu_remove_fallbacks"]) + 1
    assert int(stats["total_rebuilds"]) == int(baseline_stats["total_rebuilds"]) + 1
    assert manager.ntotal == 0


def test_snapshot_refresh_throttled(patcher: PatchManager) -> None:
    config = DenseIndexConfig(
        snapshot_refresh_interval_seconds=3600.0,
        snapshot_refresh_writes=3,
    )
    manager = FaissVectorStore(dim=8, config=config)

    calls: list[int] = []

    def fake_serialize(self: FaissVectorStore) -> bytes:  # pragma: no cover - simple stub
        calls.append(1)
        return b"snapshot"

    patcher.setattr(FaissVectorStore, "serialize", fake_serialize)

    base = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
    for _ in range(3):
        noise = np.random.rand(8).astype(np.float32)
        manager.add([base + noise], [str(uuid.uuid4())])

    assert len(calls) == 1, "Expected throttle to coalesce the first two refresh attempts"

    noise = np.random.rand(8).astype(np.float32)
    manager.add([base + noise], [str(uuid.uuid4())])
    assert len(calls) == 1, "Expected writes below threshold to defer additional refreshes"

    manager.flush_snapshot()
    assert len(calls) == 2, "flush_snapshot should bypass the throttle policy"


def test_service_close_flushes_dense_snapshot(tmp_path: Path, patcher: PatchManager) -> None:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": 2},
        "fusion": {"k0": 10.0},
        "retrieval": {"bm25_top_k": 5, "splade_top_k": 5, "dense_top_k": 5},
    }
    config_path = tmp_path / "shutdown_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    manager = HybridSearchConfigManager(config_path)
    config = manager.get()
    config = replace(
        config,
        dense=replace(
            config.dense,
            snapshot_refresh_interval_seconds=3600.0,
            snapshot_refresh_writes=100,
        ),
    )
    manager._config = config
    feature_generator = FeatureGenerator(embedding_dim=16)
    faiss_index = FaissVectorStore(dim=feature_generator.embedding_dim, config=config.dense)
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
        faiss_index=ManagedFaissAdapter(faiss_index),
        opensearch=opensearch,
        registry=registry,
        observability=observability,
    )

    calls: list[int] = []

    real_serialize = FaissVectorStore.serialize

    def spy_serialize(self: FaissVectorStore) -> bytes:  # pragma: no cover - exercised in test
        calls.append(1)
        return real_serialize(self)

    patcher.setattr(FaissVectorStore, "serialize", spy_serialize)

    document = _write_document_artifacts(
        tmp_path,
        doc_id="shutdown-doc",
        namespace="ops",
        text="graceful shutdown triggers snapshot",
        metadata={},
        feature_generator=feature_generator,
    )
    ingestion.upsert_documents([document])
    baseline = len(calls)

    service.close()

    assert len(calls) >= baseline + 1, "service.close() should flush a final snapshot"


# --- test_hybrid_search_scale.py ---

DATASET_PATH = Path("Data/HybridScaleFixture/dataset.jsonl")


# --- test_hybrid_search_scale.py ---


@pytest.fixture(scope="session")
def scale_dataset() -> Sequence[Mapping[str, object]]:
    if not DATASET_PATH.exists():
        pytest.skip("Large-scale real vector fixture not generated")
    return load_dataset(DATASET_PATH)


# --- test_hybrid_search_scale.py ---


def _build_config(tmp_path: Path) -> HybridSearchConfigManager:
    config_payload = {
        "dense": {"index_type": "flat", "oversample": 3},
        "fusion": {
            "k0": 60.0,
            "mmr_lambda": 0.6,
            "cosine_dedupe_threshold": 0.95,
            "max_chunks_per_doc": 3,
            "strict_highlights": False,
        },
        "retrieval": {"bm25_top_k": 50, "splade_top_k": 50, "dense_top_k": 50},
    }
    config_path = tmp_path / "scale_hybrid_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return HybridSearchConfigManager(config_path)


# --- test_hybrid_search_scale.py ---


@pytest.fixture
def scale_stack(tmp_path: Path, scale_dataset: Sequence[Mapping[str, object]]) -> Callable[
    ...,
    tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FaissVectorStore,
        OpenSearchSimulator,
    ],
]:
    def factory(
        *,
        force_remove_ids_fallback: bool = False,
    ) -> tuple[
        ChunkIngestionPipeline,
        HybridSearchService,
        ChunkRegistry,
        HybridSearchValidator,
        FaissVectorStore,
        OpenSearchSimulator,
    ]:
        manager = _build_config(tmp_path)
        config = manager.get()
        embedding_dim = infer_embedding_dim(scale_dataset)
        feature_generator = FeatureGenerator(embedding_dim=embedding_dim)
        dense_config = replace(config.dense, force_remove_ids_fallback=force_remove_ids_fallback)
        faiss_index = FaissVectorStore(dim=embedding_dim, config=dense_config)
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
            faiss_index=ManagedFaissAdapter(faiss_index),
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


# --- test_hybrid_search_scale.py ---


@pytest.mark.real_vectors
@pytest.mark.scale_vectors
def test_hybrid_scale_suite(
    scale_stack: Callable[
        ...,
        tuple[
            ChunkIngestionPipeline,
            HybridSearchService,
            ChunkRegistry,
            HybridSearchValidator,
            FaissVectorStore,
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

    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    gc.collect()
    before_rss = process.memory_info().rss
    dense_report = validator._scale_dense_metrics(  # pylint: disable=protected-access
        service_module.DEFAULT_SCALE_THRESHOLDS,
        random.Random(9876),
    )
    gc.collect()
    after_rss = process.memory_info().rss
    delta_gib = max(0.0, (after_rss - before_rss) / (1024 ** 3))
    assert (
        delta_gib < 0.5
    ), f"scale_dense_metrics should not increase RSS by >=0.5 GiB (observed {delta_gib:.2f} GiB)"
    assert dense_report.details.get("sampled_chunks", 0) > 0


# --- test_hybridsearch_gpu_only.py ---

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("faiss is required for these tests", allow_module_level=True)


# --- test_hybridsearch_gpu_only.py ---


def _toy_data(n: int = 2048, d: int = 128) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((64, d), dtype=np.float32)
    return xb, xq


# --- test_hybridsearch_gpu_only.py ---


def _target_device() -> int:
    return int(os.getenv("HYBRIDSEARCH_FAISS_DEVICE", "0"))


# --- test_hybridsearch_gpu_only.py ---


def _make_id_resolver(vector_ids: list[str]) -> Callable[[int], str | None]:
    registry = ChunkRegistry()
    bridge = {registry.to_faiss_id(vid): vid for vid in vector_ids}
    return bridge.get


# --- test_hybridsearch_gpu_only.py ---


def _emit_vectors(xb: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    vectors = [row.copy() for row in xb]
    vector_ids = [str(uuid.uuid4()) for _ in vectors]
    return vectors, vector_ids


# --- test_hybridsearch_gpu_only.py ---


def _assert_gpu_index(manager: FaissVectorStore) -> None:
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


# --- test_hybridsearch_gpu_only.py ---


@GPU_MARK
def test_gpu_flat_end_to_end() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(
        index_type="flat",
        nprobe=1,
        device=_target_device(),
        persist_mode="disabled",
    )
    manager = FaissVectorStore(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


# --- test_hybridsearch_gpu_only.py ---


@GPU_MARK
def test_gpu_ivf_flat_build_and_search() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(
        index_type="ivf_flat",
        nlist=256,
        nprobe=8,
        device=_target_device(),
        persist_mode="disabled",
    )
    manager = FaissVectorStore(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


# --- test_hybridsearch_gpu_only.py ---


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
        persist_mode="disabled",
    )
    manager = FaissVectorStore(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


# --- test_hybridsearch_gpu_only.py ---


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


# --- test_hybridsearch_gpu_only.py ---


@GPU_MARK
def test_gpu_cosine_helpers_preserve_input_buffers() -> None:
    xb, _ = _toy_data(n=256)
    shared_query = xb[0]
    shared_corpus = xb[:32]
    query_before = shared_query.copy()
    corpus_before = shared_corpus.copy()
    xb_before = xb.copy()

    resources = faiss.StandardGpuResources()

    cosine_against_corpus_gpu(
        shared_query,
        shared_corpus,
        device=_target_device(),
        resources=resources,
    )

    assert np.allclose(shared_query, query_before)
    assert np.allclose(shared_corpus, corpus_before)
    assert np.allclose(xb, xb_before)

    pairwise_inner_products(
        shared_corpus,
        device=_target_device(),
        resources=resources,
    )

    assert np.allclose(shared_query, query_before)
    assert np.allclose(shared_corpus, corpus_before)
    assert np.allclose(xb, xb_before)


# --- test_hybridsearch_gpu_only.py ---


def test_gpu_clone_strict_coarse_quantizer() -> None:
    cfg = DenseIndexConfig(
        index_type="flat",
        device=_target_device(),
        persist_mode="disabled",
    )
    manager = FaissVectorStore(dim=32, config=cfg)
    cpu_index = faiss.IndexFlatIP(32)
    mapped = faiss.IndexIDMap2(cpu_index)
    gpu_index = manager._maybe_to_gpu(mapped)
    base = gpu_index.index if hasattr(gpu_index, "index") else gpu_index
    if hasattr(faiss, "downcast_index"):
        base = faiss.downcast_index(base)
    assert "Gpu" in type(base).__name__, "Expected GPU index after strict cloning"


# --- test_hybridsearch_gpu_only.py ---


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


def test_result_shaper_doc_cap_ignores_filtered_duplicates() -> None:
    embedding = np.ones(8, dtype=np.float32)
    duplicate = embedding.copy()
    distinct = np.concatenate((np.ones(4), np.zeros(4))).astype(np.float32)

    chunk_primary = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        vector_id="vec-1",
        namespace="default",
        text="Primary chunk for the document",
        metadata={},
        features=ChunkFeatures({}, {}, embedding),
        token_count=4,
        source_chunk_idxs=[0],
        doc_items_refs=[],
    )
    chunk_duplicate = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-2",
        vector_id="vec-2",
        namespace="default",
        text="Primary chunk for the document",
        metadata={},
        features=ChunkFeatures({}, {}, duplicate),
        token_count=4,
        source_chunk_idxs=[1],
        doc_items_refs=[],
    )
    chunk_distinct = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-3",
        vector_id="vec-3",
        namespace="default",
        text="A distinct chunk that should still be emitted",
        metadata={},
        features=ChunkFeatures({}, {}, distinct),
        token_count=4,
        source_chunk_idxs=[2],
        doc_items_refs=[],
    )

    opensearch = OpenSearchSimulator()
    opensearch.bulk_upsert([chunk_primary, chunk_duplicate, chunk_distinct])

    shaper = ResultShaper(
        opensearch,
        FusionConfig(max_chunks_per_doc=2, cosine_dedupe_threshold=0.9),
    )
    request = HybridSearchRequest(
        query="primary chunk",
        namespace=None,
        filters={},
        page_size=5,
    )

    ordered = [chunk_primary, chunk_duplicate, chunk_distinct]
    fused_scores = {
        chunk_primary.vector_id: 1.0,
        chunk_duplicate.vector_id: 0.95,
        chunk_distinct.vector_id: 0.9,
    }
    channel_scores = {"dense": fused_scores}

    results = shaper.shape(ordered, fused_scores, request, channel_scores)

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-3"]
    assert all(result.doc_id == "doc-1" for result in results)


# --- test_hybridsearch_gpu_only.py ---


def test_result_shaper_enforces_global_budgets() -> None:
    embedding = np.ones(8, dtype=np.float32)
    chunk_a = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-a",
        vector_id="vec-a",
        namespace="default",
        text="Alpha beta gamma",
        metadata={},
        features=ChunkFeatures({}, {}, embedding.copy()),
        token_count=3,
        source_chunk_idxs=[0],
        doc_items_refs=[],
    )
    chunk_b = ChunkPayload(
        doc_id="doc-2",
        chunk_id="chunk-b",
        vector_id="vec-b",
        namespace="default",
        text="Delta epsilon zeta",
        metadata={},
        features=ChunkFeatures(
            {}, {}, np.concatenate((np.ones(4), np.zeros(4))).astype(np.float32)
        ),
        token_count=3,
        source_chunk_idxs=[1],
        doc_items_refs=[],
    )
    chunk_c = ChunkPayload(
        doc_id="doc-3",
        chunk_id="chunk-c",
        vector_id="vec-c",
        namespace="default",
        text="Eta theta iota",
        metadata={},
        features=ChunkFeatures(
            {}, {}, np.concatenate((np.zeros(4), np.ones(4))).astype(np.float32)
        ),
        token_count=3,
        source_chunk_idxs=[2],
        doc_items_refs=[],
    )

    opensearch = OpenSearchSimulator()
    opensearch.bulk_upsert([chunk_a, chunk_b, chunk_c])

    shaper = ResultShaper(
        opensearch,
        FusionConfig(
            max_chunks_per_doc=2,
            token_budget=6,
            byte_budget=60,
        ),
    )
    request = HybridSearchRequest(query="alpha epsilon", namespace=None, filters={}, page_size=5)
    ordered = [chunk_a, chunk_b, chunk_c]
    fused_scores = {chunk.vector_id: score for chunk, score in zip(ordered, (1.0, 0.9, 0.8))}
    channel_scores: Mapping[str, Mapping[str, float]] = {
        "dense": fused_scores,
        "bm25": {},
        "splade": {},
    }

    results = shaper.shape(ordered, fused_scores, request, channel_scores)
    assert len(results) == 2, "Budgets should stop shaping before exhausting candidates"
    assert {result.vector_id for result in results} == {"vec-a", "vec-b"}
    assert all(result.highlights for result in results), "Highlights should come from lexical store"


# --- test_hybridsearch_gpu_only.py ---


def test_gpu_nprobe_applied_during_search() -> None:
    xb, xq = _toy_data(n=512, d=64)
    cfg = DenseIndexConfig(
        index_type="ivf_flat",
        nlist=64,
        nprobe=32,
        device=_target_device(),
        persist_mode="disabled",
    )
    manager = FaissVectorStore(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    manager.search(xq[0], top_k=5)
    base = manager._index.index if hasattr(manager._index, "index") else manager._index
    if hasattr(faiss, "downcast_index"):
        base = faiss.downcast_index(base)
    assert hasattr(base, "nprobe")
    assert int(base.nprobe) == cfg.nprobe


# --- test_hybridsearch_gpu_only.py ---


def test_gpu_similarity_uses_supplied_device() -> None:
    cfg = DenseIndexConfig(
        index_type="flat",
        device=_target_device(),
        persist_mode="disabled",
    )
    manager = FaissVectorStore(dim=32, config=cfg)

    captured: dict[str, object] = {}

    def fake_pairwise(resources, A, B, metric, device):  # type: ignore[no-untyped-def]
        captured["resources"] = resources
        captured["device"] = device
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)

    q = np.ones(32, dtype=np.float32)
    corpus = np.ones((3, 32), dtype=np.float32)
    cosine_against_corpus_gpu(
        q,
        corpus,
        device=manager.device,
        resources=manager.gpu_resources,
        pairwise_fn=fake_pairwise,
    )

    assert captured.get("device") == manager.device
    assert captured.get("resources") is manager.gpu_resources


def test_operations_module_is_removed() -> None:
    module_name = "DocsToKG.HybridSearch.operations"
    sys.modules.pop(module_name, None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_results_module_is_removed() -> None:
    module_name = "DocsToKG.HybridSearch.results"
    sys.modules.pop(module_name, None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_similarity_module_is_removed() -> None:
    module_name = "DocsToKG.HybridSearch.similarity"
    sys.modules.pop(module_name, None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_retrieval_module_is_removed() -> None:
    module_name = "DocsToKG.HybridSearch.retrieval"
    sys.modules.pop(module_name, None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_schema_module_is_removed() -> None:
    module_name = "DocsToKG.HybridSearch.schema"
    sys.modules.pop(module_name, None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
