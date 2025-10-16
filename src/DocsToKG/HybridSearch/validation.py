# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.validation",
#   "purpose": "Validation CLI and dataset utilities for hybrid search",
#   "sections": [
#     {
#       "id": "hybridsearchvalidator",
#       "name": "HybridSearchValidator",
#       "anchor": "class-hybridsearchvalidator",
#       "kind": "class"
#     },
#     {
#       "id": "load-dataset",
#       "name": "load_dataset",
#       "anchor": "function-load-dataset",
#       "kind": "function"
#     },
#     {
#       "id": "infer-embedding-dim",
#       "name": "infer_embedding_dim",
#       "anchor": "function-infer-embedding-dim",
#       "kind": "function"
#     },
#     {
#       "id": "run-pytest-suites",
#       "name": "run_pytest_suites",
#       "anchor": "function-run-pytest-suites",
#       "kind": "function"
#     },
#     {
#       "id": "run-real-vector-ci",
#       "name": "run_real_vector_ci",
#       "anchor": "function-run-real-vector-ci",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Automated validation harness for the hybrid search stack."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

import faiss  # type: ignore

from .config import HybridSearchConfigManager
from .features import FeatureGenerator
from .ingest import ChunkIngestionPipeline
from .observability import Observability
from .service import HybridSearchService
from .storage import ChunkRegistry, OpenSearchSimulator
from .types import (
    ChunkPayload,
    DocumentInput,
    HybridSearchRequest,
    HybridSearchResult,
    ValidationReport,
    ValidationSummary,
)
from .vectorstore import FaissVectorStore, pairwise_inner_products
from .vectorstore import restore_state as vectorstore_restore_state
from .vectorstore import serialize_state as vectorstore_serialize_state

# --- Globals ---

__all__ = (
    "BASIC_DENSE_SELF_HIT_THRESHOLD",
    "BASIC_SPARSE_RELEVANCE_THRESHOLD",
    "DEFAULT_SCALE_THRESHOLDS",
    "HybridSearchValidator",
    "infer_embedding_dim",
    "load_dataset",
    "main",
    "run_pytest_suites",
    "run_real_vector_ci",
)

DEFAULT_SCALE_THRESHOLDS: Dict[str, float] = {
    "dense_self_hit": 0.99,
    "dense_recall_at_10": 0.95,
    "dense_perturb_top3": 0.95,
    "bm25_hit_rate@10": 0.8,
    "splade_hit_rate@10": 0.8,
    "dense_hit_rate@10": 0.8,
    "rrf_hit_rate@10": 0.8,
    "mmr_redundancy_reduction": 0.1,
    "mmr_hit_rate_delta": 0.2,
    "latency_p95_ms": 300.0,
    "gpu_headroom_fraction": 0.2,
}

BASIC_DENSE_SELF_HIT_THRESHOLD = 0.99
BASIC_SPARSE_RELEVANCE_THRESHOLD = 0.90


# --- Public Classes ---

class HybridSearchValidator:
    """Execute validation sweeps and persist reports.

    Attributes:
        _ingestion: Chunk ingestion pipeline used for preparing test data.
        _service: Hybrid search service under validation.
        _registry: Registry exposing ingested chunk metadata.
        _opensearch: OpenSearch simulator for lexical storage inspection.

    Examples:
        >>> validator = HybridSearchValidator(
        ...     ingestion=ChunkIngestionPipeline(...),  # doctest: +SKIP
        ...     service=HybridSearchService(...),      # doctest: +SKIP
        ...     registry=ChunkRegistry(),
        ...     opensearch=OpenSearchSimulator(),
        ... )
        >>> isinstance(validator, HybridSearchValidator)
        True
    """

    def __init__(
        self,
        *,
        ingestion: ChunkIngestionPipeline,
        service: HybridSearchService,
        registry: ChunkRegistry,
        opensearch: OpenSearchSimulator,
    ) -> None:
        """Bind ingestion, retrieval, and storage components for validation sweeps.

        Args:
            ingestion: Ingestion pipeline used to materialise chunk data.
            service: Hybrid search service under test.
            registry: Chunk registry exposing ingested metadata.
            opensearch: OpenSearch simulator for lexical validation.

        Returns:
            None
        """

        self._ingestion = ingestion
        self._service = service
        self._registry = registry
        self._opensearch = opensearch
        self._validation_resources: Optional["faiss.StandardGpuResources"] = None

    def run(
        self, dataset: Sequence[Mapping[str, object]], output_root: Optional[Path] = None
    ) -> ValidationSummary:
        """Execute standard validation against a dataset.

        Args:
            dataset: Loaded dataset entries containing document and query payloads.
            output_root: Optional directory where reports should be written.

        Returns:
            ValidationSummary capturing per-check reports.
        """
        started = datetime.now(UTC)
        documents = [self._to_document(entry["document"]) for entry in dataset]
        self._ingestion.upsert_documents(documents)
        reports: List[ValidationReport] = []
        reports.append(self._check_ingest_integrity())
        reports.append(self._check_dense_self_hit())
        reports.append(self._check_sparse_relevance(dataset))
        reports.append(self._check_namespace_filters(dataset))
        reports.append(self._check_pagination(dataset))
        reports.append(self._check_highlights(dataset))
        reports.append(self._check_backup_restore(dataset))
        calibration = self._run_calibration(dataset)
        reports.append(calibration)
        completed = datetime.now(UTC)
        summary = ValidationSummary(reports=reports, started_at=started, completed_at=completed)
        self._persist_reports(
            summary, output_root, calibration.details if calibration.details else None
        )
        return summary

    def run_scale(
        self,
        dataset: Sequence[Mapping[str, object]],
        *,
        output_root: Optional[Path] = None,
        thresholds: Optional[Mapping[str, float]] = None,
        query_sample_size: int = 120,
    ) -> ValidationSummary:
        """Execute a comprehensive scale validation suite.

        Args:
            dataset: Dataset entries containing documents and queries.
            output_root: Optional directory for detailed metrics output.
            thresholds: Optional overrides for scale validation thresholds.
            query_sample_size: Number of queries sampled for specific checks.

        Returns:
            ValidationSummary with detailed per-check reports.
        """
        merged_thresholds: Dict[str, float] = dict(DEFAULT_SCALE_THRESHOLDS)
        if thresholds:
            merged_thresholds.update(thresholds)

        started = datetime.now(UTC)
        documents = [self._to_document(entry["document"]) for entry in dataset]
        inputs_by_doc = {doc.doc_id: doc for doc in documents}
        self._ingestion.upsert_documents(documents)
        rng = random.Random(1337)

        reports: List[ValidationReport] = []
        extras: Dict[str, Mapping[str, object]] = {}

        data_report = self._scale_data_sanity(documents, dataset)
        reports.append(data_report)
        extras[data_report.name] = data_report.details

        crud_report = self._scale_crud_namespace(documents, dataset, inputs_by_doc, rng)
        reports.append(crud_report)
        extras[crud_report.name] = crud_report.details

        dense_report = self._scale_dense_metrics(merged_thresholds, rng)
        reports.append(dense_report)
        extras[dense_report.name] = dense_report.details

        relevance_report = self._scale_channel_relevance(
            dataset, merged_thresholds, rng, query_sample_size
        )
        reports.append(relevance_report)
        extras[relevance_report.name] = relevance_report.details

        fusion_report = self._scale_fusion_mmr(dataset, merged_thresholds, rng, query_sample_size)
        reports.append(fusion_report)
        extras[fusion_report.name] = fusion_report.details

        pagination_report = self._scale_pagination(dataset, rng)
        reports.append(pagination_report)
        extras[pagination_report.name] = pagination_report.details

        shaping_report = self._scale_result_shaping(dataset, rng)
        reports.append(shaping_report)
        extras[shaping_report.name] = shaping_report.details

        backup_report = self._scale_backup_restore(dataset, rng, query_sample_size)
        reports.append(backup_report)
        extras[backup_report.name] = backup_report.details

        acl_report = self._scale_acl(dataset)
        reports.append(acl_report)
        extras[acl_report.name] = acl_report.details

        performance_report = self._scale_performance(
            dataset, merged_thresholds, rng, query_sample_size
        )
        reports.append(performance_report)
        extras[performance_report.name] = performance_report.details

        stability_report = self._scale_stability(dataset, inputs_by_doc, rng, query_sample_size)
        reports.append(stability_report)
        extras[stability_report.name] = stability_report.details

        calibration = self._run_calibration(dataset)
        reports.append(calibration)
        extras[calibration.name] = calibration.details if calibration.details else {}

        completed = datetime.now(UTC)
        summary = ValidationSummary(reports=reports, started_at=started, completed_at=completed)
        self._persist_reports(
            summary,
            output_root,
            calibration.details if calibration.details else None,
            extras=extras,
        )
        return summary

    def _to_document(self, payload: Mapping[str, object]) -> DocumentInput:
        """Transform dataset document payload into a `DocumentInput`.

        Args:
            payload: Dataset document dictionary containing paths and metadata.

        Returns:
            Corresponding `DocumentInput` instance for ingestion.
        """
        return DocumentInput(
            doc_id=str(payload["doc_id"]),
            namespace=str(payload["namespace"]),
            chunk_path=Path(str(payload["chunk_file"])).resolve(),
            vector_path=Path(str(payload["vector_file"])).resolve(),
            metadata=dict(payload.get("metadata", {})),
        )

    def _check_ingest_integrity(self) -> ValidationReport:
        """Verify that ingested chunks contain valid embeddings.

        Args:
            None

        Returns:
            ValidationReport describing integrity checks for ingested chunks.
        """
        all_chunks = self._registry.all()
        ok = all(len(chunk.features.embedding.shape) == 1 for chunk in all_chunks)
        details = {"total_chunks": len(all_chunks)}
        return ValidationReport(name="ingest_integrity", passed=ok, details=details)

    def _check_dense_self_hit(self) -> ValidationReport:
        """Ensure dense search returns each chunk as its own nearest neighbor.

        Args:
            None

        Returns:
            ValidationReport summarizing dense self-hit accuracy.
        """
        total = 0
        hits_met = 0
        for chunk in self._registry.all():
            total += 1
            hits = self._ingestion.faiss_index.search(chunk.features.embedding, 1)
            if hits and hits[0].vector_id == chunk.vector_id:
                hits_met += 1
        rate = hits_met / total if total else 0.0
        passed = rate >= BASIC_DENSE_SELF_HIT_THRESHOLD
        return ValidationReport(
            name="dense_self_hit",
            passed=passed,
            details={
                "total_chunks": total,
                "correct_hits": hits_met,
                "self_hit_rate": rate,
                "threshold": BASIC_DENSE_SELF_HIT_THRESHOLD,
            },
        )

    def _check_sparse_relevance(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Check that sparse channels retrieve expected documents.

        Args:
            dataset: Dataset entries with expected relevance metadata.

        Returns:
            ValidationReport covering sparse channel relevance.
        """
        total = 0
        hits_met = 0
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                total += 1
                expected = query.get("expected_doc_id")
                request = self._request_for_query(query)
                response = self._service.search(request)
                if response.results and (not expected or response.results[0].doc_id == expected):
                    hits_met += 1
        rate = hits_met / total if total else 0.0
        passed = rate >= BASIC_SPARSE_RELEVANCE_THRESHOLD
        return ValidationReport(
            name="sparse_relevance",
            passed=passed,
            details={
                "total_queries": total,
                "top1_matches": hits_met,
                "hit_rate": rate,
                "threshold": BASIC_SPARSE_RELEVANCE_THRESHOLD,
            },
        )

    def _check_namespace_filters(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Ensure namespace-constrained queries do not leak results.

        Args:
            dataset: Dataset entries containing namespace-constrained queries.

        Returns:
            ValidationReport indicating whether namespace isolation holds.
        """
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                namespace = query.get("namespace")
                if not namespace:
                    continue
                request = self._request_for_query(query)
                response = self._service.search(request)
                for result in response.results:
                    if result.namespace != namespace:
                        ok = False
                        break
            if not ok:
                break
        return ValidationReport(name="namespace_filter", passed=ok, details={})

    def _check_pagination(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Verify pagination cursors do not create duplicate results.

        Args:
            dataset: Dataset entries providing queries for pagination tests.

        Returns:
            ValidationReport detailing pagination stability.
        """
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                request = self._request_for_query(query, page_size=2)
                first = self._service.search(request)
                if not first.next_cursor:
                    continue
                request.cursor = first.next_cursor
                second = self._service.search(request)
                seen = {(result.doc_id, result.chunk_id) for result in first.results}
                overlap = any((result.doc_id, result.chunk_id) in seen for result in second.results)
                if overlap:
                    ok = False
                    break
            if not ok:
                break
        return ValidationReport(name="pagination_stability", passed=ok, details={})

    def _check_highlights(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Assert that each result includes highlight snippets.

        Args:
            dataset: Dataset entries with queries to validate highlight generation.

        Returns:
            ValidationReport indicating highlight completeness.
        """
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                request = self._request_for_query(query)
                response = self._service.search(request)
                if any(not result.highlights for result in response.results):
                    ok = False
                    break
            if not ok:
                break
        return ValidationReport(name="highlight_presence", passed=ok, details={})

    def _check_backup_restore(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Validate backup/restore by round-tripping the FAISS index.

        Args:
            dataset: Dataset entries used to verify consistency post-restore.

        Returns:
            ValidationReport capturing backup/restore status.
        """
        if not dataset:
            return ValidationReport(
                name="backup_restore", passed=False, details={"error": "dataset empty"}
            )
        queries = dataset[0].get("queries", [])
        if not queries:
            return ValidationReport(
                name="backup_restore", passed=False, details={"error": "missing queries"}
            )
        faiss_bytes = self._ingestion.faiss_index.serialize()
        first_query = self._request_for_query(queries[0])
        before = self._service.search(first_query)
        self._ingestion.faiss_index.restore(faiss_bytes)
        after = self._service.search(first_query)
        ok = bool(
            before.results and after.results and before.results[0].doc_id == after.results[0].doc_id
        )
        return ValidationReport(name="backup_restore", passed=ok, details={})

    def _request_for_query(
        self, query: Mapping[str, object], page_size: int = 5
    ) -> HybridSearchRequest:
        """Construct a `HybridSearchRequest` for a dataset query payload.

        Args:
            query: Query payload from the dataset.
            page_size: Desired page size for generated requests.

        Returns:
            HybridSearchRequest ready for execution against the service.
        """
        return HybridSearchRequest(
            query=str(query["query"]),
            namespace=query.get("namespace"),
            filters=dict(query.get("filters", {})),
            page_size=page_size,
            cursor=None,
            diversification=bool(query.get("diversification", False)),
            diagnostics=True,
        )

    def _persist_reports(
        self,
        summary: ValidationSummary,
        output_root: Optional[Path],
        calibration_details: Optional[Mapping[str, object]],
        *,
        extras: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        """Persist validation summaries and detailed metrics to disk.

        Args:
            summary: Validation summary containing reports.
            output_root: Optional directory for storing artifacts.
            calibration_details: Optional calibration metrics to persist.
            extras: Optional mapping of additional metrics categories.

        Returns:
            None
        """
        root = output_root or Path("reports/validation")
        directory = root / summary.started_at.strftime("%Y%m%d%H%M%S")
        directory.mkdir(parents=True, exist_ok=True)
        reports_json = [
            {
                "name": report.name,
                "passed": report.passed,
                "details": report.details,
            }
            for report in summary.reports
        ]
        (directory / "summary.json").write_text(
            json.dumps(reports_json, indent=2), encoding="utf-8"
        )
        human_lines = [
            f"Validation started at: {summary.started_at.isoformat()}",
            f"Validation completed at: {summary.completed_at.isoformat()}",
            f"Overall status: {'PASS' if summary.passed else 'FAIL'}",
        ]
        for report in summary.reports:
            human_lines.append(f"- {report.name}: {'PASS' if report.passed else 'FAIL'}")
        (directory / "summary.txt").write_text("\n".join(human_lines), encoding="utf-8")
        if calibration_details is not None:
            (directory / "calibration.json").write_text(
                json.dumps(calibration_details, indent=2), encoding="utf-8"
            )
        if extras:
            (directory / "metrics.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")

    def _collect_queries(
        self, dataset: Sequence[Mapping[str, object]]
    ) -> List[tuple[Mapping[str, object], Mapping[str, object]]]:
        """Collect (document, query) pairs from the dataset.

        Args:
            dataset: Loaded dataset entries containing documents and queries.

        Returns:
            List of tuples pairing document payloads with query payloads.
        """
        pairs: List[tuple[Mapping[str, object], Mapping[str, object]]] = []
        for entry in dataset:
            document_payload = entry.get("document", {})
            for query in entry.get("queries", []):
                pairs.append((document_payload, query))
        return pairs

    def _sample_queries(
        self,
        dataset: Sequence[Mapping[str, object]],
        sample_size: int,
        rng: random.Random,
    ) -> List[tuple[Mapping[str, object], Mapping[str, object]]]:
        """Sample a subset of (document, query) pairs for randomized checks.

        Args:
            dataset: Dataset entries containing documents and queries.
            sample_size: Maximum number of pairs to return.
            rng: Random generator used for sampling.

        Returns:
            List of sampled (document, query) pairs.
        """
        pairs = self._collect_queries(dataset)
        if not pairs:
            return []
        if sample_size >= len(pairs):
            return pairs
        return rng.sample(pairs, sample_size)

    def _scale_data_sanity(
        self,
        documents: Sequence[DocumentInput],
        dataset: Sequence[Mapping[str, object]],
    ) -> ValidationReport:
        """Validate that ingested corpus statistics look healthy.

        Args:
            documents: Document inputs ingested during the scale run.
            dataset: Dataset entries used for validation.

        Returns:
            ValidationReport summarizing data sanity findings.
        """
        total_chunks = self._registry.count()
        namespaces = sorted({doc.namespace for doc in documents})
        dims: set[int] = set()
        invalid_vectors = 0
        for chunk in self._registry.all():
            vector = chunk.features.embedding
            dims.add(vector.shape[0])
            if not np.isfinite(vector).all():
                invalid_vectors += 1
        acl_missing = sum(1 for doc in documents if not doc.metadata.get("acl"))
        query_pairs = self._collect_queries(dataset)
        passed = len(dims) == 1 and invalid_vectors == 0 and acl_missing == 0
        details: Dict[str, object] = {
            "total_documents": len(documents),
            "total_queries": len(query_pairs),
            "total_chunks": total_chunks,
            "namespaces": namespaces,
            "vector_dimensions": sorted(dims),
            "invalid_vector_count": invalid_vectors,
            "documents_missing_acl": acl_missing,
        }
        return ValidationReport(name="scale_data_sanity", passed=passed, details=details)

    def _scale_crud_namespace(
        self,
        documents: Sequence[DocumentInput],
        dataset: Sequence[Mapping[str, object]],
        inputs_by_doc: Mapping[str, DocumentInput],
        rng: random.Random,
    ) -> ValidationReport:
        """Exercise CRUD operations to ensure namespace isolation stays intact.

        Args:
            documents: Documents ingested for the validation run.
            dataset: Dataset entries providing queries for verification.
            inputs_by_doc: Mapping back to original document inputs.
            rng: Random generator used to select documents.

        Returns:
            ValidationReport describing CRUD and namespace violations if any.
        """
        initial_registry = self._registry.count()
        initial_faiss = self._ingestion.faiss_index.ntotal

        doc_ids = [doc.doc_id for doc in documents]
        update_count = min(max(10, len(doc_ids) // 10), len(doc_ids)) or 1
        update_doc_ids = rng.sample(doc_ids, update_count)
        self._ingestion.upsert_documents([inputs_by_doc[doc_id] for doc_id in update_doc_ids])

        update_ok = (
            self._registry.count() == initial_registry
            and self._ingestion.faiss_index.ntotal == initial_faiss
        )

        vector_ids = [chunk.vector_id for chunk in self._registry.all()]
        delete_count = min(max(10, len(vector_ids) // 20), len(vector_ids) // 2 or 1)
        delete_ids = rng.sample(vector_ids, delete_count)
        deleted_doc_ids = set()
        for vector_id in delete_ids:
            chunk = self._registry.get(vector_id)
            if chunk is not None:
                deleted_doc_ids.add(chunk.doc_id)
        self._ingestion.delete_chunks(delete_ids)

        delete_ok = (
            self._registry.count() == initial_registry - delete_count
            and self._ingestion.faiss_index.ntotal == initial_faiss - delete_count
        )

        if deleted_doc_ids:
            self._ingestion.upsert_documents([inputs_by_doc[doc_id] for doc_id in deleted_doc_ids])

        restore_ok = (
            self._registry.count() == initial_registry
            and self._ingestion.faiss_index.ntotal == initial_faiss
        )

        namespace_pairs: Dict[str, List[Mapping[str, object]]] = {}
        for document_payload, query_payload in self._collect_queries(dataset):
            namespace = query_payload.get("namespace") or document_payload.get("namespace")
            if namespace:
                namespace_pairs.setdefault(str(namespace), []).append(query_payload)

        namespace_violations: List[str] = []
        for namespace, queries in namespace_pairs.items():
            sample = queries[: min(5, len(queries))]
            for query_payload in sample:
                request = self._request_for_query(query_payload)
                response = self._service.search(request)
                for result in response.results:
                    if result.namespace != namespace:
                        namespace_violations.append(namespace)
                        break
                if namespace_violations:
                    break

        details: Dict[str, object] = {
            "updates_tested": update_count,
            "deletes_tested": delete_count,
            "namespaces_checked": sorted(namespace_pairs.keys()),
            "namespace_violations": namespace_violations,
            "registry_count": self._registry.count(),
            "faiss_ntotal": self._ingestion.faiss_index.ntotal,
        }

        passed = update_ok and delete_ok and restore_ok and not namespace_violations
        return ValidationReport(name="scale_crud_namespace", passed=passed, details=details)

    def _scale_dense_metrics(
        self,
        thresholds: Mapping[str, float],
        rng: random.Random,
    ) -> ValidationReport:
        """Assess dense retrieval self-hit, perturbation, and recall metrics.

        Args:
            thresholds: Threshold values for dense metric success.
            rng: Random generator for selecting sample chunks.

        Returns:
            ValidationReport containing dense metric measurements.
        """
        all_chunks = self._registry.all()
        if not all_chunks:
            return ValidationReport(
                name="scale_dense_metrics",
                passed=False,
                details={"error": "registry empty"},
            )

        sample_size = min(max(200, len(all_chunks) // 4), len(all_chunks))
        sampled_chunks = rng.sample(all_chunks, sample_size)

        top_k = min(10, len(all_chunks))
        self_hits = 0
        perturb_hits = 0
        recalls: List[float] = []

        # Precompute matrix for brute-force recall estimates.
        vector_matrix = np.stack(
            [chunk.features.embedding for chunk in all_chunks], dtype=np.float32
        )
        vector_ids = [chunk.vector_id for chunk in all_chunks]

        noise_rng = np.random.default_rng(2024)

        for chunk in sampled_chunks:
            query_vec = chunk.features.embedding
            hits = self._ingestion.faiss_index.search(query_vec, top_k)
            retrieved_ids = [hit.vector_id for hit in hits]
            if retrieved_ids and retrieved_ids[0] == chunk.vector_id:
                self_hits += 1

            noise = noise_rng.normal(scale=0.01, size=query_vec.shape).astype(np.float32)
            perturbed = query_vec + noise
            perturbed_hits = self._ingestion.faiss_index.search(perturbed, top_k)
            if any(
                hit.vector_id == chunk.vector_id
                for hit in perturbed_hits[: min(3, len(perturbed_hits))]
            ):
                perturb_hits += 1

            scores = vector_matrix @ query_vec
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            ground_truth_ids = [vector_ids[idx] for idx in top_indices]
            overlap = len(set(retrieved_ids) & set(ground_truth_ids))
            recalls.append(overlap / min(top_k, len(ground_truth_ids)) if ground_truth_ids else 0.0)

        self_hit_rate = self_hits / sample_size if sample_size else 0.0
        perturb_rate = perturb_hits / sample_size if sample_size else 0.0
        avg_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0

        details = {
            "sampled_chunks": sample_size,
            "self_hit_rate": self_hit_rate,
            "perturb_top3_rate": perturb_rate,
            "recall_at_10": avg_recall,
        }

        passed = (
            self_hit_rate >= thresholds.get("dense_self_hit", 0.0)
            and perturb_rate >= thresholds.get("dense_perturb_top3", 0.0)
            and avg_recall >= thresholds.get("dense_recall_at_10", 0.0)
        )
        return ValidationReport(name="scale_dense_metrics", passed=passed, details=details)

    def _scale_channel_relevance(
        self,
        dataset: Sequence[Mapping[str, object]],
        thresholds: Mapping[str, float],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Measure top-10 relevance for each retrieval channel across sampled queries.

        Args:
            dataset: Dataset entries containing queries for evaluation.
            thresholds: Threshold mapping for hit-rate expectations.
            rng: Random generator used for sampling query pairs.
            query_sample_size: Number of query pairs sampled.

        Returns:
            ValidationReport capturing per-channel relevance metrics.
        """
        sampled_pairs = self._sample_queries(dataset, query_sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_channel_relevance",
                passed=False,
                details={"error": "no queries available"},
            )

        feature_generator = self._service._feature_generator
        bm25_hits = 0
        splade_hits = 0
        dense_hits = 0
        rrf_hits = 0

        bm25_ranks: List[int] = []
        splade_ranks: List[int] = []
        dense_ranks: List[int] = []
        rrf_ranks: List[int] = []

        doc_to_embedding: Dict[str, np.ndarray] = {}
        for chunk in self._registry.all():
            doc_to_embedding.setdefault(chunk.doc_id, chunk.features.embedding)

        for document_payload, query_payload in sampled_pairs:
            expected_doc_id = str(
                query_payload.get("expected_doc_id") or document_payload.get("doc_id")
            )
            request = self._request_for_query(query_payload, page_size=10)
            filters = dict(request.filters)
            if request.namespace:
                filters["namespace"] = request.namespace

            features = feature_generator.compute_features(request.query)
            dense_query_vector = doc_to_embedding.get(expected_doc_id, features.embedding)

            bm25_results, _ = self._opensearch.search_bm25(features.bm25_terms, filters, top_k=10)
            bm25_doc_ids = [chunk.doc_id for chunk, _ in bm25_results]
            if expected_doc_id in bm25_doc_ids:
                bm25_hits += 1
                bm25_ranks.append(bm25_doc_ids.index(expected_doc_id) + 1)

            splade_results, _ = self._opensearch.search_splade(
                features.splade_weights, filters, top_k=10
            )
            splade_doc_ids = [chunk.doc_id for chunk, _ in splade_results]
            if expected_doc_id in splade_doc_ids:
                splade_hits += 1
                splade_ranks.append(splade_doc_ids.index(expected_doc_id) + 1)

            dense_results = self._ingestion.faiss_index.search(dense_query_vector, 10)
            dense_doc_ids: List[str] = []
            for hit in dense_results:
                payload = self._registry.get(hit.vector_id)
                if payload is not None:
                    dense_doc_ids.append(payload.doc_id)
            if expected_doc_id in dense_doc_ids:
                dense_hits += 1
                dense_ranks.append(dense_doc_ids.index(expected_doc_id) + 1)

            fused_response = self._service.search(request)
            fused_doc_ids = [result.doc_id for result in fused_response.results[:10]]
            if expected_doc_id in fused_doc_ids:
                rrf_hits += 1
                rrf_ranks.append(fused_doc_ids.index(expected_doc_id) + 1)

        total_queries = len(sampled_pairs)
        bm25_rate = bm25_hits / total_queries
        splade_rate = splade_hits / total_queries
        dense_rate = dense_hits / total_queries
        rrf_rate = rrf_hits / total_queries

        details: Dict[str, object] = {
            "query_count": total_queries,
            "bm25_hit_rate@10": bm25_rate,
            "splade_hit_rate@10": splade_rate,
            "dense_hit_rate@10": dense_rate,
            "rrf_hit_rate@10": rrf_rate,
            "bm25_avg_rank": statistics.mean(bm25_ranks) if bm25_ranks else None,
            "splade_avg_rank": statistics.mean(splade_ranks) if splade_ranks else None,
            "dense_avg_rank": statistics.mean(dense_ranks) if dense_ranks else None,
            "rrf_avg_rank": statistics.mean(rrf_ranks) if rrf_ranks else None,
        }

        passed = (
            bm25_rate >= thresholds.get("bm25_hit_rate@10", 0.0)
            and splade_rate >= thresholds.get("splade_hit_rate@10", 0.0)
            and dense_rate >= thresholds.get("dense_hit_rate@10", 0.0)
            and rrf_rate >= thresholds.get("rrf_hit_rate@10", 0.0)
        )
        return ValidationReport(name="scale_channel_relevance", passed=passed, details=details)

    def _scale_fusion_mmr(
        self,
        dataset: Sequence[Mapping[str, object]],
        thresholds: Mapping[str, float],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Evaluate diversification impact of MMR versus RRF on sampled queries.

        Args:
            dataset: Dataset entries containing documents and queries.
            thresholds: Diversification thresholds for redundancy and hit-rate deltas.
            rng: Random generator for sampling queries.
            query_sample_size: Number of query pairs to evaluate.

        Returns:
            ValidationReport detailing redundancy reduction and hit-rate deltas.
        """
        sampled_pairs = self._sample_queries(dataset, query_sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_fusion_mmr",
                passed=False,
                details={"error": "no queries available"},
            )

        chunk_lookup = {(chunk.doc_id, chunk.chunk_id): chunk for chunk in self._registry.all()}

        redundancy_reductions: List[float] = []
        rrf_cosines: List[float] = []
        mmr_cosines: List[float] = []
        rrf_hits = 0
        mmr_hits = 0

        for document_payload, query_payload in sampled_pairs:
            expected_doc_id = str(
                query_payload.get("expected_doc_id") or document_payload.get("doc_id")
            )

            baseline_request = self._request_for_query(query_payload, page_size=10)
            baseline_response = self._service.search(baseline_request)
            baseline_doc_ids = [result.doc_id for result in baseline_response.results[:10]]
            baseline_vectors = self._embeddings_for_results(baseline_response.results, chunk_lookup)
            baseline_cos = self._average_pairwise_cos(baseline_vectors)
            rrf_cosines.append(baseline_cos)
            if expected_doc_id in baseline_doc_ids:
                rrf_hits += 1

            mmr_request = self._request_for_query(query_payload, page_size=10)
            mmr_request.diversification = True
            mmr_response = self._service.search(mmr_request)
            mmr_doc_ids = [result.doc_id for result in mmr_response.results[:10]]
            mmr_vectors = self._embeddings_for_results(mmr_response.results, chunk_lookup)
            mmr_cos = self._average_pairwise_cos(mmr_vectors)
            mmr_cosines.append(mmr_cos)
            if expected_doc_id in mmr_doc_ids:
                mmr_hits += 1

            redundancy_reductions.append(baseline_cos - mmr_cos)

        total_queries = len(sampled_pairs)
        mean_reduction = (
            float(sum(redundancy_reductions) / len(redundancy_reductions))
            if redundancy_reductions
            else 0.0
        )
        mean_rrf_cos = float(sum(rrf_cosines) / len(rrf_cosines)) if rrf_cosines else 0.0
        mean_mmr_cos = float(sum(mmr_cosines) / len(mmr_cosines)) if mmr_cosines else 0.0
        rrf_rate = rrf_hits / total_queries
        mmr_rate = mmr_hits / total_queries

        details = {
            "query_count": total_queries,
            "rrf_hit_rate@10": rrf_rate,
            "mmr_hit_rate@10": mmr_rate,
            "avg_rrf_pairwise_cosine": mean_rrf_cos,
            "avg_mmr_pairwise_cosine": mean_mmr_cos,
            "redundancy_reduction": mean_reduction,
            "hit_rate_delta": abs(rrf_rate - mmr_rate),
        }

        passed = mean_reduction >= thresholds.get("mmr_redundancy_reduction", 0.0) and abs(
            rrf_rate - mmr_rate
        ) <= thresholds.get("mmr_hit_rate_delta", float("inf"))
        return ValidationReport(name="scale_fusion_mmr", passed=passed, details=details)

    def _scale_pagination(
        self,
        dataset: Sequence[Mapping[str, object]],
        rng: random.Random,
        sample_size: int = 40,
    ) -> ValidationReport:
        """Ensure page cursors generate disjoint result sets across pages.

        Args:
            dataset: Dataset entries containing documents and queries.
            rng: Random generator for sampling query pairs.
            sample_size: Number of queries to evaluate for pagination.

        Returns:
            ValidationReport indicating pagination overlap issues.
        """
        sampled_pairs = self._sample_queries(dataset, sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_pagination",
                passed=False,
                details={"error": "no queries available"},
            )

        overlap_failures = 0
        order_mismatches = 0
        checked = 0

        for _, query_payload in sampled_pairs:
            base_request = self._request_for_query(query_payload, page_size=20)
            first_response = self._service.search(base_request)
            first_keys = [(res.doc_id, res.chunk_id) for res in first_response.results]
            if first_response.next_cursor:
                second_request = self._request_for_query(query_payload, page_size=20)
                second_request.cursor = first_response.next_cursor
                second_response = self._service.search(second_request)
                second_keys = [(res.doc_id, res.chunk_id) for res in second_response.results]
                if set(first_keys) & set(second_keys):
                    overlap_failures += 1
            repeat_request = self._request_for_query(query_payload, page_size=20)
            repeat_response = self._service.search(repeat_request)
            repeat_keys = [(res.doc_id, res.chunk_id) for res in repeat_response.results]
            if first_keys != repeat_keys:
                order_mismatches += 1
            checked += 1

        details = {
            "queries_checked": checked,
            "overlap_failures": overlap_failures,
            "order_mismatches": order_mismatches,
        }
        passed = overlap_failures == 0 and order_mismatches == 0
        return ValidationReport(name="scale_pagination", passed=passed, details=details)

    def _scale_result_shaping(
        self,
        dataset: Sequence[Mapping[str, object]],
        rng: random.Random,
        sample_size: int = 40,
    ) -> ValidationReport:
        """Confirm per-document limits, deduping, and highlights behave as expected.

        Args:
            dataset: Dataset entries containing documents and queries.
            rng: Random generator used for sampling query pairs.
            sample_size: Number of sampled queries to evaluate.

        Returns:
            ValidationReport summarizing result shaping concerns.
        """
        sampled_pairs = self._sample_queries(dataset, sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_result_shaping",
                passed=False,
                details={"error": "no queries available"},
            )

        config = self._service._config_manager.get()
        max_per_doc = config.fusion.max_chunks_per_doc
        dedupe_threshold = config.fusion.cosine_dedupe_threshold
        chunk_lookup = {(chunk.doc_id, chunk.chunk_id): chunk for chunk in self._registry.all()}
        device = int(config.dense.device)

        doc_limit_violations = 0
        dedupe_violations = 0
        highlight_missing = 0

        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=20)
            response = self._service.search(request)
            doc_counts: Dict[str, int] = {}
            embeddings: List[np.ndarray] = []

            for result in response.results:
                doc_counts[result.doc_id] = doc_counts.get(result.doc_id, 0) + 1
                chunk = chunk_lookup.get((result.doc_id, result.chunk_id))
                if chunk is not None:
                    embeddings.append(chunk.features.embedding)
                if not result.highlights:
                    highlight_missing += 1

            if any(count > max_per_doc for count in doc_counts.values()):
                doc_limit_violations += 1

            # Dedupe check based on cosine similarity
            if len(embeddings) > 1:
                matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
                sims = pairwise_inner_products(
                    matrix,
                    device=device,
                    resources=self._ensure_validation_resources(),
                )
                if np.any(np.triu(sims, k=1) >= dedupe_threshold):
                    dedupe_violations += 1

        details = {
            "queries_checked": len(sampled_pairs),
            "doc_limit_violations": doc_limit_violations,
            "dedupe_violations": dedupe_violations,
            "missing_highlights": highlight_missing,
        }
        passed = doc_limit_violations == 0 and dedupe_violations == 0 and highlight_missing == 0
        return ValidationReport(name="scale_result_shaping", passed=passed, details=details)

    def _scale_backup_restore(
        self,
        dataset: Sequence[Mapping[str, object]],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Validate snapshot/restore under randomized workloads.

        Args:
            dataset: Dataset entries with documents and queries.
            rng: Random generator for sampling queries.
            query_sample_size: Maximum number of query pairs to evaluate.

        Returns:
            ValidationReport highlighting snapshot mismatches, if any.
        """
        sampled_pairs = self._sample_queries(dataset, min(30, query_sample_size), rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_backup_restore",
                passed=False,
                details={"error": "no queries available"},
            )

        snapshot = vectorstore_serialize_state(self._ingestion.faiss_index, self._registry)

        baseline_results: List[List[tuple[str, float]]] = []
        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=15)
            response = self._service.search(request)
            baseline_results.append(
                [(result.doc_id, round(result.score, 6)) for result in response.results[:15]]
            )

        vectorstore_restore_state(self._ingestion.faiss_index, snapshot)

        mismatches = 0
        for (_, query_payload), expected in zip(sampled_pairs, baseline_results):
            request = self._request_for_query(query_payload, page_size=15)
            response = self._service.search(request)
            observed = [(result.doc_id, round(result.score, 6)) for result in response.results[:15]]
            if observed != expected:
                mismatches += 1

        details = {
            "queries_checked": len(sampled_pairs),
            "mismatches": mismatches,
        }
        passed = mismatches == 0
        return ValidationReport(name="scale_backup_restore", passed=passed, details=details)

    def _scale_acl(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Ensure per-namespace ACL tags are enforced across queries.

        Args:
            dataset: Dataset entries providing documents and queries.

        Returns:
            ValidationReport listing ACL violations, if discovered.
        """
        namespace_to_acl: Dict[str, str] = {}
        namespace_queries: Dict[str, List[Mapping[str, object]]] = {}
        for entry in dataset:
            document = entry.get("document", {})
            namespace = str(document.get("namespace", ""))
            metadata = document.get("metadata", {})
            acl_entries = metadata.get("acl", [])
            if acl_entries:
                namespace_to_acl.setdefault(namespace, acl_entries[0])
            for query in entry.get("queries", []):
                namespace_queries.setdefault(namespace, []).append(query)

        violations: List[str] = []
        checked = 0
        for namespace, queries in namespace_queries.items():
            acl_tag = namespace_to_acl.get(namespace)
            if not acl_tag:
                continue
            sample = queries[: min(5, len(queries))]
            for query_payload in sample:
                request = self._request_for_query(query_payload, page_size=10)
                filters = dict(request.filters)
                filters["acl"] = [acl_tag]
                request.filters = filters
                response = self._service.search(request)
                checked += 1
                for result in response.results:
                    acl_values = result.metadata.get("acl")
                    if isinstance(acl_values, list):
                        allowed = acl_tag in acl_values
                    else:
                        allowed = acl_values == acl_tag
                    if not allowed:
                        violations.append(f"namespace={namespace}")
                        break
                # Cross-namespace negative check
                for other_namespace, other_acl in namespace_to_acl.items():
                    if other_namespace == namespace:
                        continue
                    negative_filters = dict(request.filters)
                    negative_filters["acl"] = [other_acl]
                    negative_request = self._request_for_query(query_payload, page_size=10)
                    negative_request.filters = negative_filters
                    negative_response = self._service.search(negative_request)
                    if negative_response.results:
                        violations.append(f"cross-namespace:{namespace}->{other_namespace}")
                        break

        details = {
            "queries_checked": checked,
            "violations": violations,
            "namespaces": sorted(namespace_to_acl.keys()),
        }
        passed = not violations
        return ValidationReport(name="scale_acl", passed=passed, details=details)

    def _scale_performance(
        self,
        dataset: Sequence[Mapping[str, object]],
        thresholds: Mapping[str, float],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Benchmark latency and throughput against configured thresholds.

        Args:
            dataset: Dataset entries containing documents and queries.
            thresholds: Threshold values for latency and headroom metrics.
            rng: Random generator used for sampling query pairs.
            query_sample_size: Number of queries to sample for benchmarking.

        Returns:
            ValidationReport detailing performance metrics.
        """
        sampled_pairs = self._sample_queries(dataset, min(120, query_sample_size * 2), rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_performance",
                passed=False,
                details={"error": "no queries available"},
            )

        total_timings: List[float] = []
        bm25_timings: List[float] = []
        splade_timings: List[float] = []
        dense_timings: List[float] = []
        wall_start = time.perf_counter()

        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=10)
            iter_start = time.perf_counter()
            response = self._service.search(request)
            total_timings.append(
                response.timings_ms.get("total_ms", (time.perf_counter() - iter_start) * 1000)
            )
            bm25_timings.append(response.timings_ms.get("bm25_ms", 0.0))
            splade_timings.append(response.timings_ms.get("splade_ms", 0.0))
            dense_timings.append(response.timings_ms.get("dense_ms", 0.0))

        wall_elapsed = time.perf_counter() - wall_start
        qps = len(sampled_pairs) / wall_elapsed if wall_elapsed > 0 else float("inf")

        p50 = self._percentile(total_timings, 50)
        p95 = self._percentile(total_timings, 95)
        p99 = self._percentile(total_timings, 99)

        estimated_usage_mb = (
            self._ingestion.faiss_index.ntotal * self._ingestion.faiss_index._dim * 4
        ) / (1024 * 1024)
        assumed_capacity_mb = 24000.0
        headroom_fraction = (
            1.0
            if assumed_capacity_mb <= 0
            else max(0.0, 1.0 - (estimated_usage_mb / assumed_capacity_mb))
        )

        details = {
            "query_count": len(sampled_pairs),
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "bm25_avg_ms": statistics.mean(bm25_timings) if bm25_timings else 0.0,
            "splade_avg_ms": statistics.mean(splade_timings) if splade_timings else 0.0,
            "dense_avg_ms": statistics.mean(dense_timings) if dense_timings else 0.0,
            "observed_qps": qps,
            "estimated_vector_memory_mb": estimated_usage_mb,
            "headroom_fraction": headroom_fraction,
        }

        passed = p95 <= thresholds.get(
            "latency_p95_ms", float("inf")
        ) and headroom_fraction >= thresholds.get("gpu_headroom_fraction", 0.0)
        return ValidationReport(name="scale_performance", passed=passed, details=details)

    def _run_calibration(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Record calibration metrics (self-hit accuracy) across oversampling factors.

        Args:
            dataset: Dataset entries (unused, provided for symmetry).

        Returns:
            ValidationReport summarizing calibration accuracy per oversample setting.
        """
        oversamples = [1, 2, 3]
        results: List[Mapping[str, object]] = []
        total_chunks = max(1, self._registry.count())
        for oversample in oversamples:
            hits = 0
            for chunk in self._registry.all():
                top_k = max(1, oversample * 3)
                search_hits = self._ingestion.faiss_index.search(chunk.features.embedding, top_k)
                if search_hits and search_hits[0].vector_id == chunk.vector_id:
                    hits += 1
            accuracy = hits / total_chunks
            results.append({"oversample": oversample, "self_hit_accuracy": accuracy})
        passed = all(
            entry["self_hit_accuracy"] >= 0.95 for entry in results if entry["oversample"] >= 2
        )
        return ValidationReport(name="calibration_sweep", passed=passed, details={"dense": results})

    def _embeddings_for_results(
        self,
        results: Sequence[HybridSearchResult],
        chunk_lookup: Mapping[tuple[str, str], ChunkPayload],
        limit: int = 10,
    ) -> List[np.ndarray]:
        """Retrieve embeddings for the top-N results using a chunk lookup.

        Args:
            results: Search results from which embeddings are needed.
            chunk_lookup: Mapping from (doc_id, chunk_id) to stored payloads.
            limit: Maximum number of results to consider.

        Returns:
            List of embedding vectors associated with the results.
        """
        embeddings: List[np.ndarray] = []
        for result in results[:limit]:
            chunk = chunk_lookup.get((result.doc_id, result.chunk_id))
            if chunk is None:
                continue
            embeddings.append(chunk.features.embedding)
        return embeddings

    def _average_pairwise_cos(self, embeddings: Sequence[np.ndarray]) -> float:
        """Compute average pairwise cosine similarity for a set of embeddings.

        Args:
            embeddings: Sequence of embedding vectors.

        Returns:
            Mean pairwise cosine similarity, or 0.0 when insufficient points exist.
        """
        if len(embeddings) < 2:
            return 0.0
        matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
        config = self._service._config_manager.get()
        sims = pairwise_inner_products(
            matrix,
            device=int(config.dense.device),
            resources=self._ensure_validation_resources(),
        )
        upper = np.triu(sims, k=1)
        values = upper[np.triu_indices_from(upper, k=1)]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    def _ensure_validation_resources(self) -> "faiss.StandardGpuResources":
        """Lazy-create and cache GPU resources for validation-only cosine checks.

        Args:
            None

        Returns:
            FAISS GPU resources reused across validation runs.
        """

        if self._validation_resources is None:
            self._validation_resources = faiss.StandardGpuResources()
        return self._validation_resources

    def _percentile(self, values: Sequence[float], percentile: float) -> float:
        """Return percentile value for a sequence; defaults to 0.0 when empty.

        Args:
            values: Sequence of numeric values.
            percentile: Desired percentile expressed as a value between 0 and 100.

        Returns:
            Percentile value cast to float, or 0.0 when the sequence is empty.
        """
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))

    def _scale_stability(
        self,
        dataset: Sequence[Mapping[str, object]],
        inputs_by_doc: Mapping[str, DocumentInput],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Stress-test search stability under repeated queries and churn.

        Args:
            dataset: Dataset entries containing documents and queries.
            inputs_by_doc: Mapping of document IDs to ingestion inputs.
            rng: Random generator used for sampling queries and churn operations.
            query_sample_size: Number of queries to sample for stability checks.

        Returns:
            ValidationReport detailing stability mismatches.
        """
        sampled_pairs = self._sample_queries(dataset, min(40, query_sample_size), rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_stability",
                passed=False,
                details={"error": "no queries available"},
            )

        repeat_mismatches = 0
        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=15)
            baseline = [res.doc_id for res in self._service.search(request).results]
            for _ in range(4):
                repeat_request = self._request_for_query(query_payload, page_size=15)
                repeat_results = [
                    res.doc_id for res in self._service.search(repeat_request).results
                ]
                if repeat_results != baseline:
                    repeat_mismatches += 1
                    break

        churn_doc_ids = list(inputs_by_doc.keys())
        churn_count = min(30, len(churn_doc_ids))
        if churn_count:
            churn_samples = rng.sample(churn_doc_ids, churn_count)
            self._ingestion.upsert_documents([inputs_by_doc[doc_id] for doc_id in churn_samples])

        churn_failures = 0
        for document_payload, query_payload in sampled_pairs[: min(15, len(sampled_pairs))]:
            expected_doc_id = str(
                query_payload.get("expected_doc_id") or document_payload.get("doc_id")
            )
            request = self._request_for_query(query_payload, page_size=15)
            response = self._service.search(request)
            if expected_doc_id and not any(
                res.doc_id == expected_doc_id for res in response.results
            ):
                churn_failures += 1

        details = {
            "queries_checked": len(sampled_pairs),
            "repeat_mismatches": repeat_mismatches,
            "churn_failures": churn_failures,
        }
        passed = repeat_mismatches == 0 and churn_failures == 0
        return ValidationReport(name="scale_stability", passed=passed, details=details)


# --- Public Functions ---

def load_dataset(path: Path) -> List[Mapping[str, object]]:
    """Load a JSONL dataset describing documents and queries.

    Args:
        path: Path to a JSONL file containing dataset entries.

    Returns:
        List of parsed dataset rows suitable for validation routines.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        json.JSONDecodeError: If any line contains invalid JSON.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    dataset: List[Mapping[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        dataset.append(json.loads(line))
    return dataset


def infer_embedding_dim(dataset: Sequence[Mapping[str, object]]) -> int:
    """Infer dense embedding dimensionality from dataset vector artifacts.

    Args:
        dataset: Sequence of dataset entries containing vector metadata.

    Returns:
        Inferred embedding dimensionality, defaulting to 2560 when unknown.
    """

    for entry in dataset:
        document = entry.get("document", {})
        vector_file = document.get("vector_file")
        if not vector_file:
            continue
        path = Path(str(vector_file))
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            vector = payload.get("Qwen3-4B", {}).get("vector")
            if isinstance(vector, list) and vector:
                return len(vector)
    return 2560


def run_pytest_suites(mode: str, extra_args: Sequence[str]) -> int:
    """Execute hybrid search pytest suites for the requested ``mode``.

    Args:
        mode: Selector for which regression suites to invoke (synthetic, real, scale).
        extra_args: Additional pytest arguments forwarded from the CLI.

    Returns:
        int: Process return code from the invoked pytest command.
    """

    if mode == "synthetic":
        pytest_args: list[str] = []
    elif mode == "real":
        pytest_args = ["--real-vectors", "-m", "real_vectors and not scale_vectors"]
    elif mode == "scale":
        pytest_args = ["--real-vectors", "--scale-vectors", "-m", "scale_vectors"]
    else:
        pytest_args = ["--real-vectors", "--scale-vectors"]

    command = [sys.executable, "-m", "pytest", *pytest_args, *extra_args]
    return subprocess.call(command)


def run_real_vector_ci(output_dir: Path, extra_args: Sequence[str]) -> int:
    """Execute the real-vector CI regression suite and persist validation artifacts.

    Args:
        output_dir: Directory where validation artifacts should be written.
        extra_args: Additional pytest arguments appended to each command invocation.

    Returns:
        int: Non-zero exit code indicates regression failure; zero means success.
    """

    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["REAL_VECTOR_REPORT_DIR"] = str(resolved)

    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "--real-vectors",
            "tests/test_hybrid_search_real_vectors.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "--real-vectors",
            "--scale-vectors",
            "tests/test_hybrid_search_scale.py",
        ],
    ]

    for base_command in commands:
        result = subprocess.call([*base_command, *extra_args])
        if result != 0:
            print(
                f"Real-vector regression suite failed. Check artifacts in {resolved}",
                file=sys.stderr,
            )
            return result

    print(f"Real-vector validation artifacts stored in {resolved}")
    return 0


# --- Module Entry Points ---

def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for running hybrid search validation suites.

    Args:
        argv: Optional list of command-line arguments overriding `sys.argv`.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Hybrid search validation harness")
    parser.add_argument(
        "--dataset",
        default="Data/HybridScaleFixture/dataset.jsonl",
        help="Path to JSONL dataset (defaults to the scale fixture at Data/HybridScaleFixture/dataset.jsonl).",
    )
    parser.add_argument("--config", help="Path to hybrid search config JSON")
    parser.add_argument("--output", default=None, help="Optional output directory for reports")
    parser.add_argument(
        "--mode",
        choices=["basic", "scale"],
        default="basic",
        help="Validation mode (basic for quick checks, scale for full suite)",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional JSON file overriding scale acceptance thresholds",
    )
    parser.add_argument(
        "--query-sample-size",
        type=int,
        default=120,
        help="Maximum queries to sample for scale validations",
    )
    parser.add_argument(
        "--run-tests",
        choices=("synthetic", "real", "scale", "all"),
        help="Execute hybrid search pytest suites instead of validation sweeps.",
    )
    parser.add_argument(
        "--run-real-ci",
        action="store_true",
        help="Run the real-vector CI regression suite and persist validation artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/real_vectors"),
        help="Directory for real-vector regression artifacts (used with --run-real-ci).",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest when executing test suites.",
    )
    args = parser.parse_args(argv)

    extra_pytest_args: list[str] = list(args.pytest_args or [])
    if extra_pytest_args and not (args.run_tests or args.run_real_ci):
        parser.error("--pytest-args can only be used with --run-tests or --run-real-ci")

    if args.run_real_ci:
        raise SystemExit(run_real_vector_ci(args.output_dir, extra_pytest_args))

    if args.run_tests:
        raise SystemExit(run_pytest_suites(args.run_tests, extra_pytest_args))

    dataset = load_dataset(Path(args.dataset))
    if args.config is None:
        parser.error("--config is required when running validation sweeps")

    manager = HybridSearchConfigManager(Path(args.config))
    config = manager.get()
    embedding_dim = infer_embedding_dim(dataset)
    feature_generator = FeatureGenerator(embedding_dim=embedding_dim)
    faiss_index = FaissVectorStore(dim=embedding_dim, config=config.dense)
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
    thresholds_override = None
    if args.thresholds:
        thresholds_override = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))

    output_root = Path(args.output) if args.output else None

    if args.mode == "scale":
        summary = validator.run_scale(
            dataset,
            output_root=output_root,
            thresholds=thresholds_override,
            query_sample_size=args.query_sample_size,
        )
    else:
        summary = validator.run(dataset, output_root=output_root)
    print(json.dumps({"passed": summary.passed, "report_count": len(summary.reports)}, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
