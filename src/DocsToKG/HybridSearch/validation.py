"""Automated validation harness for the hybrid search stack."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .ingest import ChunkIngestionPipeline
from .observability import Observability
from .config import HybridSearchConfigManager
from .dense import FaissIndexManager
from .features import FeatureGenerator
from .retrieval import HybridSearchService
from .storage import ChunkRegistry, OpenSearchSimulator
from .types import DocumentInput, HybridSearchRequest, ValidationReport, ValidationSummary


def load_dataset(path: Path) -> List[Mapping[str, object]]:
    """Load a JSONL dataset describing documents and queries."""

    lines = path.read_text(encoding="utf-8").splitlines()
    dataset: List[Mapping[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        dataset.append(json.loads(line))
    return dataset


def infer_embedding_dim(dataset: Sequence[Mapping[str, object]]) -> int:
    """Infer dense embedding dimensionality from dataset vector artifacts."""

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


class HybridSearchValidator:
    """Execute validation sweeps and persist reports."""

    def __init__(
        self,
        *,
        ingestion: ChunkIngestionPipeline,
        service: HybridSearchService,
        registry: ChunkRegistry,
        opensearch: OpenSearchSimulator,
    ) -> None:
        self._ingestion = ingestion
        self._service = service
        self._registry = registry
        self._opensearch = opensearch

    def run(self, dataset: Sequence[Mapping[str, object]], output_root: Optional[Path] = None) -> ValidationSummary:
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
        self._persist_reports(summary, output_root, calibration.details if calibration.details else None)
        return summary

    def _to_document(self, payload: Mapping[str, object]) -> DocumentInput:
        return DocumentInput(
            doc_id=str(payload["doc_id"]),
            namespace=str(payload["namespace"]),
            chunk_path=Path(str(payload["chunk_file"])).resolve(),
            vector_path=Path(str(payload["vector_file"])).resolve(),
            metadata=dict(payload.get("metadata", {})),
        )

    def _check_ingest_integrity(self) -> ValidationReport:
        all_chunks = self._registry.all()
        ok = all(len(chunk.features.embedding.shape) == 1 for chunk in all_chunks)
        details = {"total_chunks": len(all_chunks)}
        return ValidationReport(name="ingest_integrity", passed=ok, details=details)

    def _check_dense_self_hit(self) -> ValidationReport:
        ok = True
        for chunk in self._registry.all():
            hits = self._ingestion.faiss_index.search(chunk.features.embedding, 1)
            if not hits or hits[0].vector_id != chunk.vector_id:
                ok = False
                break
        return ValidationReport(name="dense_self_hit", passed=ok, details={})

    def _check_sparse_relevance(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                expected = query.get("expected_doc_id")
                request = self._request_for_query(query)
                response = self._service.search(request)
                if not response.results:
                    ok = False
                    break
                if expected and response.results[0].doc_id != expected:
                    ok = False
                    break
            if not ok:
                break
        return ValidationReport(name="sparse_relevance", passed=ok, details={})

    def _check_namespace_filters(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
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
        if not dataset:
            return ValidationReport(name="backup_restore", passed=False, details={"error": "dataset empty"})
        queries = dataset[0].get("queries", [])
        if not queries:
            return ValidationReport(name="backup_restore", passed=False, details={"error": "missing queries"})
        faiss_bytes = self._ingestion.faiss_index.serialize()
        first_query = self._request_for_query(queries[0])
        before = self._service.search(first_query)
        self._ingestion.faiss_index.restore(faiss_bytes)
        after = self._service.search(first_query)
        ok = bool(before.results and after.results and before.results[0].doc_id == after.results[0].doc_id)
        return ValidationReport(name="backup_restore", passed=ok, details={})

    def _request_for_query(self, query: Mapping[str, object], page_size: int = 5) -> HybridSearchRequest:
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
    ) -> None:
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
        (directory / "summary.json").write_text(json.dumps(reports_json, indent=2), encoding="utf-8")
        human_lines = [
            f"Validation started at: {summary.started_at.isoformat()}",
            f"Validation completed at: {summary.completed_at.isoformat()}",
            f"Overall status: {'PASS' if summary.passed else 'FAIL'}",
        ]
        for report in summary.reports:
            human_lines.append(f"- {report.name}: {'PASS' if report.passed else 'FAIL'}")
        (directory / "summary.txt").write_text("\n".join(human_lines), encoding="utf-8")
        if calibration_details is not None:
            (directory / "calibration.json").write_text(json.dumps(calibration_details, indent=2), encoding="utf-8")

    def _run_calibration(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
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
        passed = all(entry["self_hit_accuracy"] >= 0.95 for entry in results if entry["oversample"] >= 2)
        return ValidationReport(name="calibration_sweep", passed=passed, details={"dense": results})


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid search validation harness")
    parser.add_argument(
        "--dataset",
        default="tests/data/real_hybrid_dataset/dataset.jsonl",
        help="Path to JSONL dataset (defaults to real fixture at tests/data/real_hybrid_dataset/dataset.jsonl). Use tests/data/hybrid_dataset.jsonl for the synthetic suite.",
    )
    parser.add_argument("--config", required=True, help="Path to hybrid search config JSON")
    parser.add_argument("--output", default=None, help="Optional output directory for reports")
    args = parser.parse_args(argv)

    dataset = load_dataset(Path(args.dataset))
    manager = HybridSearchConfigManager(Path(args.config))
    config = manager.get()
    embedding_dim = infer_embedding_dim(dataset)
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
    summary = validator.run(dataset, output_root=Path(args.output) if args.output else None)
    print(json.dumps({"passed": summary.passed, "report_count": len(summary.reports)}, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
