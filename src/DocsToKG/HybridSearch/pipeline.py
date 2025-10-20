# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.pipeline",
#   "purpose": "Ingestion pipeline, feature generation, and observability helpers",
#   "sections": [
#     {
#       "id": "countersample",
#       "name": "CounterSample",
#       "anchor": "class-countersample",
#       "kind": "class"
#     },
#     {
#       "id": "histogramsample",
#       "name": "HistogramSample",
#       "anchor": "class-histogramsample",
#       "kind": "class"
#     },
#     {
#       "id": "gaugesample",
#       "name": "GaugeSample",
#       "anchor": "class-gaugesample",
#       "kind": "class"
#     },
#     {
#       "id": "metricscollector",
#       "name": "MetricsCollector",
#       "anchor": "class-metricscollector",
#       "kind": "class"
#     },
#     {
#       "id": "tracerecorder",
#       "name": "TraceRecorder",
#       "anchor": "class-tracerecorder",
#       "kind": "class"
#     },
#     {
#       "id": "observability",
#       "name": "Observability",
#       "anchor": "class-observability",
#       "kind": "class"
#     },
#     {
#       "id": "ingesterror",
#       "name": "IngestError",
#       "anchor": "class-ingesterror",
#       "kind": "class"
#     },
#     {
#       "id": "retryableingesterror",
#       "name": "RetryableIngestError",
#       "anchor": "class-retryableingesterror",
#       "kind": "class"
#     },
#     {
#       "id": "ingestmetrics",
#       "name": "IngestMetrics",
#       "anchor": "class-ingestmetrics",
#       "kind": "class"
#     },
#     {
#       "id": "batchcommitresult",
#       "name": "BatchCommitResult",
#       "anchor": "class-batchcommitresult",
#       "kind": "class"
#     },
#     {
#       "id": "ingestsummary",
#       "name": "IngestSummary",
#       "anchor": "class-ingestsummary",
#       "kind": "class"
#     },
#     {
#       "id": "chunkingestionpipeline",
#       "name": "ChunkIngestionPipeline",
#       "anchor": "class-chunkingestionpipeline",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Hybrid-search ingestion pipeline, feature normalisation, and observability.

This module is the operational counterpart to the HybridSearch README section on
chunk ingestion. It streams DocParsing outputs (`*.chunk.jsonl` + embeddings or
Parquet vectors), derives lexical/dense features, and applies namespace routing
while emitting structured telemetry. Key responsibilities include:

- Loading chunk + vector artifacts in lockstep, validating manifests, and
  invoking `LexicalIndex` / `DenseVectorStore` adapters to keep sparse and dense
  stores synchronised.
- Normalising lexical payloads (BM25 stats, SPLADE weights) and dense vectors so
  the downstream FAISS store can assume contiguous `float32` tensors.
- Surfacing ingestion metrics through `Observability`—latency histograms, batch
  counters, GPU utilisation snapshots—mirroring the “Observability” guidance in
  the package README.
- Guarding against misordered DocParsing artifacts by streaming vectors lazily
  with a bounded cache; exceeding the cache limit raises `IngestError` so broken
  inputs are detected before mutating the stores.
- Providing retryable error classes to distinguish between transient ingestion
  issues (e.g., FAISS temp-memory exhaustion) and terminal data problems.

Agents extending ingestion should consult this module together with
`faiss-gpu-wheel-reference.md` to understand how dense features flow into the
custom FAISS GPU wheel managed by `store.ManagedFaissAdapter`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from .features import (  # noqa: F401
    FeatureGenerator,
    sliding_window,
    tokenize,
    tokenize_with_spans,
)
from .interfaces import DenseVectorStore, LexicalIndex
from .types import ChunkFeatures, ChunkPayload, DocumentInput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import ChunkRegistry

# --- Globals ---

__all__ = (
    "CounterSample",
    "GaugeSample",
    "HistogramSample",
    "MetricsCollector",
    "Observability",
    "TraceRecorder",
    "FeatureGenerator",
    "sliding_window",
    "tokenize",
    "tokenize_with_spans",
    "ChunkIngestionPipeline",
    "IngestError",
    "IngestMetrics",
    "BatchCommitResult",
    "IngestSummary",
    "RetryableIngestError",
    "TRAINING_SAMPLE_RNG",
)

TRAINING_SAMPLE_RNG = np.random.default_rng(13)

_PYARROW_MODULE = None
_PYARROW_PARQUET = None

_DEFAULT_VECTOR_CACHE_LIMIT = 10_000


# --- Public Classes ---


@dataclass
class CounterSample:
    """Sample from a counter metric with labels and value."""

    name: str
    labels: Mapping[str, str]
    value: float


@dataclass
class HistogramSample:
    """Sample from a histogram metric with percentile statistics."""

    name: str
    labels: Mapping[str, str]
    count: int
    p50: float
    p95: float
    p99: float


@dataclass
class GaugeSample:
    """Sample from a gauge metric capturing the latest recorded value."""

    name: str
    labels: Mapping[str, str]
    value: float


class MetricsCollector:
    """In-memory metrics collector compatible with Prometheus-style summaries."""

    def __init__(self) -> None:
        """Initialise empty counter, histogram, and gauge registries."""
        self._lock = RLock()
        self._histogram_window = 512

        def _deque_factory() -> Deque[float]:
            return deque(maxlen=self._histogram_window)

        self._counters: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], float] = (
            defaultdict(float)
        )
        self._histograms: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], Deque[float]] = (
            defaultdict(_deque_factory)
        )
        self._gauges: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}

    def increment(self, name: str, amount: float = 1.0, **labels: str) -> None:
        """Increase a counter metric by ``amount`` for the supplied label set.

        Args:
            name: Counter metric identifier.
            amount: Amount to add to the counter.
            **labels: Key/value labels that partition the metric stream.
        """
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += amount

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a new observation for a histogram metric.

        Args:
            name: Histogram metric identifier.
            value: Observation value to append.
            **labels: Key/value labels that partition the metric stream.
        """
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._histograms[key].append(value)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        """Store the latest value for a gauge metric.

        Args:
            name: Gauge metric identifier.
            value: Current value to store.
            **labels: Key/value labels that partition the metric stream.
        """
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._gauges[key] = value

    def percentile(self, name: str, percentile: float, **labels: str) -> Optional[float]:
        """Return the requested percentile for a histogram metric if available.

        Args:
            name: Histogram metric identifier.
            percentile: Desired percentile expressed between 0.0 and 1.0.
            **labels: Key/value labels that partition the metric stream.

        Returns:
            The percentile value when samples exist, otherwise ``None``.
        """
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            samples = list(self._histograms.get(key, []))
        if not samples:
            return None
        percentile = min(max(percentile, 0.0), 1.0)
        sorted_samples = sorted(samples)
        idx = int(percentile * (len(sorted_samples) - 1))
        return sorted_samples[idx]

    def export_counters(self) -> Iterable[CounterSample]:
        """Yield counter samples suitable for serialization."""
        with self._lock:
            items = list(self._counters.items())
        for (name, labels), value in items:
            yield CounterSample(name=name, labels=dict(labels), value=value)

    def export_histograms(self) -> Iterable[HistogramSample]:
        """Yield histogram samples enriched with common percentiles."""
        with self._lock:
            items = list(self._histograms.items())
        for (name, labels), samples in items:
            sorted_samples = sorted(samples)
            count = len(sorted_samples)
            if count == 0:
                continue
            p50 = sorted_samples[int(0.5 * (count - 1))]
            p95 = sorted_samples[int(0.95 * (count - 1))]
            p99 = sorted_samples[int(0.99 * (count - 1))]
            yield HistogramSample(
                name=name, labels=dict(labels), count=count, p50=p50, p95=p95, p99=p99
            )

    def export_gauges(self) -> Iterable[GaugeSample]:
        """Yield gauge samples representing the latest recorded values."""
        with self._lock:
            items = list(self._gauges.items())
        for (name, labels), value in items:
            yield GaugeSample(name=name, labels=dict(labels), value=value)


class TraceRecorder:
    """Context manager producing timing spans for tracing."""

    def __init__(self, metrics: MetricsCollector, logger: logging.Logger) -> None:
        """Create a recorder backed by a metrics sink and structured logger."""
        self._metrics = metrics
        self._logger = logger

    @contextmanager
    def span(self, name: str, **attributes: str) -> Iterator[None]:
        """Record a timed span, emitting metrics and logs with ``attributes``."""
        start = time.perf_counter()
        try:
            yield
            status = "ok"
        except Exception:
            status = "error"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._metrics.observe(f"trace_{name}_ms", duration_ms, **attributes)
            payload = {"span": name, "duration_ms": round(duration_ms, 3), "status": status}
            payload.update(attributes)
            self._logger.info("hybrid-trace", extra={"event": payload})


class Observability:
    """Facade for metrics, structured logging, and tracing."""

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        """Initialise observers with an optional external logger."""
        self._metrics = MetricsCollector()
        self._logger = logger or logging.getLogger("DocsToKG.HybridSearch")
        self._tracer = TraceRecorder(self._metrics, self._logger)

    @property
    def metrics(self) -> MetricsCollector:
        """Return the metrics collector used by the ingestion pipeline."""
        return self._metrics

    @property
    def logger(self) -> logging.Logger:
        """Return the structured logger used for observability events."""
        return self._logger

    def trace(self, name: str, **attributes: str) -> Iterator[None]:
        """Create a tracing span that records timing and metadata."""
        return self._tracer.span(name, **attributes)

    def metrics_snapshot(self) -> Dict[str, list[Mapping[str, object]]]:
        """Export a JSON-serializable snapshot of counters, histograms, and gauges."""
        counters = [sample.__dict__ for sample in self._metrics.export_counters()]
        histograms = [sample.__dict__ for sample in self._metrics.export_histograms()]
        gauges = [sample.__dict__ for sample in self._metrics.export_gauges()]
        return {"counters": counters, "histograms": histograms, "gauges": gauges}


class IngestError(RuntimeError):
    """Base exception for ingestion failures.

    Args:
        message: Description of the ingestion failure.

    Examples:
        >>> raise IngestError("invalid chunk metadata")
        Traceback (most recent call last):
        ...
        IngestError: invalid chunk metadata
    """


class RetryableIngestError(IngestError):
    """Errors that callers should retry (e.g., transient model inference).

    Args:
        message: Description of the transient ingestion issue.

    Examples:
        >>> raise RetryableIngestError("embedding service unavailable")
        Traceback (most recent call last):
        ...
        RetryableIngestError: embedding service unavailable
    """


@dataclass(slots=True)
class IngestMetrics:
    """Simple metrics bundle used by tests.

    Attributes:
        chunks_upserted: Number of chunks upserted during ingestion runs.
        chunks_deleted: Number of chunks removed from storage backends.

    Examples:
        >>> metrics = IngestMetrics(chunks_upserted=3)
        >>> metrics.chunks_upserted
        3
    """

    chunks_upserted: int = 0
    chunks_deleted: int = 0


def _ensure_pyarrow_vectors() -> Tuple[object, object]:
    """Return the pyarrow modules required for parquet ingestion."""

    global _PYARROW_MODULE, _PYARROW_PARQUET
    if _PYARROW_MODULE is not None and _PYARROW_PARQUET is not None:
        return _PYARROW_MODULE, _PYARROW_PARQUET
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via ingestion error path
        raise IngestError(
            "Parquet vector ingestion requires the optional dependency 'pyarrow'. "
            "Install DocsToKG[docparse-parquet] or add pyarrow to the environment."
        ) from exc
    _PYARROW_MODULE = pa
    _PYARROW_PARQUET = pq
    return pa, pq


@dataclass(slots=True)
class BatchCommitResult:
    """Summary describing a single committed ingestion batch."""

    chunk_count: int = 0
    namespaces: Tuple[str, ...] = ()
    vector_ids: Optional[Tuple[str, ...]] = None


@dataclass(slots=True)
class IngestSummary:
    """Aggregate summary returned from :meth:`ChunkIngestionPipeline.upsert_documents`."""

    chunk_count: int = 0
    namespaces: Tuple[str, ...] = ()
    vector_ids: Optional[Tuple[str, ...]] = None


class ChunkIngestionPipeline:
    """Coordinate loading of chunk/vector artifacts and dual writes.

    Attributes:
        _faiss: FAISS index manager responsible for vector persistence.
        _opensearch: Lexical index handling sparse storage.
        _registry: Registry mapping vector identifiers to chunk metadata.
        _metrics: Aggregated ingestion metrics recorded during operations.
        _observability: Observability facade for tracing and logging.

    Examples:
        >>> pipeline = ChunkIngestionPipeline(
        ...     faiss_index=DenseVectorStore(...),
        ...     opensearch=OpenSearchSimulator(),  # from DocsToKG.HybridSearch.store  # doctest: +SKIP
        ...     registry=ChunkRegistry(),
        ... )
        >>> isinstance(pipeline.metrics.chunks_upserted, int)
        True
    """

    def __init__(
        self,
        *,
        faiss_index: DenseVectorStore,
        opensearch: LexicalIndex,
        registry: ChunkRegistry,
        observability: Optional[Observability] = None,
        vector_cache_limit: int = _DEFAULT_VECTOR_CACHE_LIMIT,
        vector_cache_stats_hook: Optional[Callable[[int, DocumentInput], None]] = None,
    ) -> None:
        """Initialise the ingestion pipeline with storage backends and instrumentation.

        Args:
            faiss_index: FAISS index manager responsible for dense vector storage.
            opensearch: Lexical index handling sparse storage.
            registry: Registry mapping vector identifiers to chunk metadata.
            observability: Optional observability façade for tracing and metrics.
            vector_cache_limit: Maximum number of unmatched vector artifacts that may
                remain resident in memory while reconciling chunk/vector streams.
                Defaults to ``10_000`` which bounds memory growth for misordered
                artifacts.
            vector_cache_stats_hook: Optional callback invoked whenever the
                unmatched vector cache changes size. Observability backends may use
                this to surface drift in dashboards.

        Returns:
            None: Constructors perform initialisation side effects only.
        """

        self._faiss = faiss_index
        self._opensearch = opensearch
        self._registry = registry
        self._metrics = IngestMetrics()
        self._observability = observability or Observability()
        self._vector_cache_limit = int(vector_cache_limit)
        if self._vector_cache_limit < 1:
            self._vector_cache_limit = 0
        self._vector_cache_stats_hook = vector_cache_stats_hook
        self._faiss.set_id_resolver(self._registry.resolve_faiss_id)
        attach = getattr(self._registry, "attach_dense_store", None)
        if callable(attach):
            attach(self._faiss)

    @property
    def metrics(self) -> IngestMetrics:
        """Expose cumulative ingestion metrics for inspection.

        Args:
            None

        Returns:
            IngestMetrics capturing counts of upserts and deletions.
        """
        return self._metrics

    @property
    def faiss_index(self) -> DenseVectorStore:
        """Access the FAISS index manager used for vector persistence.

        Args:
            None

        Returns:
            DenseVectorStore associated with the ingestion pipeline.
        """
        return self._faiss

    def upsert_documents(
        self,
        documents: Sequence[DocumentInput],
        *,
        collect_vector_ids: bool = False,
    ) -> IngestSummary:
        """Ingest pre-computed chunk artifacts into FAISS and OpenSearch.

        Args:
            documents: Sequence of document inputs referencing chunk/vector files.
            collect_vector_ids: When ``True`` the resulting summary includes
                committed vector identifiers. Defaults to ``False`` to avoid
                retaining large payload metadata in memory.

        Returns:
            IngestSummary aggregating counts and optional identifiers for the
            committed chunks.

        Raises:
            RetryableIngestError: When transformation fails due to transient issues.
        """

        formats = {
            document.vector_path.suffix.lower()
            for document in documents
            if document.vector_path
        }
        if formats:
            unsupported = formats - {".jsonl", ".parquet"}
            if unsupported:
                raise IngestError(
                    "Unsupported DocParsing vector formats: " + ", ".join(sorted(unsupported))
                )
            if len(formats) > 1:
                examples = {}
                for fmt in sorted(formats):
                    for document in documents:
                        if document.vector_path.suffix.lower() == fmt:
                            examples[fmt] = document.vector_path
                            break
                detail = ", ".join(f"{fmt}: {path}" for fmt, path in examples.items())
                raise IngestError(
                    "Mixed DocParsing vector formats detected; normalise the corpus before ingestion. "
                    + detail
                )

        total_chunks = 0
        namespaces: Set[str] = set()
        vector_ids: List[str] | None = [] if collect_vector_ids else None

        try:
            for document in documents:
                with self._observability.trace(
                    "ingest_document", namespace=document.namespace
                ):
                    loaded = self._load_precomputed_chunks(document)
                if not loaded:
                    continue
                staged_deletions = self._delete_existing_for_doc(
                    document.doc_id, document.namespace
                )
                batch = self._commit_batch(
                    loaded, collect_vector_ids=collect_vector_ids
                )
                total_chunks += batch.chunk_count
                namespaces.update(batch.namespaces)
                if vector_ids is not None and batch.vector_ids:
                    vector_ids.extend(batch.vector_ids)
                if staged_deletions:
                    self.delete_chunks(staged_deletions)
        except RetryableIngestError:
            raise
        except IngestError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            self._observability.logger.exception(
                "chunk-ingest-error", extra={"event": {"error": str(exc)}}
            )
            raise RetryableIngestError("Failed to transform document") from exc

        return IngestSummary(
            chunk_count=total_chunks,
            namespaces=tuple(sorted(namespaces)),
            vector_ids=tuple(vector_ids) if vector_ids is not None else None,
        )

    def delete_chunks(self, vector_ids: Sequence[str]) -> None:
        """Delete chunks from FAISS, OpenSearch, and the registry by vector id.

        Args:
            vector_ids: Collection of vector identifiers to remove.

        Returns:
            None

        Raises:
            None
        """
        with self._observability.trace("ingest_delete", count=str(len(vector_ids))):
            self._faiss.remove(vector_ids)
            self._opensearch.bulk_delete(vector_ids)
            self._registry.delete(vector_ids)
        self._metrics.chunks_deleted += len(vector_ids)
        self._observability.metrics.increment("delete_chunks", len(vector_ids))

    def _commit_batch(
        self,
        batch: Sequence[ChunkPayload],
        *,
        collect_vector_ids: bool = False,
    ) -> BatchCommitResult:
        """Persist a batch of chunks across dense, sparse, and registry stores."""

        if not batch:
            return BatchCommitResult(
                chunk_count=0,
                namespaces=(),
                vector_ids=() if collect_vector_ids else None,
            )

        count = len(batch)
        vector_ids = [chunk.vector_id for chunk in batch]
        faiss_added = False
        lexical_upserted = False

        with self._observability.trace("ingest_dual_write", count=str(count)):
            try:
                self._prepare_faiss(batch)
                self._faiss.add(
                    [chunk.features.embedding for chunk in batch],
                    vector_ids,
                )
                faiss_added = True
                self._opensearch.bulk_upsert(batch)
                lexical_upserted = True
                self._registry.upsert(batch)
            except Exception:
                if lexical_upserted:
                    try:
                        self._opensearch.bulk_delete(vector_ids)
                    except Exception:
                        self._observability.logger.exception(
                            "chunk-ingest-rollback-lexical-error",
                            extra={
                                "event": {
                                    "vector_ids": vector_ids,
                                    "stage": "lexical",
                                }
                            },
                        )
                if faiss_added:
                    try:
                        self._faiss.remove(vector_ids)
                    except Exception:
                        self._observability.logger.exception(
                            "chunk-ingest-rollback-faiss-error",
                            extra={
                                "event": {
                                    "vector_ids": vector_ids,
                                    "stage": "faiss",
                                }
                            },
                        )
                raise

        self._metrics.chunks_upserted += count
        self._observability.metrics.increment("ingest_chunks", count)

        namespaces = tuple(sorted({chunk.namespace for chunk in batch}))
        self._observability.logger.info(
            "chunk-upsert",
            extra={"event": {"count": count, "namespaces": list(namespaces)}},
        )

        vector_ids_result = (
            tuple(vector_ids)
            if collect_vector_ids
            else None
        )

        return BatchCommitResult(
            chunk_count=count,
            namespaces=namespaces,
            vector_ids=vector_ids_result,
        )

    def _prepare_faiss(self, new_chunks: Sequence[ChunkPayload]) -> None:
        """Train the FAISS index if required before adding new vectors.

        Args:
            new_chunks: Newly ingested chunks whose embeddings may train the index.

        Returns:
            None
        """
        if not self._faiss.needs_training():
            return
        training_vectors = self._training_sample(new_chunks)
        self._faiss.train(training_vectors)

    def _training_sample(self, new_chunks: Sequence[ChunkPayload]) -> Sequence[np.ndarray]:
        """Select representative embeddings for FAISS training.

        Args:
            new_chunks: Candidate chunks from the current ingestion batch.

        Returns:
            Sequence of embedding vectors used for index training.
        """
        nlist = int(getattr(self._faiss.config, "nlist", 1024))
        factor = max(1, int(getattr(self._faiss.config, "ivf_train_factor", 8)))
        target = max(1024, nlist * factor)

        reservoir: List[ChunkPayload] = []
        # Reservoir sampling over existing registry entries
        iterator = self._registry.iter_all()
        i = 0
        for item in iterator:
            if i < target:
                reservoir.append(item)
            else:
                j = int(TRAINING_SAMPLE_RNG.integers(0, i + 1))
                if j < target:
                    reservoir[j] = item
            i += 1
        # Stream over new chunks as well
        for item in new_chunks:
            if i < target:
                reservoir.append(item)
            else:
                j = int(TRAINING_SAMPLE_RNG.integers(0, i + 1))
                if j < target:
                    reservoir[j] = item
            i += 1
        if not reservoir:
            return [chunk.features.embedding for chunk in new_chunks]

        cache: Dict[str, np.ndarray] = {}
        for chunk in reservoir:
            embedding = chunk.features.embedding
            if isinstance(embedding, np.ndarray):
                cache[chunk.vector_id] = np.asarray(embedding, dtype=np.float32)

        matrix = self._registry.resolve_embeddings(
            [chunk.vector_id for chunk in reservoir], cache=cache
        )
        return [row for row in matrix]

    def _load_precomputed_chunks(self, document: DocumentInput) -> List[ChunkPayload]:
        """Load chunk and vector artifacts from disk for a document.

        Args:
            document: Document input describing artifact locations and metadata.

        Returns:
            List of populated `ChunkPayload` instances.

        Raises:
            IngestError: If chunk and vector artifacts are inconsistent or missing.
        """
        chunk_entries = self._read_jsonl(document.chunk_path)
        vector_iter = self._iter_vector_file(document.vector_path)
        vector_cache: Dict[str, Mapping[str, object]] = {}

        def _maybe_track_cache() -> None:
            if self._vector_cache_stats_hook is not None:
                self._vector_cache_stats_hook(len(vector_cache), document)
            if self._vector_cache_limit and len(vector_cache) > self._vector_cache_limit:
                raise IngestError(
                    "Vector cache grew beyond the configured safety limit; verify DocParsing "
                    "artifacts are ordered and sorted consistently by UUID before re-running ingestion."
                )

        def _pop_vector(vector_id: str) -> Optional[Mapping[str, object]]:
            cached = vector_cache.pop(vector_id, None)
            _maybe_track_cache()
            if cached is not None:
                return cached
            for entry in vector_iter:
                current_id = str(entry.get("uuid") or entry.get("UUID"))
                if not current_id:
                    continue
                if current_id == vector_id:
                    return entry
                vector_cache[current_id] = entry
                _maybe_track_cache()
            return None

        payloads: List[ChunkPayload] = []
        missing: List[str] = []
        for entry in chunk_entries:
            vector_id = str(entry.get("uuid") or entry.get("UUID"))
            vector_payload = vector_cache.pop(vector_id, None)
            _maybe_track_cache()
            if vector_payload is None:
                vector_payload = _pop_vector(vector_id)
            if vector_payload is None:
                missing.append(vector_id)
                continue
            features = self._features_from_vector(vector_payload)
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "doc_id": document.doc_id,
                    "chunk_id": entry.get("chunk_id"),
                    "source_path": entry.get("source_path"),
                }
            )
            text = str(entry.get("text", ""))
            payloads.append(
                ChunkPayload(
                    doc_id=document.doc_id,
                    chunk_id=str(entry.get("chunk_id")),
                    vector_id=vector_id,
                    namespace=document.namespace,
                    text=text,
                    metadata=metadata,
                    features=features,
                    token_count=int(entry.get("num_tokens", 0)),
                    source_chunk_idxs=tuple(int(idx) for idx in entry.get("source_chunk_idxs", [])),
                    doc_items_refs=tuple(str(ref) for ref in entry.get("doc_items_refs", [])),
                    char_offset=self._resolve_char_offset(entry, text),
                )
            )
        if missing:
            raise IngestError(
                "Missing vector entries for chunk UUIDs: " + ", ".join(sorted(set(missing)))
            )

        extra_vector_ids: List[str] = []
        max_preview = 10
        truncated = False
        if vector_cache:
            cache_ids = [str(vector_id) for vector_id in vector_cache.keys()]
            if len(cache_ids) > max_preview:
                extra_vector_ids.extend(sorted(cache_ids)[:max_preview])
                truncated = True
            else:
                extra_vector_ids.extend(sorted(cache_ids))

        missing_uuid_entries = False
        for extra_entry in vector_iter:
            extra_uuid = extra_entry.get("uuid") or extra_entry.get("UUID")
            if extra_uuid:
                if len(extra_vector_ids) < max_preview:
                    extra_vector_ids.append(str(extra_uuid))
                else:
                    truncated = True
            else:
                missing_uuid_entries = True

        if extra_vector_ids or missing_uuid_entries:
            details = sorted(set(extra_vector_ids))
            message = "Found vector entries without matching chunks"
            if details:
                preview = ", ".join(details[:max_preview])
                if truncated or len(details) > max_preview:
                    preview += ", ..."
                message += f": {preview}"
            if missing_uuid_entries and not details:
                message += ": <missing UUID>"
            elif missing_uuid_entries:
                message += " (additional entries missing UUIDs)"
            raise IngestError(message)
        return payloads

    def _resolve_char_offset(
        self, entry: Mapping[str, object], text: str
    ) -> Tuple[int, int]:
        """Determine the character span for ``entry`` if metadata is present."""

        span_fields = (
            "char_offset",
            "char_offsets",
            "char_range",
            "char_ranges",
            "text_char_range",
            "text_char_offsets",
        )
        for field in span_fields:
            if field in entry:
                span = self._normalise_char_span(entry[field])
                if span is not None:
                    return span
        return (0, len(text))

    @staticmethod
    def _normalise_char_span(span: object) -> Optional[Tuple[int, int]]:
        """Normalise heterogeneous span payloads into a ``(start, end)`` tuple."""

        def _coerce(value: object) -> Optional[int]:
            if isinstance(value, bool):
                return None
            try:
                result = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            if result < 0:
                return None
            return result

        if isinstance(span, (list, tuple)):
            if len(span) >= 2:
                start = _coerce(span[0])
                end = _coerce(span[1])
                if start is not None and end is not None and end >= start:
                    return (start, end)
            return None

        if isinstance(span, Mapping):
            start: Optional[int] = None
            for key in ("start", "begin", "offset", "char_start"):
                if key in span:
                    start = _coerce(span.get(key))
                if start is not None:
                    break
            end: Optional[int] = None
            for key in ("end", "stop", "finish", "char_end"):
                if key in span:
                    end = _coerce(span.get(key))
                if end is not None:
                    break
            if start is not None and end is not None and end >= start:
                return (start, end)
            if start is not None and "length" in span:
                length = _coerce(span.get("length"))
                if length is not None:
                    end = start + length
                    if end >= start:
                        return (start, end)
            return None

        if isinstance(span, str):
            matches = re.findall(r"-?\d+", span)
            if len(matches) >= 2:
                start = _coerce(matches[0])
                end = _coerce(matches[1])
                if start is not None and end is not None and end >= start:
                    return (start, end)
            return None

        return None

    def _delete_existing_for_doc(self, doc_id: str, namespace: str) -> None:
        """Remove previously ingested chunks for a document/namespace pair.

        Args:
            doc_id: Document identifier whose chunks should be removed.
            namespace: Namespace to scope the deletion.

        Returns:
            Tuple of vector identifiers currently associated with the document.
        """
        return tuple(self._registry.vector_ids_for(doc_id, namespace))

    def _features_from_vector(self, payload: Mapping[str, object]) -> ChunkFeatures:
        """Convert stored vector payload into ChunkFeatures.

        Args:
            payload: Serialized feature payload from the vector JSONL artifact.

        Returns:
            ChunkFeatures object with BM25, SPLADE, and dense embeddings.

        Raises:
            IngestError: If the dense embedding has unexpected dimensionality.
        """
        bm25 = payload.get("BM25", {})
        bm25_terms = self._weights_from_payload(bm25)
        splade = payload.get("SpladeV3") or payload.get("SPLADEv3") or {}
        splade_weights = self._weights_from_payload(splade)
        dense = payload.get("Qwen3-4B") or payload.get("Qwen3_4B") or {}
        vector = np.asarray(dense.get("vector", []), dtype=np.float32)
        if vector.ndim != 1:
            raise IngestError("Dense vector must be one-dimensional")
        expected = int(self._faiss.dim)
        if vector.size != expected:
            raise IngestError(
                f"Dense vector dimension mismatch: expected {expected}, got {vector.size}"
            )
        return ChunkFeatures(
            bm25_terms=bm25_terms,
            splade_weights=splade_weights,
            embedding=vector,
        )

    def _weights_from_payload(self, payload: Mapping[str, object]) -> Dict[str, float]:
        """Deserialize sparse weight payloads into a term-to-weight mapping.

        Args:
            payload: Mapping containing `terms`/`tokens` and `weights` arrays.

        Returns:
            Dictionary mapping each term to its corresponding weight.
        """
        terms = payload.get("terms") or payload.get("tokens") or []
        weights = payload.get("weights") or []
        return {str(term): float(weight) for term, weight in zip(terms, weights)}

    def _read_vector_file(self, path: Path) -> List[Dict[str, object]]:
        """Eagerly load vector records from JSONL or Parquet."""

        return list(self._iter_vector_file(path))

    def _iter_vector_file(self, path: Path) -> Iterator[Dict[str, object]]:
        """Stream vector records from ``path`` regardless of format."""

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from self._iter_jsonl(path)
            return
        if suffix == ".parquet":
            yield from self._iter_parquet(path)
            return
        raise IngestError(f"Unsupported vector artifact format for {path}")

    def _iter_parquet(self, path: Path) -> Iterator[Dict[str, object]]:
        """Yield rows from a parquet vectors artifact lazily."""

        if not path.exists():
            raise IngestError(f"Artifact file {path} not found")
        _, pq = _ensure_pyarrow_vectors()
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as exc:  # pragma: no cover - IO errors
            raise IngestError(f"Unable to read parquet vectors from {path}: {exc}") from exc
        for record_batch in parquet_file.iter_batches():
            for entry in record_batch.to_pylist():
                metadata = entry.get("model_metadata")
                if isinstance(metadata, str) and metadata:
                    try:
                        entry["model_metadata"] = json.loads(metadata)
                    except json.JSONDecodeError:
                        entry["model_metadata"] = {}
                elif metadata in (None, ""):
                    entry["model_metadata"] = {}
                yield entry

    def _iter_jsonl(self, path: Path) -> Iterator[Dict[str, object]]:
        """Yield parsed JSON objects from a JSONL artifact lazily."""

        if not path.exists():
            raise IngestError(f"Artifact file {path} not found")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)

    def _read_jsonl(self, path: Path) -> List[Dict[str, object]]:
        """Return the contents of a JSONL artifact eagerly."""

        return list(self._iter_jsonl(path))
