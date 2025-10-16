# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.ingest",
#   "purpose": "Implements DocsToKG.HybridSearch.ingest behaviors and helpers",
#   "sections": [
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
#       "id": "chunkingestionpipeline",
#       "name": "ChunkIngestionPipeline",
#       "anchor": "class-chunkingestionpipeline",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Ingestion pipeline that materializes pre-computed chunk artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .interfaces import LexicalIndex
from .observability import Observability
from .storage import ChunkRegistry
from .types import ChunkFeatures, ChunkPayload, DocumentInput
from .vectorstore import FaissVectorStore

# --- Globals ---

__all__ = (
    "ChunkIngestionPipeline",
    "IngestError",
    "IngestMetrics",
    "RetryableIngestError",
    "TRAINING_SAMPLE_RNG",
)

TRAINING_SAMPLE_RNG = np.random.default_rng(13)


# --- Public Classes ---

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
        ...     faiss_index=FaissVectorStore.build_in_memory(),
        ...     opensearch=OpenSearchSimulator(),  # from DocsToKG.HybridSearch.devtools.opensearch_simulator  # doctest: +SKIP
        ...     registry=ChunkRegistry(),
        ... )
        >>> isinstance(pipeline.metrics.chunks_upserted, int)
        True
    """

    def __init__(
        self,
        *,
        faiss_index: FaissVectorStore,
        opensearch: LexicalIndex,
        registry: ChunkRegistry,
        observability: Optional[Observability] = None,
    ) -> None:
        """Initialise the ingestion pipeline with storage backends and instrumentation.

        Args:
            faiss_index: FAISS index manager responsible for dense vector storage.
            opensearch: Lexical index handling sparse storage.
            registry: Registry mapping vector identifiers to chunk metadata.
            observability: Optional observability façade for tracing and metrics.

        Returns:
            None: Constructors perform initialisation side effects only.
        """

        self._faiss = faiss_index
        self._opensearch = opensearch
        self._registry = registry
        self._metrics = IngestMetrics()
        self._observability = observability or Observability()
        self._faiss.set_id_resolver(self._registry.resolve_faiss_id)

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
    def faiss_index(self) -> FaissVectorStore:
        """Access the FAISS index manager used for vector persistence.

        Args:
            None

        Returns:
            FaissVectorStore associated with the ingestion pipeline.
        """
        return self._faiss

    def upsert_documents(self, documents: Sequence[DocumentInput]) -> List[ChunkPayload]:
        """Ingest pre-computed chunk artifacts into FAISS and OpenSearch.

        Args:
            documents: Sequence of document inputs referencing chunk/vector files.

        Returns:
            List of `ChunkPayload` objects that were successfully upserted.

        Raises:
            RetryableIngestError: When transformation fails due to transient issues.
        """
        new_chunks: List[ChunkPayload] = []
        try:
            for document in documents:
                with self._observability.trace("ingest_document", namespace=document.namespace):
                    loaded = self._load_precomputed_chunks(document)
                    if loaded:
                        self._delete_existing_for_doc(document.doc_id, document.namespace)
                    new_chunks.extend(loaded)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._observability.logger.exception(
                "chunk-ingest-error", extra={"event": {"error": str(exc)}}
            )
            raise RetryableIngestError("Failed to transform document") from exc

        if not new_chunks:
            return []

        with self._observability.trace("ingest_dual_write", count=str(len(new_chunks))):
            self._prepare_faiss(new_chunks)
            self._faiss.add(
                [chunk.features.embedding for chunk in new_chunks],
                [chunk.vector_id for chunk in new_chunks],
            )
            self._opensearch.bulk_upsert(new_chunks)
            self._registry.upsert(new_chunks)
        self._metrics.chunks_upserted += len(new_chunks)
        self._observability.metrics.increment("ingest_chunks", len(new_chunks))
        self._observability.logger.info(
            "chunk-upsert",
            extra={
                "event": {
                    "count": len(new_chunks),
                    "namespaces": sorted({chunk.namespace for chunk in new_chunks}),
                }
            },
        )
        return new_chunks

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
        existing = self._registry.all()
        population = list(existing) + list(new_chunks)
        if not population:
            return [chunk.features.embedding for chunk in new_chunks]
        nlist = int(getattr(self._faiss.config, "nlist", 1024))
        factor = max(1, int(getattr(self._faiss.config, "ivf_train_factor", 8)))
        sample_size = min(len(population), max(1024, nlist * factor))
        if sample_size >= len(population):
            sample = population
        else:
            indices = TRAINING_SAMPLE_RNG.choice(len(population), size=sample_size, replace=False)
            sample = [population[idx] for idx in indices]
        return [chunk.features.embedding for chunk in sample]

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
        vector_entries = {entry["UUID"]: entry for entry in self._read_jsonl(document.vector_path)}
        payloads: List[ChunkPayload] = []
        for entry in chunk_entries:
            vector_id = str(entry.get("uuid"))
            vector_payload = vector_entries.get(vector_id)
            if vector_payload is None:
                raise IngestError(f"Missing vector entry for chunk {vector_id}")
            features = self._features_from_vector(vector_payload)
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "doc_id": document.doc_id,
                    "chunk_id": entry.get("chunk_id"),
                    "source_path": entry.get("source_path"),
                }
            )
            payloads.append(
                ChunkPayload(
                    doc_id=document.doc_id,
                    chunk_id=str(entry.get("chunk_id")),
                    vector_id=vector_id,
                    namespace=document.namespace,
                    text=str(entry.get("text", "")),
                    metadata=metadata,
                    features=features,
                    token_count=int(entry.get("num_tokens", 0)),
                    source_chunk_idxs=tuple(int(idx) for idx in entry.get("source_chunk_idxs", [])),
                    doc_items_refs=tuple(str(ref) for ref in entry.get("doc_items_refs", [])),
                    char_offset=(0, len(str(entry.get("text", "")))),
                )
            )
        return payloads

    def _delete_existing_for_doc(self, doc_id: str, namespace: str) -> None:
        """Remove previously ingested chunks for a document/namespace pair.

        Args:
            doc_id: Document identifier whose chunks should be removed.
            namespace: Namespace to scope the deletion.

        Returns:
            None
        """
        existing_vector_ids = [
            chunk.vector_id
            for chunk in self._registry.all()
            if chunk.doc_id == doc_id and chunk.namespace == namespace
        ]
        if existing_vector_ids:
            self.delete_chunks(existing_vector_ids)

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
        splade = payload.get("SpladeV3", {})
        splade_weights = self._weights_from_payload(splade)
        dense = payload.get("Qwen3-4B", {})
        vector = np.array(dense.get("vector", []), dtype=np.float32)
        if vector.ndim != 1:
            raise IngestError("Dense vector must be one-dimensional")
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

    def _read_jsonl(self, path: Path) -> List[Dict[str, object]]:
        """Load JSONL content from disk and parse each line into a dictionary.

        Args:
            path: Path to the JSONL artifact.

        Returns:
            List of parsed entries.

        Raises:
            IngestError: If the artifact file is missing.
            json.JSONDecodeError: If any JSON line cannot be parsed.
        """
        if not path.exists():
            raise IngestError(f"Artifact file {path} not found")
        lines = path.read_text(encoding="utf-8").splitlines()
        entries: List[Dict[str, object]] = []
        for line in lines:
            if not line.strip():
                continue
            entries.append(json.loads(line))
        return entries
