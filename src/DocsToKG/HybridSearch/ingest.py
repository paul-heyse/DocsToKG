"""Ingestion pipeline that materializes pre-computed chunk artifacts."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .dense import FaissIndexManager
from .observability import Observability
from .storage import ChunkRegistry, OpenSearchSimulator
from .types import ChunkFeatures, ChunkPayload, DocumentInput


class IngestError(RuntimeError):
    """Base exception for ingestion failures."""


class RetryableIngestError(IngestError):
    """Errors that callers should retry (e.g., transient model inference)."""


@dataclass(slots=True)
class IngestMetrics:
    """Simple metrics bundle used by tests."""

    chunks_upserted: int = 0
    chunks_deleted: int = 0


class ChunkIngestionPipeline:
    """Coordinate loading of chunk/vector artifacts and dual writes."""

    def __init__(
        self,
        *,
        faiss_index: FaissIndexManager,
        opensearch: OpenSearchSimulator,
        registry: ChunkRegistry,
        observability: Optional[Observability] = None,
    ) -> None:
        self._faiss = faiss_index
        self._opensearch = opensearch
        self._registry = registry
        self._metrics = IngestMetrics()
        self._observability = observability or Observability()
        self._faiss.set_id_resolver(self._registry.resolve_faiss_id)

    @property
    def metrics(self) -> IngestMetrics:
        return self._metrics

    @property
    def faiss_index(self) -> FaissIndexManager:
        return self._faiss

    def upsert_documents(self, documents: Sequence[DocumentInput]) -> List[ChunkPayload]:
        new_chunks: List[ChunkPayload] = []
        try:
            for document in documents:
                with self._observability.trace("ingest_document", namespace=document.namespace):
                    loaded = self._load_precomputed_chunks(document)
                    if loaded:
                        self._delete_existing_for_doc(document.doc_id, document.namespace)
                    new_chunks.extend(loaded)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._observability.logger.exception("chunk-ingest-error", extra={"event": {"error": str(exc)}})
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
            extra={"event": {"count": len(new_chunks), "namespaces": sorted({chunk.namespace for chunk in new_chunks})}},
        )
        return new_chunks

    def delete_chunks(self, vector_ids: Sequence[str]) -> None:
        with self._observability.trace("ingest_delete", count=str(len(vector_ids))):
            self._faiss.remove(vector_ids)
            self._opensearch.bulk_delete(vector_ids)
            self._registry.delete(vector_ids)
        self._metrics.chunks_deleted += len(vector_ids)
        self._observability.metrics.increment("delete_chunks", len(vector_ids))

    def _prepare_faiss(self, new_chunks: Sequence[ChunkPayload]) -> None:
        if not self._faiss.needs_training():
            return
        training_vectors = self._training_sample(new_chunks)
        self._faiss.train(training_vectors)

    def _training_sample(self, new_chunks: Sequence[ChunkPayload]) -> Sequence[np.ndarray]:
        existing = self._registry.all()
        population = list(existing) + list(new_chunks)
        if not population:
            return [chunk.features.embedding for chunk in new_chunks]
        sample_size = min(len(population), max(2048, self._faiss.config.nlist * 256))
        if sample_size >= len(population):
            sample = population
        else:
            sample = random.sample(population, sample_size)
        return [chunk.features.embedding for chunk in sample]

    def _load_precomputed_chunks(self, document: DocumentInput) -> List[ChunkPayload]:
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
        existing_vector_ids = [
            chunk.vector_id
            for chunk in self._registry.all()
            if chunk.doc_id == doc_id and chunk.namespace == namespace
        ]
        if existing_vector_ids:
            self.delete_chunks(existing_vector_ids)

    def _features_from_vector(self, payload: Mapping[str, object]) -> ChunkFeatures:
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
        terms = payload.get("terms") or payload.get("tokens") or []
        weights = payload.get("weights") or []
        return {str(term): float(weight) for term, weight in zip(terms, weights)}

    def _read_jsonl(self, path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            raise IngestError(f"Artifact file {path} not found")
        lines = path.read_text(encoding="utf-8").splitlines()
        entries: List[Dict[str, object]] = []
        for line in lines:
            if not line.strip():
                continue
            entries.append(json.loads(line))
        return entries
