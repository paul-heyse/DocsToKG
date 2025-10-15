"""Synthetic data factories and benchmarking primitives for DocParsing tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Sequence

__all__ = [
    "SyntheticBenchmarkResult",
    "generate_synthetic_chunks",
    "generate_synthetic_vectors",
    "simulate_embedding_benchmark",
    "format_benchmark_summary",
]


@dataclass
class SyntheticBenchmarkResult:
    """Synthetic measurements comparing naive vs streaming embeddings."""

    num_chunks: int
    chunk_tokens: int
    dense_dimension: int
    naive_time_s: float
    streaming_time_s: float
    naive_peak_mb: float
    streaming_peak_mb: float

    @property
    def throughput_gain(self) -> float:
        """Multiplicative throughput gain of the streaming path."""

        return self.naive_time_s / self.streaming_time_s if self.streaming_time_s else 0.0

    @property
    def memory_reduction(self) -> float:
        """Fractional memory reduction achieved by streaming embeddings."""

        if self.naive_peak_mb == 0:
            return 0.0
        return 1.0 - (self.streaming_peak_mb / self.naive_peak_mb)


def generate_synthetic_chunks(
    num_docs: int = 1,
    chunks_per_doc: int = 3,
    base_tokens: int = 120,
) -> List[dict]:
    """Produce deterministic chunk rows for testing without Docling."""

    rows: List[dict] = []
    for doc_idx in range(num_docs):
        doc_id = f"doc-{doc_idx}"
        source_path = f"/synthetic/{doc_id}.doctags"
        for chunk_idx in range(chunks_per_doc):
            token_count = base_tokens + chunk_idx * 7
            text = f"Synthetic paragraph {chunk_idx} for {doc_id}."
            chunk_uuid = str(uuid.uuid4())
            rows.append(
                {
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "chunk_id": chunk_idx,
                    "source_chunk_idxs": [chunk_idx],
                    "num_tokens": token_count,
                    "text": text,
                    "doc_items_refs": [],
                    "page_nos": [chunk_idx + 1],
                    "schema_version": "docparse/1.1.0",
                    "provenance": {
                        "parse_engine": "docling-html",
                        "docling_version": "synthetic",
                        "has_image_captions": False,
                        "has_image_classification": False,
                        "num_images": 0,
                    },
                    "uuid": chunk_uuid,
                }
            )
    return rows


def generate_synthetic_vectors(
    chunks: Sequence[dict],
    dense_dim: int = 2560,
) -> List[dict]:
    """Generate synthetic embedding rows aligned with ``chunks``."""

    vectors: List[dict] = []
    for chunk in chunks:
        uuid_value = chunk["uuid"]
        text = chunk["text"]
        terms = text.lower().split()
        weights = [round(1.0 / max(len(terms), 1), 4) for _ in terms]
        base = sum(ord(ch) for ch in text) % 997
        dense_vector = [round(((base + i) % 997) / 997.0, 6) for i in range(dense_dim)]
        vectors.append(
            {
                "UUID": uuid_value,
                "BM25": {
                    "terms": terms,
                    "weights": weights,
                    "k1": 1.5,
                    "b": 0.75,
                    "avgdl": 128.0,
                    "N": max(len(chunks), 1),
                },
                "SPLADEv3": {
                    "tokens": terms,
                    "weights": weights,
                },
                "Qwen3_4B": {
                    "model_id": "Qwen/Qwen3-Embedding-4B",
                    "vector": dense_vector,
                    "dimension": dense_dim,
                },
                "model_metadata": {
                    "splade": {"batch_size": 1},
                    "qwen": {"batch_size": 1, "dtype": "bfloat16"},
                },
                "schema_version": "embeddings/1.0.0",
            }
        )
    return vectors


def simulate_embedding_benchmark(
    num_chunks: int = 512,
    chunk_tokens: int = 384,
    dense_dim: int = 2560,
) -> SyntheticBenchmarkResult:
    """Estimate streaming improvements using a closed-form synthetic model."""

    naive_time = max(num_chunks * chunk_tokens / 52000.0, 0.05)
    naive_time_s = round(naive_time, 3)
    streaming_time_s = round(naive_time_s * 0.58, 3)
    naive_peak_mb = round(num_chunks * dense_dim * 4 / (1024**2), 2)
    streaming_peak_mb = round(naive_peak_mb * 0.42, 2)
    return SyntheticBenchmarkResult(
        num_chunks=num_chunks,
        chunk_tokens=chunk_tokens,
        dense_dimension=dense_dim,
        naive_time_s=naive_time_s,
        streaming_time_s=streaming_time_s,
        naive_peak_mb=naive_peak_mb,
        streaming_peak_mb=streaming_peak_mb,
    )


def format_benchmark_summary(result: SyntheticBenchmarkResult) -> str:
    """Return a human readable summary for CLI and documentation output."""

    speedup = result.throughput_gain
    reduction = result.memory_reduction
    return (
        "Synthetic benchmark summary\n"
        f"Chunks processed: {result.num_chunks} (@ {result.chunk_tokens} tokens)\n"
        f"Dense dimension: {result.dense_dimension}\n"
        f"Naive time: {result.naive_time_s:.3f}s, Streaming time: {result.streaming_time_s:.3f}s\n"
        f"Throughput gain: {speedup:.2f}x, Peak memory reduction: {reduction:.0%}\n"
        f"Naive peak memory: {result.naive_peak_mb:.2f} MiB\n"
        f"Streaming peak memory: {result.streaming_peak_mb:.2f} MiB"
    )
