"""Tests for synthetic benchmarking helpers."""

from __future__ import annotations

from tests.docparsing.synthetic import (
    format_benchmark_summary,
    generate_synthetic_chunks,
    generate_synthetic_vectors,
    simulate_embedding_benchmark,
)


def test_simulate_embedding_benchmark_reports_speedup() -> None:
    """Synthetic benchmark should model a throughput gain and memory reduction."""

    result = simulate_embedding_benchmark(num_chunks=128, chunk_tokens=256, dense_dim=1024)

    assert result.throughput_gain > 1.0
    assert 0.0 < result.memory_reduction < 1.0

    summary = format_benchmark_summary(result)
    assert "Synthetic benchmark summary" in summary
    assert "Throughput gain:" in summary
    assert f"{result.num_chunks}" in summary
    assert f"{result.dense_dimension}" in summary


def test_generate_synthetic_vectors_aligns_with_chunks() -> None:
    """Synthetic vector generation should preserve ordering and metadata."""

    chunks = generate_synthetic_chunks(num_docs=2, chunks_per_doc=2, base_tokens=50)
    vectors = generate_synthetic_vectors(chunks, dense_dim=8)

    assert len(vectors) == len(chunks)
    for chunk, vector in zip(chunks, vectors):
        assert vector["UUID"] == chunk["uuid"]
        assert len(vector["Qwen3_4B"]["vector"]) == 8
        assert vector["BM25"]["terms"], "Token terms should be populated"
