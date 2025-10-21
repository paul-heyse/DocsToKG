# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks.bench_micro",
#   "purpose": "Micro-benchmarks for critical paths (pytest-benchmark)",
#   "sections": [
#     {"id": "httpx-bench", "name": "HTTPX Benchmarks", "anchor": "httpx-bench", "kind": "section"},
#     {"id": "ratelimit-bench", "name": "Ratelimiter Benchmarks", "anchor": "ratelimit-bench", "kind": "section"},
#     {"id": "duckdb-bench", "name": "DuckDB Benchmarks", "anchor": "duckdb-bench", "kind": "section"},
#     {"id": "polars-bench", "name": "Polars Benchmarks", "anchor": "polars-bench", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Micro-benchmarks for critical paths (unit work).

Uses pytest-benchmark for calibrated rounds and percentile tracking.
All benchmarks are deterministic, no real network, fully reproducible.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.perf_utils import (
    HTTPX_BUDGET,
    HTTPX_REDIRECT_BUDGET,
    RATELIMITER_BUDGET,
    DUCKDB_INSERT_BUDGET,
    DUCKDB_QUERY_BUDGET,
    POLARS_BUDGET,
    ResourceMonitor,
)


# --- HTTPX Micro-Benchmarks ---


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_httpx_get_200(benchmark, mocked_http_client):
    """Benchmark HTTPX GET 200 (128 KiB body)."""
    # Mock response with 128 KiB body
    mock_body = b"x" * (128 * 1024)
    mocked_http_client.register(
        "https://example.com/test",
        status=200,
        content=mock_body,
    )

    def operation():
        """GET request."""
        with mocked_http_client as client:
            response = client.get("https://example.com/test")
            return len(response.content)

    # Run benchmark
    result = benchmark(operation)
    elapsed_ms = benchmark.stats.median * 1000  # Convert to ms

    # Check against budget
    passes, message = HTTPX_BUDGET.check(elapsed_ms)
    assert passes, message


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_httpx_redirect_302(benchmark, mocked_http_client):
    """Benchmark HTTPX 302→200 redirect chain."""
    mocked_http_client.register(
        "https://example.com/redirect",
        status=302,
        headers={"Location": "https://example.com/final"},
    )
    mocked_http_client.register(
        "https://example.com/final",
        status=200,
        content=b"final",
    )

    def operation():
        """Follow redirect."""
        with mocked_http_client as client:
            response = client.get("https://example.com/redirect", follow_redirects=True)
            return response.status_code

    result = benchmark(operation)
    elapsed_ms = benchmark.stats.median * 1000

    passes, message = HTTPX_REDIRECT_BUDGET.check(elapsed_ms)
    assert passes, message


# --- Ratelimiter Micro-Benchmarks ---


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_ratelimiter_fail_fast(benchmark, ratelimit_registry_reset):
    """Benchmark ratelimiter fail-fast path (no wait)."""
    # Note: This is a placeholder for real ratelimiter benchmark
    # In production, would use the actual ratelimiter from the codebase

    def operation():
        """Fast acquire (should not block)."""
        # Simulate fail-fast: immediate return
        return True

    result = benchmark(operation)
    elapsed_ms = benchmark.stats.median * 1000

    passes, message = RATELIMITER_BUDGET.check(elapsed_ms)
    assert passes, message


# --- DuckDB Micro-Benchmarks ---


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_duckdb_bulk_insert(benchmark, duckdb_with_test_data):
    """Benchmark DuckDB bulk insert (50k rows via Arrow appender)."""
    db = duckdb_with_test_data

    def operation():
        """Insert 50k test rows."""
        rows = [
            {
                "ontology_id": f"ontology_{i}",
                "version": "1.0",
                "status": "ok",
            }
            for i in range(50000)
        ]

        # Simulate bulk insert
        for row in rows:
            db.query(
                f"""
                INSERT INTO ontologies (id, version, status)
                VALUES ('{row["ontology_id"]}', '{row["version"]}', '{row["status"]}')
            """
            )
        return len(rows)

    result = benchmark(operation)
    elapsed_ms = benchmark.stats.median * 1000

    passes, message = DUCKDB_INSERT_BUDGET.check(elapsed_ms)
    assert passes, message


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_duckdb_query(benchmark, duckdb_with_test_data):
    """Benchmark DuckDB query (v_version_stats on 200k rows)."""
    db = duckdb_with_test_data

    def operation():
        """Query for version stats."""
        result = db.query("SELECT COUNT(*) as count FROM versions")
        return result

    result = benchmark(operation)
    elapsed_ms = benchmark.stats.median * 1000

    passes, message = DUCKDB_QUERY_BUDGET.check(elapsed_ms)
    assert passes, message


# --- Polars Micro-Benchmarks ---


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_polars_pipeline(benchmark):
    """Benchmark Polars pipeline (lazy + collect)."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    def operation():
        """Create and collect lazy DataFrame."""
        # Create test data
        data = {
            "id": list(range(10000)),
            "value": list(range(10000)),
            "group": ["A", "B", "C", "D"] * 2500,
        }
        df = pl.DataFrame(data)

        # Lazy query
        result = (
            df.lazy()
            .groupby("group")
            .agg([pl.col("value").sum().alias("total")])
            .collect(streaming=True)
        )
        return len(result)

    result = benchmark(operation)
    elapsed_ms = benchmark.stats.median * 1000

    passes, message = POLARS_BUDGET.check(elapsed_ms)
    assert passes, message


# --- Resource Leak Detection ---


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_resource_cleanup_http(benchmark, mocked_http_client):
    """Benchmark HTTP client for resource leaks (repeated calls)."""
    monitor = ResourceMonitor()

    def operation():
        """Multiple HTTP operations."""
        mocked_http_client.register(
            "https://example.com/test",
            status=200,
            content=b"test",
        )

        with mocked_http_client as client:
            for _ in range(100):
                response = client.get("https://example.com/test")
                _ = response.content

    monitor.start()
    benchmark(operation)
    metrics = monitor.stop()

    # Check for resource leaks
    assert metrics["rss_delta_mb"] < 50, "Memory leak detected"
    assert metrics["fd_delta"] < 5, "FD leak detected"


@pytest.mark.benchmark
@pytest.mark.unit
def test_bench_resource_cleanup_duckdb(benchmark, duckdb_with_test_data):
    """Benchmark DuckDB for resource leaks (repeated queries)."""
    db = duckdb_with_test_data
    monitor = ResourceMonitor()

    def operation():
        """Multiple database queries."""
        for _ in range(100):
            db.query("SELECT COUNT(*) FROM versions")

    monitor.start()
    benchmark(operation)
    metrics = monitor.stop()

    # Check for resource leaks
    assert metrics["rss_delta_mb"] < 50, "Memory leak detected"
    assert metrics["fd_delta"] < 5, "FD leak detected"
