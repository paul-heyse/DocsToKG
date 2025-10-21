# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks",
#   "purpose": "Performance benchmarking infrastructure for Optimization 10",
#   "sections": [
#     {"id": "micro-benches", "name": "Micro-Benchmarks", "anchor": "micro-benches", "kind": "section"},
#     {"id": "macro-perf", "name": "Macro E2E Performance", "anchor": "macro-perf", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Performance benchmarking suite for Optimization 10.

This package contains comprehensive performance benchmarks for critical
paths across the codebase:

Micro-Benchmarks (unit work):
- HTTPX policy and streaming
- Ratelimiter acquisition
- Archive extraction
- DuckDB operations
- Polars pipelines

Macro E2E (end-to-end):
- smoke_perf: PR lane benchmarks (fast, <25s)
- nightly_perf: Full suite (hours, comprehensive)

Profiling & Regression:
- Baseline storage and comparison
- Resource leak detection
- CPU/memory profiling hooks
"""
