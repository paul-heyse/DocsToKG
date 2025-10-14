## Why

Our hybrid-vector validation today focuses on small curated fixtures (≈3 documents) that exercise correctness but do not emulate production scale or the comprehensive operational checks outlined in the test plan. As a result we cannot confidently assess behavior once hundreds of vectors are present, nor do we cover performance, pagination, ACL, calibration sweeps, or deterministic replay expectations. We need to extend the regression harness so it ingests at least 500 real vectors and executes the broader suite to catch scale-specific regressions before onboarding production corpora.

## What Changes

- Expand the hybrid test harness to build and load a ≥500-vector fixture sourced from the real corpus, preserving namespaces, ACL metadata, and reproducible sampling/hashing for auditability.
- Add automated test flows that cover the complete large-scale suite: (1) data sanity/schema validation, (2) CRUD + namespace isolation, (3) dense FAISS self-hit/recall and optional ground truth comparisons, (4) BM25 relevance, (5) SPLADE relevance, (6) pagination stability (OpenSearch PIT and FAISS slicing), (7) hybrid fusion + MMR diversification effectiveness, (8) result shaping collapse/dedupe/highlighting, (9) backup/restore parity, (10) ACL enforcement, (11) performance & capacity profiling, (12) stability/determinism replays, (13) calibration sweeps for FAISS/fusion parameters, and (14) consolidated regression reporting with explicit acceptance thresholds.
- Instrument the harness to emit structured artifacts (JSON/CSV summaries, latency/throughput traces, calibration heatmaps) and integrate with CI jobs that fail when thresholds (e.g., dense self-hit ≥0.99, BM25/SPLADE hit-rate@10 ≥0.8, RRF redundancy reduction ≥20%, p95 latency ≤ SLA, GPU headroom ≥20%) are not satisfied.

## Impact

- Affected specs: `hybrid-search`
- Affected code: hybrid validation harness, test fixtures, performance instrumentation, CI scripts for hybrid regression suites.
