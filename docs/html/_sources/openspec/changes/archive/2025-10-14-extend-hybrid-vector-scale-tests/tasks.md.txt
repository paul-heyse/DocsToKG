## 1. Large-Scale Fixture Preparation
- [x] 1.1 Sample ≥500 chunk/vector pairs from the real corpus covering at least three namespaces and multiple ACL combinations; capture sample metadata distribution for reference.
- [x] 1.2 Validate each sampled record (UUID presence, SPLADE token/weight parity, 2560-d vector norms, namespace/ACL fields) and persist the expanded fixture (chunks, vectors, manifest, query labels) with reproducible seed and file hashes.
- [x] 1.3 Author fixture README detailing regeneration steps, storage footprint, and CI usage; ensure the fixture can be streamed without loading entire files into memory.

## 2. Functional & Correctness Testing
- [x] 2.1 Extend data sanity checks to operate over the 500+ vector fixture and fail on schema/dimension violations or OpenSearch mapping mismatches.
- [x] 2.2 Implement high-volume CRUD + namespace isolation tests that exercise bulk upsert/update/delete, FAISS/OpenSearch parity (`ntotal`, record counts), and namespace filters returning exclusive results.
- [x] 2.3 Build dense FAISS correctness tests (self-hit @1 ≥0.99, perturbation resilience, optional Flat-IP ground truth recall@10 ≥0.95) using randomized samples from the large fixture.
- [x] 2.4 Add BM25 and SPLADE curated query suites (≥100 queries each) with hit-rate@10 ≥0.8 acceptance thresholds; log per-query metrics for debugging.
- [x] 2.5 Construct pagination stability tests for OpenSearch PIT + `search_after` and FAISS page slicing, verifying repeatability and absence of overlap across multiple iterations.
- [x] 2.6 Implement hybrid fusion evaluation: compute RRF metrics (hit-rate@k, MRR@k, NDCG@k) versus individual channels, execute MMR diversification (λ=0.6) and confirm ≥20% redundancy reduction with ≤2% absolute hit-rate loss.
- [x] 2.7 Extend result shaping assertions (collapse budget, cosine dedupe ≥0.98 threshold, highlight coverage, context token budgeting) using the scale dataset.
- [x] 2.8 Add regression coverage for ACL scenarios (role-scoped queries per namespace) ensuring forbidden namespaces return zero results and audit logs capture access attempts.

## 3. Operational, Performance, and Stability Coverage
- [x] 3.1 Implement backup/restore parity tests that snapshot OpenSearch + serialize FAISS, restore into a clean environment, and compare result sets/score tolerances; log diff summaries.
- [x] 3.2 Add performance profiling harness to capture per-channel and end-to-end latency percentiles (P50/P95/P99), throughput vs concurrency curves, and GPU/OpenSearch resource telemetry; store CSV/JSON outputs.
- [x] 3.3 Introduce stability/determinism loops (repeat query batches ≥10 times, post-ingest churn replays) with assertions on ordering consistency and variance bounds.
- [x] 3.4 Execute calibration sweeps for FAISS (`nprobe`, PQ `M`/`nbits`), fusion score normalization strategies, and SPLADE token distributions; produce recommendation artifacts and guardrail alerts for drift (norm anomalies, zero-token docs).

## 4. Reporting & Automation
- [x] 4.1 Integrate all new checks into the validation harness with configurable acceptance thresholds and produce consolidated JSON + human-readable (markdown/HTML) reports summarizing each category.
- [x] 4.2 Persist per-test artifacts (metrics, logs, heatmaps) to a designated report directory with timestamped runs and attach thresholds used during execution.
- [x] 4.3 Create developer and CI entrypoints (CLI flags, scripts) to run synthetic-only, real-vector fast, and full-scale suites; document command usage.
- [x] 4.4 Configure CI/nightly jobs to execute the full-scale suite, collect artifacts, evaluate thresholds, and mark builds failed when any category falls below acceptance; integrate alerting for latency/recall regressions.
- [x] 4.5 Provide documentation for interpreting the scale report, updating thresholds, and extending the regression harness with new corpora or capabilities.
