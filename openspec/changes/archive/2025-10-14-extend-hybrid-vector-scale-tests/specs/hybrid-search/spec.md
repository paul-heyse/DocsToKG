## MODIFIED Requirements

### Requirement: Validation and Calibration Suite
An automated validation harness SHALL cover ingest integrity (field presence, dimension checks), dense self-hit accuracy (≥0.95 @1 for IVF or 1.00 for Flat), sparse relevance sanity (≥90% self-match @10), namespace filtering, pagination stability, fusion efficacy, highlight packaging, and calibration sweeps for `nprobe`/PQ parameters. The harness SHALL support both lightweight synthetic fixtures and a curated real-vector dataset that includes full-dimension Qwen embeddings and SPLADE weights, verify that chunk/vector UUIDs align, ensure BM25/SPLADE features deserialize without loss, and emit timestamped JSON + text reports under `reports/validation/<timestamp>/` for all runs. The harness SHALL additionally support a high-volume fixture (≥500 vectors) that enables scale-oriented correctness, operational, and performance testing while enforcing acceptance thresholds.

#### Scenario: Scale validation run
- **GIVEN** a curated real-vector fixture containing at least 500 chunk/vector pairs across multiple namespaces
- **WHEN** the validation command executes in scale mode
- **THEN** ingest integrity, CRUD parity, dense/BM25/SPLADE correctness, pagination, fusion (RRF/MMR), result shaping, backup/restore, ACL, performance, stability, and calibration checks run end-to-end with recorded metrics and thresholds
- **AND** the resulting report captures per-check pass/fail status, latency percentiles, recall metrics, redundancy measurements, snapshot parity, and guardrail alerts
- **AND** the run fails if any acceptance threshold is missed

## ADDED Requirements

### Requirement: Scale Regression Coverage
The hybrid search regression suite SHALL execute a comprehensive battery of tests over a ≥500-vector corpus that includes: (1) data sanity and schema verification, (2) batch ingest/update/delete with namespace isolation, (3) dense FAISS self-hit/recall evaluations (including optional CPU ground truth), (4) BM25 and SPLADE curated relevance checks, (5) pagination stability across OpenSearch PIT and FAISS, (6) hybrid fusion (RRF) effectiveness and MMR diversification analysis, (7) result shaping collapse/dedupe/highlight validation, (8) backup/restore parity comparisons, (9) ACL security enforcement, (10) performance and capacity profiling, (11) stability/determinism replays, (12) calibration sweeps for FAISS and fusion parameters, and (13) consolidated reporting with acceptance thresholds. Acceptance thresholds SHALL include at minimum: dense self-hit ≥0.99 (Flat) or ≥0.95 (IVF), ANN recall@10 ≥0.95 for calibrated `nprobe`, BM25 and SPLADE hit-rate@10 ≥0.8 on curated sets, RRF outperforming best single channel on NDCG@10, MMR reducing redundancy by ≥20% with ≤2% absolute hit-rate loss, pagination with zero overlaps, backup/restore score parity within defined float tolerances, ACL isolation with zero cross-namespace leakage, p95 end-to-end latency within the documented SLA, GPU memory headroom ≥20%, and deterministic repeats across ≥10 runs. The suite SHALL persist artifacts (JSON/CSV) with inputs, parameters, metrics, and pass/fail outcomes for each category.

#### Scenario: Execute scale regression suite
- **GIVEN** the ≥500-vector fixture and configured thresholds
- **WHEN** the regression entrypoint runs
- **THEN** each test category enumerated above executes, collects metrics (e.g., hit-rate@k, recall@k, latency percentiles, GPU memory usage), and persists artifacts to the designated report directory
- **AND** the suite fails fast if ingest counts diverge or serialization fails, otherwise proceeds and marks specific categories as failed when thresholds are unmet

#### Scenario: Integrate scale suite with CI/nightly automation
- **GIVEN** the continuous integration job is configured with access to the large fixture and compute resources
- **WHEN** the job triggers the scale regression suite
- **THEN** artifacts (reports, logs, metric summaries) are archived, acceptance thresholds are enforced, and build status reflects the pass/fail outcome
- **AND** alerts are raised for latency regressions, recall drops, or guardrail breaches captured during calibration sweeps
