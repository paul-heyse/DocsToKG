# Agents Guide - OntologyDownload

Last updated: 2025-10-19

## Mission and Scope
- Mission: Deliver deterministic, secure ontology planning, downloading, and validation that downstream pipelines can trust without rework.
- Scope boundary: In-scope—resolver planning, HTTP fetching with integrity controls, manifest generation, validator execution. Out-of-scope—graph ingestion, vector indexing, domain-specific ontology parsing beyond integrity checks.

## High-Level Architecture & Data Flow
```mermaid
flowchart LR
  A[configs/sources.yaml<br/>ResolvedConfig] --> B[planning.plan_all]
  B --> C{resolvers/*}
  C --> D[io/network.download_stream]
  D --> E[validation.run_validators]
  E --> F[manifests.write_lockfile<br/>validation-report.jsonl]
  D -.-> G[((cache)]):::cache
  classDef cache stroke-dasharray: 3 3;
```
- Components: `cli.py` orchestrator, `planning/` deterministic planner, `resolvers/` provider adapters, `io/` hardened HTTP + filesystem, `validation/` plugin runner.
- Primary data edges: plan items (`plan.jsonl`) → resolver fetch specs → streamed artifacts → validation results + manifests.
- One failure path to consider: Resolver timeout cascades; ensure retry/backoff in `io/network.py` and surface status in manifests.

## Hot Paths & Data Shapes
- Hot paths: `planning.plan_all`, `resolvers.<Provider>.plan/fetch`, `io.network.download_stream`, `validation.run_validators`.
- Typical payload sizes: individual ontologies span roughly 5 MiB–120 MiB in recent synthetic harness runs (P50 ≈ 42 MiB, P90 ≈ 102 MiB, P95 ≈ 111 MiB); manifests tens of KB; plan JSONL entries remain <1 KB.【53616f†L1-L11】
- Key schemas/models: `manifests.py` schema (manifest v1.0), `settings.DownloadConfiguration`, `FetchSpec` dataclass, validation report JSON (see `docs/schemas/ontology-downloader-config.json`).

## Performance Objectives & Baselines
- Targets: TODO define P50/P95 latency per download; TODO throughput target (artifacts/hour); maintain deterministic sorting <1 s for 1k specs.
- Known baseline:
  - `tests/ontology_download/test_download.py::test_download_stream_rate_limiting` – not currently present in tree (track for reintroduction).
  - `tests/ontology_download/test_download_behaviour.py::test_download_stream_fetches_fixture` (closest rate-limit regression guard) – 1 passed in 2.69 s (pytest -q, 2025-10-19 sandbox run).【783aa7†L1-L3】
  - Synthetic throughput probe via the `DocsToKG.OntologyDownload.testing.TestingEnvironment` harness (mirroring the recommended profiling path) recorded: P50 latency ≈ 459 ms, P95 latency ≈ 1.10 s, median throughput ≈ 91.5 MiB/s (P10 ≈ 67.8 MiB/s), chunk size steady at 1 MiB.【53616f†L1-L11】
- Measurement recipe:
  ```bash
  direnv exec . pytest tests/ontology_download/test_download.py::test_download_stream_retries -q
  direnv exec . python -m cProfile -m DocsToKG.OntologyDownload.cli pull hp --dry-run
  ```

## Profiling & Optimization Playbook
- Quick profile:
  ```bash
  direnv exec . python -m cProfile -m DocsToKG.OntologyDownload.cli pull hp --dry-run
  direnv exec . pyinstrument -r html -o profile.html python -m DocsToKG.OntologyDownload.cli pull hp
  ```
- Tactics:
  - Batch resolver metadata fetches where APIs support bulk endpoints (respect rate limits).
  - Stream downloads via `download_stream`; avoid accumulating bytes in memory.
  - Reuse session pool (`SessionPool`) to amortize TLS handshakes.
  - Cache checksum files and HEAD metadata (`_cached_getaddrinfo`, token buckets) judiciously.
  - Fuse manifest transformations (normalize once, reuse across validators) to reduce JSON churn.

## Complexity & Scalability Guidance
- Planning: expected O(n log n) due to sorting fallback chains; ensure no quadratic nested loops when merging defaults/spec overrides.
- Fetching: per artifact O(content_size); ensure linear streaming and constant memory aside from chunk buffer.
- Memory growth: primarily bounded by concurrent downloads (`DownloadConfiguration.concurrent_downloads`) and validator temp directories; chunk buffers currently stream at 1 MiB (verified via synthetic harness).【53616f†L1-L11】
- Large‑N strategy: paginate plan execution using `pull --plan` with sharded plan files; throttle `concurrent_plans`/`concurrent_downloads` to bound open connections.

## I/O, Caching & Concurrency
- I/O patterns: HTTP GET/HEAD via `io/network.py`; writes to `LOCAL_ONTOLOGY_DIR/<id>/<version>/`; manifests + reports in `ARTIFACTS_DIR`.
- Cache keys & invalidation: token bucket keyed by `(service, host)`; DNS cache TTL 120 s; `SessionPool` keyed by service+host; manual invalidation via `io.rate_limit.reset()` or clearing `SessionPool`.
- Concurrency: safe parallelism limited to planner worker pool + download worker pool; shared state (token buckets, DNS cache, session pool) behind locks—avoid ad-hoc threading outside these utilities.

## Invariants to Preserve (change with caution)
- Ordering/determinism: `plan_all` must produce stable ordering for identical inputs.
- Idempotency: repeated `pull` on same plan should yield identical artifacts/manifests.
- Stable IDs/hashes: manifest `fingerprint`, `sha256`, and directory naming must not change without migration.
- Compatibility guarantees: exported API (`__all__`) and schema structure consumed by downstream pipelines.

## Preferred Refactor Surfaces
- Extension points: resolver plugins via `plugins.py`, validator loaders (`load_validator_plugins`), planning merge helpers (`merge_defaults`).
- Low‑risk files: add config knobs in `settings.py`, extend CLI subcommands in `cli.py`, enhance logging in `logging_utils.py`.
- High-risk areas: `io/network.py` streaming logic, manifest schema definitions, cache layout—refactor only with exhaustive tests + migration notes.

## Code Documentation Requirements
- Keep module docstrings and NAVMAP blocks (`api.py`, `resolvers.py`, `__init__.py`) aligned with actual sections.
- Public functions/classes require docstrings covering purpose, parameters, returns, raises; include example usage in CLI docstrings.
- When adding new resolver/validator modules, include docstring summary plus inline comments explaining rate-limit/cache behavior.
- Cross-reference `MODULE_ORGANIZATION_GUIDE.md`, `CODE_ANNOTATION_STANDARDS.md`, and `STYLE_GUIDE.md`; ensure new NAVMAP entries remain consistent.

## Test Matrix & Quality Gates
```bash
direnv exec . just fmt && direnv exec . just lint && direnv exec . just typecheck
direnv exec . pytest tests/ontology_download -q
direnv exec . pytest tests/ontology_download/test_download.py::test_download_stream_rate_limiting -q
```
- TODO add perf-focused test case (e.g., `pytest -q -k "benchmark_ontology_download"` when available).
- Maintain golden fixtures under `tests/ontology_download/fixtures/` for resolvers; update alongside code.

## Failure Modes & Debug Hints
| Symptom | Likely cause | Quick checks |
|---|---|---|
| Repeated HTTP 429/503 | Rate limit misconfiguration or retry loop | Inspect `logs/ontofetch-*.jsonl` for `sleep_sec`, verify `DownloadConfiguration.rate_limits` |
| Non-deterministic manifests | Unstable resolver ordering or timestamp usage | Re-run `plan --dry-run` twice, assert diff; audit sorting keys |
| Validator OOM | Processing huge RDF in memory | Toggle streaming validators, increase chunking, profile `validation.run_validators` |
| SHA mismatch | Upstream content drift or cache corruption | Clear cache directory, re-fetch, compare `expected_checksum` |

## Canonical Commands
```bash
direnv exec . python -m DocsToKG.OntologyDownload.cli plan hp --json
direnv exec . python -m DocsToKG.OntologyDownload.cli pull hp --force
direnv exec . python -m DocsToKG.OntologyDownload.cli validate --in data/ontologies
direnv exec . python -m DocsToKG.OntologyDownload.cli plan-diff hp --baseline artifacts/plan-baseline.json
direnv exec . python -m DocsToKG.OntologyDownload.cli prune --keep 5 --dry-run --json
```
- `just` aliases (if available): `just ontology.plan`, `just ontology.pull`, `just ontology.validate`, `just ontology.doctor`.

## Indexing Hints
- Read first: `cli.py` (entry points), `planning.plan_all`, `io/network.py`, `resolvers/__init__.py`, `validation/__init__.py`.
- High-signal tests: `tests/ontology_download/test_cli.py`, `test_download.py`, `test_resolvers.py`, `test_validators.py`.
- Key schemas/contracts: `docs/schemas/ontology-downloader-config.json`, manifest helpers in `manifests.py`, plugin registry logic in `plugins.py`.

## Ownership & Documentation Links
- Owners/reviewers: `@paul-heyse` (primary reviewer; escalate to `#docs-to-kg` for broad visibility on risky or cross-cutting changes).
- Additional docs: `src/DocsToKG/OntologyDownload/README.md`, `docs/04-api/DocsToKG.OntologyDownload.*`.

## Changelog and Update Procedure
- Update this guide when new resolvers/validators land, performance baselines shift, or invariants change.
- Record updates alongside PRs touching `planning`, `io`, or schema files; bump `Last updated` date and adjust TODOs when resolved.
