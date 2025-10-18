# Agents Guide - ContentDownload

Last updated: 2025-10-18

## Mission and Scope
- Mission: Provide resilient, high-throughput document acquisition from multiple providers with deterministic manifests and retry semantics.
- Scope boundary: In-scope—resolver orchestration, download pipeline, caching, content normalization, manifest/telemetry. Out-of-scope—DocTags conversion, chunking/embedding, ontology-specific fetching.

## High-Level Architecture & Data Flow
```mermaid
flowchart LR
  A[Sources (provider configs)] --> B[Pipeline orchestrator]
  B --> C{Resolver plugins}
  C --> D[Networking/download]
  D --> E[Content store / manifests]
  D -.-> F[Caches]:::cache
  classDef cache stroke-dasharray: 3 3;
```
- Components: `pipeline.py` orchestration, `providers.py` resolver registry, `download.py` write pipeline, `networking.py` HTTP client, `statistics.py/telemetry.py` observability.
- Primary data edges: provider specs → resolver outputs → download tasks → file system + manifest entries.
- One failure path to consider: resolver rate-limit breach causing cascading retries—must ensure token buckets/backoff prevent hammering provider and manifest records failure with metadata.

## Hot Paths & Data Shapes
- Hot paths: `pipeline.ContentDownloadPipeline.run()` (loops providers/docs), `download.execute_download()` (streams content), `networking.fetch_with_retries()` (HTTP I/O), `providers.*Resolver.resolve()` (API calls).
- Typical payload sizes: HTML/PDF tens of KB to multi-MB; metadata JSON <10 KB (TODO gather stats).
- Key schemas/models: manifest schema in `summary.py` / `statistics.py`, resolver config dataclasses `providers/base.py`, download task models.

## Performance Objectives & Baselines
- Targets: TODO define P50 download latency per provider, throughput (docs/min) on baseline hardware, acceptable retry rates.
- Known baseline: TODO capture results from `tests/cli/test_cli_flows.py` or integration benchmark.
- Measurement recipe:
  ```bash
  direnv exec . pytest tests/cli/test_cli_flows.py -q
  direnv exec . python -m cProfile -m DocsToKG.ContentDownload.cli pull --provider arxiv --limit 50 --output /tmp/content
  ```

## Profiling & Optimization Playbook
- Quick profile:
  ```bash
  direnv exec . python -m cProfile -m DocsToKG.ContentDownload.cli pull --provider openalex --limit 20
  direnv exec . pyinstrument -r html -o profile.html python -m DocsToKG.ContentDownload.cli pull --provider zenodo --limit 20
  ```
- Tactics:
  - Batch provider lookups; cache auth tokens and metadata.
  - Stream downloads to disk (chunked reads) to avoid large memory use.
  - Reuse HTTP sessions and respect connection pooling (`networking.py`).
  - Short-circuit duplicates via manifest history or ETag checks.
  - Keep CPU worker pools sized appropriately; avoid per-doc process spawn.
  - Apply exponential backoff with jitter to reduce thundering herd on failures.

## Complexity & Scalability Guidance
- Download pipeline roughly O(number_of_documents); ensure resolver lookups are at most O(1) per doc and avoid nested loops over entire dataset.
- Memory constant aside from active download buffer (stream to disk); track concurrency to bound open file handles and sockets.
- Large‑N strategy: sharding via CLI filters (`--provider`, `--namespace`, TODO config) and concurrency controls; periodic flush of manifest to keep incremental progress.

## I/O, Caching & Concurrency
- I/O patterns: HTTP GET/HEAD, optional metadata API calls (JSON), writes to `Data/Content/` (TODO actual path), manifest JSONL (`summary.py`).
- Cache keys & invalidation: provider-specific caches (ETag/Last-Modified), local file cache keyed by provider/doc ID; ensure invalidation when `--force` used.
- Concurrency: pipeline may use thread/process pools; avoid shared mutable state outside aggregator classes; networking rate limiters (token buckets) guard concurrency.

## Invariants to Preserve (change with caution)
- Provider registry contract—adding/resolving providers must maintain unique IDs and deterministic ordering.
- Manifest schema stability—fields consumed by downstream ingestion (DocParsing) and CLI reporting; update schemas + README if changed.
- Resume semantics—pipeline must skip already-downloaded artifacts unless `--force`.
- Error handling: all failures captured with reason/status; avoid silent skips.

## Preferred Refactor Surfaces
- Add providers via `resolvers/` modules implementing base resolver interface; register in `providers.py`.
- Extend networking logic in `networking.py` (headers, retries) while keeping `download.py` streaming logic intact.
- Observability enhancements in `telemetry.py` or `statistics.py`.
- Avoid large-scale refactors in core pipeline without design review; high coupling with CLI/tests.

## Code Documentation Requirements
- Maintain NAVMAP sections in `pipeline.py`, `download.py`, `providers.py`, `networking.py`, `summary.py`.
- Public CLI functions/classes require docstrings (purpose, params, returns, raises); include examples in CLI help.
- Keep README + schemas synchronized with new flags or manifest fields.
- Follow `MODULE_ORGANIZATION_GUIDE.md`, `CODE_ANNOTATION_STANDARDS.md`, `STYLE_GUIDE.md`; update module docstrings when reorganizing sections.

## Test Matrix & Quality Gates
```bash
direnv exec . ruff check src/DocsToKG/ContentDownload tests
direnv exec . mypy src/DocsToKG/ContentDownload
direnv exec . pytest tests/cli/test_cli_flows.py -q
direnv exec . pytest tests/content_download -q  # TODO confirm folder name
```
- Golden fixtures: TODO locate sample manifest/download fixtures.
- Stress test: run CLI pull with `--limit` on multiple providers in CI; ensure rate limit code tested.

## Failure Modes & Debug Hints
| Symptom | Likely cause | Quick checks |
|---|---|---|
| Repeated HTTP 429 | Rate limit config mismatch | Inspect logs for `retry_after`; adjust token bucket settings in `networking.py`. |
| Disk full errors | No cleanup / large artifacts | Monitor output directory size; enable pruning; consider streaming to cloud storage. |
| Manifest missing entries | Exceptions before manifest write | Ensure pipeline catches errors and writes failure rows; verify `summary.py`. |
| Duplicate downloads | Resume key mismatch | Check manifest doc IDs/sig; ensure resolver returns consistent IDs. |
| CLI crash on provider | Provider auth/config missing | Validate provider config file; inspect `providers` module for required env vars. |

## Canonical Commands
```bash
direnv exec . python -m DocsToKG.ContentDownload.cli pull --provider arxiv --limit 10 --output Data/Content/arxiv
direnv exec . python -m DocsToKG.ContentDownload.cli pull --config configs/content-download.yaml --resume
direnv exec . python -m DocsToKG.ContentDownload.cli doctor   # TODO confirm command name
```

## Indexing Hints
- Read first: `pipeline.py`, `providers.py`, `download.py`, `networking.py`, `summary.py`.
- High-signal tests: `tests/cli/test_cli_flows.py`, TODO add provider-specific tests.
- Schemas/contracts: manifest structure in `summary.py`, provider config dataclasses in `providers/base.py`.

## Ownership & Documentation Links
- Owners/reviewers: TODO_OWNERS (check `CODEOWNERS` entry for `src/DocsToKG/ContentDownload/`).
- Additional docs: `src/DocsToKG/ContentDownload/README.md`, provider-specific docs if available.

## Changelog and Update Procedure
- Update hot paths, manifest schema references, and CLI examples when adding providers or changing pipeline stages.
- Keep examples runnable; adjust TODO targets/baselines as telemetry matures; refresh “Last updated” after major edits.
