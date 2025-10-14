## Context

The ContentDownload module orchestrates PDF acquisition from OpenAlex with fallback to 10 resolver sources (Unpaywall, Crossref, landing pages, arXiv, PMC, Europe PMC, CORE, DOAJ, Semantic Scholar, Wayback). The current implementation handles ~90% of happy-path scenarios but exposes operational fragility: transient network errors cause unnecessary misses, partial downloads corrupt storage, and single-threaded processing underutilizes available bandwidth. Recent production runs show 5-8% miss rate attributable to retryable failures and 2-3% corrupted artifacts requiring manual cleanup.

## Goals / Non-Goals

- Goals:
  - Eliminate transient failure misses through automatic retry with backoff (target: <1% miss rate from retryable errors)
  - Prevent partial file corruption via atomic writes (zero corrupt artifacts)
  - Clarify rate-limit configuration to prevent misconfiguration (explicit units)
  - Reduce code duplication by 15% through shared utilities
  - Enable 2-5x throughput via opt-in parallelism while respecting per-resolver rate limits
  - Support idempotent re-runs with conditional requests (ETags)
  - Unify logging for simplified operations
- Non-Goals:
  - Changing resolver priority order or adding new required dependencies
  - Real-time streaming/event-driven architecture (batch processing sufficient)
  - Distributed coordination across machines (single-process parallelism only)
  - Replacing BeautifulSoup or modifying landing page scraping patterns

## Decisions

- Decision: **Use urllib3 Retry with HTTPAdapter on shared session** for transparent transient error handling (429, 502, 503, 504) with exponential backoff and Retry-After support.
  Alternatives: custom retry loop per request, no retry. Rationale: HTTPAdapter + Retry is battle-tested, respects Retry-After headers, and requires zero per-call changes; eliminates ~80% of retryable misses observed in production.

- Decision: **Implement atomic writes with *.part + os.replace() pattern** and compute SHA-256 digests before finalizing.
  Alternatives: write direct to final path, use temp directory. Rationale:*.part suffix makes partials obvious; os.replace() is atomic on POSIX/Windows; SHA-256 enables deduplication and integrity verification; minimal code change.

- Decision: **Rename `resolver_rate_limits` → `min_interval_s`** to clarify units (seconds between calls).
  Alternatives: keep ambiguous name, add documentation. Rationale: field name encodes units explicitly; prevents "1.0 means 1 QPS" misconfiguration seen in operator feedback; one-time migration cost justified by preventing ongoing confusion.

- Decision: **Add LRU cache (maxsize=1000) for resolver API responses** keyed by (resolver_name, identifier).
  Alternatives: no caching, external cache layer. Rationale: batch runs frequently retry same DOIs; in-memory LRU is trivial (functools.lru_cache), eliminates redundant API calls, cache size tuned to typical batch (500-2000 works); no external dependencies.

- Decision: **Support conditional requests with If-None-Match/If-Modified-Since** using ETag/Last-Modified from previous manifest.
  Alternatives: always redownload, external change detection. Rationale: HTTP 304 responses are instant and free; enables safe idempotent reruns; manifest already records these headers; minimal logic addition.

- Decision: **Extract shared utilities to ContentDownload/utils.py** for `_normalize_doi`, `_normalize_pmcid`, `_strip_prefix`, `dedupe`.
  Alternatives: keep duplicates, create separate package. Rationale: identical code in two modules; utils.py is single-responsibility, no circular imports; 15% code reduction; easier to test/maintain.

- Decision: **Unify logging to single JSONL format** with per-attempt records and per-work summaries, provide CSV export script.
  Alternatives: keep separate CSV/manifest, add third logger. Rationale: JSONL is machine-readable and human-inspectable; one logger = one truth; CSV export script maintains backward compatibility; reduces cognitive load; enables richer structured queries.

- Decision: **Use ThreadPoolExecutor with max_workers=N** for bounded parallelism across works, serialize per-work pipeline.
  Alternatives: asyncio, multiprocessing, no parallelism. Rationale: ThreadPoolExecutor simple and stdlib; GIL not bottleneck (network-bound); serialize per-work preserves rate-limit semantics; N=3-5 gives 3x throughput without overwhelming sources.

- Decision: **Refactor `download_candidate` with state enum** (PENDING/WRITING) and single outcome builder.
  Alternatives: keep existing if/else chains. Rationale: explicit state machine improves readability; single return path eases testing; preserves all existing validation (EOF check, sniff logic); reduces nesting.

- Decision: **Make `--workers` opt-in (default=1)** to avoid surprising existing deployments.
  Alternatives: enable by default. Rationale: parallelism changes egress patterns; operators should consciously enable; default=1 is backward compatible; documentation encourages tuning.

- Decision: **Add `--dry-run` and `--resume-from` flags** for coverage measurement and idempotent resumption.
  Alternatives: separate scripts, always full run. Rationale: dry-run enables quick resolver testing without storage; resume-from reduces wasted API calls on reruns; both fit naturally into CLI; minimal code.

- Decision: **Extract plaintext from HTML fallbacks** using trafilatura (optional dependency).
  Alternatives: keep raw HTML only, use BeautifulSoup get_text(). Rationale: trafilatura designed for article extraction; optional = no new hard dependency; *.html.txt simplifies downstream parsing; 1-5 line addition to existing HTML save.

- Decision: **Add OpenAIRE, HAL, OSF resolvers as optional** (disabled by default).
  Alternatives: enable by default, skip entirely. Rationale: increases EU OA and preprint coverage; mirrors existing JSON-parsing patterns; disabled by default = no surprise egress; operators enable via config.

## Architecture Overview

- **Session Management**: Single `requests.Session` created with polite headers and HTTPAdapter(Retry(...)) mounted on http:// and https://. Shared across OpenAlex attempts and resolver pipeline.

- **Atomic Write Flow**: In `download_candidate`, after classification:
  1. Open `dest_path.with_suffix('.part')`
  2. Stream chunks to .part file
  3. On completion, compute SHA-256 digest
  4. `os.replace(part_path, dest_path)` (atomic)
  5. Return outcome with digest and content-length

- **Rate Limiting**: `ResolverPipeline._respect_rate_limit()` uses `min_interval_s` (renamed from `resolver_rate_limits`) to enforce minimum seconds between calls per resolver. ThreadPoolExecutor workers serialize through this bottleneck via shared `_last_invocation` dict with threading.Lock.

- **LRU Cache**: Decorate resolver API helpers with `@lru_cache(maxsize=1000)`, key=(resolver_name, normalized_identifier). Cache cleared between batch runs if `--resume-from` used.

- **Conditional Requests**: In `download_candidate`, before streaming:
  1. Read manifest for (work_id, url) to get previous ETag/Last-Modified
  2. Add If-None-Match/If-Modified-Since to request headers
  3. If 304 response, return DownloadOutcome(classification='cached', path=existing_path)
  4. Update manifest with new ETag/Last-Modified on success

- **Unified Logging**: `JsonlLogger` writes records with schema:

  ```json
  {
    "timestamp": "ISO8601",
    "record_type": "attempt"|"summary",
    "work_id": "W123",
    "resolver_name": "unpaywall",
    "resolver_order": 1,
    "url": "https://...",
    "status": "pdf"|"html"|"http_error"|"cached",
    "http_status": 200,
    "content_type": "application/pdf",
    "elapsed_ms": 1234.5,
    "sha256": "abc...",
    "content_length": 567890,
    "reason": null|"error details"
  }
  ```

  CSV export script: `jq -r '[.timestamp, .work_id, ...] | @csv' < attempts.jsonl > attempts.csv`

- **Parallel Execution**: Main loop:

  ```python
  with ThreadPoolExecutor(max_workers=args.workers) as executor:
      futures = {executor.submit(process_work, work, session_factory()): work for work in batch}
      for future in as_completed(futures):
          result = future.result()
          # aggregate metrics
  ```

  Each worker gets its own session (HTTPAdapter Retry is thread-safe), shares `ResolverPipeline._last_invocation` via Lock.

- **Refactored Download State Machine**:

  ```python
  class DownloadState(Enum):
      PENDING = "pending"
      WRITING = "writing"

  # In download_candidate:
  state = DownloadState.PENDING
  for chunk in response.iter_content(...):
      if state == PENDING:
          detected = classify_payload(sniff_buffer, ...)
          if detected:
              open file handle, state = WRITING
      elif state == WRITING:
          handle.write(chunk)
  return _build_outcome(detected, dest_path, ...)
  ```

- **Shared Utilities** (`ContentDownload/utils.py`):

  ```python
  def normalize_doi(doi: Optional[str]) -> Optional[str]: ...
  def normalize_pmcid(pmcid: Optional[str]) -> Optional[str]: ...
  def strip_prefix(value: Optional[str], prefix: str) -> Optional[str]: ...
  def dedupe(items: List[str]) -> List[str]: ...
  ```

  Imported by both main module and resolvers.

## Risks / Trade-offs

- Risk: Retry logic delays failures, increasing overall runtime. Mitigation: Retry capped at 5 attempts with exponential backoff (max ~30s per URL); fail-fast on 4xx errors; overall runtime increase <5% based on retry rate.

- Risk: ThreadPoolExecutor increases memory footprint. Mitigation: max_workers default=1 (opt-in); N=5 adds ~50MB overhead (5 sessions + buffers); document memory scaling.

- Risk: LRU cache stale on DOI updates. Mitigation: cache only within single run; cache cleared on --resume-from; DOI metadata changes rare; false negatives acceptable (just refetch).

- Risk: Breaking config change (`resolver_rate_limits` → `min_interval_s`) requires operator updates. Mitigation: detect old field name, log deprecation warning, auto-migrate value; document in migration guide; fail gracefully if both fields present.

- Risk: JSONL logging breaks existing CSV parsers. Mitigation: provide one-line jq/Python export script in migration guide; document JSONL schema; keep CSV export tested.

- Risk: Parallel workers overwhelm rate limits. Mitigation: shared `_last_invocation` dict with Lock enforces global per-resolver intervals; test with max_workers=10 and assert rate limit compliance.

- Risk: Conditional requests return stale PDFs. Mitigation: ETag/Last-Modified from authoritative sources (Unpaywall, Crossref) rarely change; operators can use --force to bypass conditional requests.

- Risk: Trafilatura dependency bloat. Mitigation: optional import with graceful fallback; --extract-html-text flag defaults to False; document as optional enhancement.

## Migration Plan

1. Implement shared utilities in ContentDownload/utils.py with unit tests for normalization/dedupe functions.
2. Add HTTPAdapter + Retry to session creation; verify with integration test that mocks 503 → 200 sequence.
3. Refactor download_candidate for atomic writes and state enum; run full test suite to ensure no regressions.
4. Add SHA-256 digest and content-length to DownloadOutcome and manifest schema.
5. Implement LRU cache decorators on resolver API helpers; measure cache hit rate in tests.
6. Add conditional request support with If-None-Match/If-Modified-Since; test 304 handling.
7. Rename `resolver_rate_limits` → `min_interval_s` with deprecation warning and auto-migration.
8. Unify logging to JSONL format; create CSV export script and document in README.
9. Extract duplicated code to utils.py; update imports in both modules.
10. Implement ThreadPoolExecutor parallelism with --workers flag; add threading.Lock to rate limiter.
11. Add --dry-run and --resume-from flags with corresponding logic.
12. Implement HTML text extraction with trafilatura (optional); add --extract-html-text flag.
13. Add OpenAIRE/HAL/OSF resolvers (disabled by default) following existing patterns.
14. Add tests for retry (429), corrupt HTML, Wayback unavailable, manifest alignment.
15. Update documentation: migration guide (config/logging changes), new CLI flags, rate limit units, parallelism tuning.
16. Run end-to-end smoke test: batch of 100 works, --workers 3, verify no regressions in success rate, manifest integrity, file atomicity.

## Open Questions

- Should we add Prometheus metrics export for production monitoring (success rates, latencies per resolver)?
- What's the preferred default for --workers in CI/production (1 for safety vs 3 for throughput)?
- Should conditional requests be opt-in (--use-etags) or opt-out (--ignore-etags)?
- Do we need a separate --max-retries flag to override HTTPAdapter default (5 attempts)?
- Should HTML text extraction use trafilatura, newspaper3k, or BeautifulSoup get_text()?
- Is there value in adding content-addressable storage (store by SHA-256) to enable natural deduplication?
