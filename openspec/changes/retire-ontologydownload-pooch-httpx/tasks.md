## 1. Rebuild `download_stream` around HTTPX streaming (`src/DocsToKG/OntologyDownload/io/network.py`)
- [ ] 1.1 Delete the `StreamingDownloader` subclass and replace it with lightweight helper functions (e.g., `_stream_httpx_response`, `_log_progress`) that operate on `httpx.Response` iterators.
- [ ] 1.2 Refactor `download_stream` to:
  - Acquire the shared client via `get_http_client()`.
  - Issue a single GET (`client.build_request` + `client.send`) with polite headers and conditional validators from the manifest.
  - On 304, materialise the cached artifact (using `_materialize_cached_file`) and short-circuit with `DownloadResult(status="cached")`.
  - On 200/206, stream chunks to a temp file, enforcing `max_uncompressed_size_gb` and `CancellationToken` checks inside the loop.
  - Emit progress telemetry every N bytes or percentage (use logger `info`/`debug` with `event="download_progress"`).
  - After streaming, compute hashes via `_compute_file_hash` / `sha256_file`, verify against `expected_hash`, and populate manifest fields from the response headers.
- [ ] 1.3 Fold HEAD logic into the GET flow: only trigger a preflight HEAD when `http_config` explicitly demands it (e.g., `http_config.perform_head_precheck`), and ensure HEAD responses re-use redirect auditing.
- [ ] 1.4 Update error handling to catch `httpx.HTTPStatusError` / `httpx.TransportError` and translate them into `DownloadFailure` with the appropriate `retryable` flag; map size violations to `PolicyError`.
- [ ] 1.5 Remove pooch usage entirely (`pooch.retrieve`, `pooch.HTTPDownloader`, `known_hash`) and adjust imports accordingly.

## 2. Remove legacy downloader artifacts & dependencies
- [ ] 2.1 Delete `SessionPool` references that remain after Phase 2 clean-up (class definition, globals, tests) and ensure the module docstring/NAVMAP reflects the new architecture.
- [ ] 2.2 Drop the `pooch` dependency from `pyproject.toml`, `optdeps.py`, and any environment docs; add guards to fail fast if downstream code still attempts to import the legacy downloader.
- [ ] 2.3 Update `io/__init__.py` exports to remove `SessionPool`/`SESSION_POOL`/`StreamingDownloader`; ensure `__all__` now lists `get_http_client`, `configure_http_client`, and `reset_http_client`.
- [ ] 2.4 Search the codebase (`rg "StreamingDownloader"`, `rg "pooch"`) and remove dead imports, docs, and comments referencing the old implementation.

## 3. Consolidate planner & checksum probes on HTTPX (`src/DocsToKG/OntologyDownload/planning.py`, `checksums.py`)
- [ ] 3.1 Update `planner_http_probe` to default to a single GET; ensure the helper reuses the GET response to populate `_populate_plan_metadata` without issuing a second request.
- [ ] 3.2 When HEAD is explicitly required (e.g., host opt-in), use `client.build_request("HEAD", ...)` and document the allow-list; avoid unconditional HEAD+GET pairs.
- [ ] 3.3 Ensure planner probes interpret cache metadata from `response.headers` / `response.extensions["hishel"]`, updating plan records accordingly.
- [ ] 3.4 Refactor `_fetch_checksum_from_url` to stream with `client.stream`, honour byte ceilings, and apply retry/backoff on `httpx` exceptions; remove any `requests` imports.
- [ ] 3.5 Verify ancillary helpers (e.g., validation probes, CLI dry-run) call the shared client once per probe and reuse metadata without re-contacting the origin.

## 4. Update configuration & rollout plumbing
- [ ] 4.1 Add configuration knobs (if absent) for progress logging cadence (bytes or percentage thresholds) and explicit HEAD opt-in; default them to the current behaviour.
- [ ] 4.2 Provide migration guidance in `settings.py` / `DownloadConfiguration` docstrings describing the deprecation of `SessionPool` and the new knobs.
- [ ] 4.3 Ensure any feature flags used during Phase 1–2 (e.g., `network.engine`) default to the HTTPX path and document removal timelines if applicable.

## 5. Testing & fixtures (`tests/ontology_download/**`, `DocsToKG/OntologyDownload/testing`)
- [ ] 5.1 Replace harness fixtures that relied on `requests` or pooch with `httpx.MockTransport` implementations capable of:
  - Serving deterministic payloads with headers for media-type validation.
  - Emitting 304 responses when `If-None-Match` / `If-Modified-Since` are present.
  - Simulating redirects, retryable failures, and size overflows.
- [ ] 5.2 Add targeted unit tests for:
  - Streaming progression (progress logs fired, bytes counted).
  - Policy violations (size cap exceeded → `PolicyError`).
  - Cached downloads returning `status="cached"` and preserving metadata.
  - Planner probe falling back to GET-only workflow.
- [ ] 5.3 Update CLI regression tests to ensure JSON/table outputs still surface `content_type`, `content_length`, `etag`, `status`, and `cache_status`.
- [ ] 5.4 Extend test utilities with a `use_mock_http_client` context manager that swaps the shared client for tests and resets it afterwards.
- [ ] 5.5 Remove or rewrite tests that assert behaviour specific to `SessionPool` internals (e.g., pool size assertions) and replace them with HTTPX-centric expectations (transport call counts, cached flag).

## 6. Documentation & developer guidance
- [ ] 6.1 Update `src/DocsToKG/OntologyDownload/README.md` to describe the new streaming pipeline, including Hishel cache directory layout, progress telemetry, and how to configure HEAD opt-ins.
- [ ] 6.2 Revise `src/DocsToKG/OntologyDownload/AGENTS.md` to instruct agents on using `httpx.MockTransport`, the new helper API, and the lack of `SessionPool`.
- [ ] 6.3 Refresh library briefs (`LibraryDocumentation/httpx.md`, `LibraryDocumentation/hishel.md`, and the transition plan) with implementation notes about the completed migration.
- [ ] 6.4 Add release notes/change log entries summarising removal of pooch and impact on downstream extensions.

## 7. Follow-ups & validation
- [ ] 7.1 Run `pytest tests/ontology_download -q` and document the passing build, including any new fixtures.
- [ ] 7.2 Execute smoke runs of the OntologyDownload CLI (`plan`, `pull --dry-run`, full download) against representative fixtures to confirm parity with prior manifests.
- [ ] 7.3 Capture telemetry snapshots (cache hit/miss counters, progress logs) before/after migration to verify no regressions.
- [ ] 7.4 File follow-up tickets (if needed) for optional Phase 7 items (rate-limit library swap, retry policy upgrades) once the HTTPX streaming path is stable.
