## 1. Transport foundation
- [x] 1.1 Add `httpx[http2]` and `hishel` to every ContentDownload dependency manifest (`pyproject.toml`, `requirements.txt`, `requirements.gpu.txt`) with pinned versions; update `openspec/AGENTS.md` to reaffirm the no-install guard rails for the new deps.
- [x] 1.2 Create `src/DocsToKG/ContentDownload/httpx_transport.py` exporting `get_http_client()`, `configure_http_client(...)`, and `reset_http_client_for_tests()`. The singleton client MUST:
  - instantiate `httpx.Client(http2=True, limits=httpx.Limits(max_connections=128, max_keepalive_connections=32, keepalive_expiry=15.0), timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0), trust_env=True)`;
  - load an `ssl.SSLContext` rooted in Certifi, respecting optional overrides from `DownloadOptions` / environment variables;
  - accept per-call overrides in `configure_http_client(proxy_mounts=None, transport=None, event_hooks=None)`.
- [x] 1.3 Wrap the client with `hishel.CacheControlTransport`, storing entries under `${CACHE_DIR}/http/ContentDownload` with a shadow index for telemetry. Provide `purge_http_cache()` for tests/ops. Ensure cache keys use fully normalised URLs (leveraging existing `normalize_url` helper).
- [x] 1.4 Install HTTPX event hooks (`httpx.EventHook`) that:
  - add structured telemetry fields (`network.client=httpx`, cache metadata, attempt number) before/after each request;
  - call `response.raise_for_status()` once per response (while still allowing Tenacity to return final non-2xx responses when retries exhaust);
  - close intermediate responses when Tenacity schedules additional attempts.

## 2. Networking module migration
- [x] 2.1 Delete `ThreadLocalSessionFactory`, `create_session`, and `TENACITY_SLEEP` exposure of `time.sleep` in `src/DocsToKG/ContentDownload/networking.py`. Replace internal helpers with imports from `httpx_transport` (`get_http_client`, `purge_http_cache`); keep the public module exports intact.
- [x] 2.2 Update `request_with_retries` so the Tenacity predicates handle `httpx.TimeoutException`, `httpx.TransportError`, and `httpx.ProtocolError`, and the result predicate accepts `httpx.Response`. Ensure `_close_response_safely` accepts HTTPX responses and that elapsed sleep tracking continues to surface in logging.
- [x] 2.3 Refactor `head_precheck`, `_head_precheck_via_get`, and `RobotsCache._fetch` to call `httpx` via `request_with_retries`, passing explicit timeout/backoff parameters and ensuring 304/conditional logic is honoured through Hishel. Remove any `.close()` calls that assumed `requests.Response`.
- [x] 2.4 Update `stream_candidate_payload` and related downloader helpers to use `with httpx_client.stream("GET", …)` writing into the existing temp-file flow. Confirm progress callbacks, range/resume guards, and telemetry behave as before.
- [x] 2.5 Remove residual imports of `requests` in ContentDownload modules; add defensive shims so legacy tests trying to patch `create_session` raise a clear `RuntimeError` pointing to `httpx_transport`.

## 3. Call-site and test updates
- [x] 3.1 Update resolver modules (`src/DocsToKG/ContentDownload/resolvers/*.py`) so every HTTP operation routes through `request_with_retries` or, where streaming is required, through the new HTTPX streaming helper. Verify JSON decoding uses `response.json()` from HTTPX and closes responses via context managers.
- [x] 3.2 Adjust `download.py`, `pipeline.py`, and `tests/content_download` fixtures to obtain the HTTPX client for robots cache, head pre-check, and streaming paths. Ensure conditional 304 results still produce `CachedResult`/`ModifiedResult`.
- [x] 3.3 Replace `requests` mocks in tests with `httpx.MockTransport`. Provide fixtures in `tests/conftest.py` to override `get_http_client()` with a deterministic transport; update property-based and benchmark tests to patch the new wait/sleep hooks (`httpx_transport.configure_http_client` or Tenacity’s wait strategy) instead of `networking.random.uniform`.
- [x] 3.4 Add regression tests covering: (a) cache hits (304 → `CachedResult`), (b) retry-after driven waits on `httpx.Response`, (c) streaming downloads preserving atomic writes, and (d) robots cache reusing the shared client.

## 4. Documentation and rollout
- [x] 4.1 Revise `src/DocsToKG/ContentDownload/README.md`, `AGENTS.md`, and the library docs (`httpx.md`, `hishel.md`, transition plan) to explain the HTTPX transport, cache directory, dependency pins, configuration overrides, and testing approach.
- [x] 4.2 Update troubleshooting guides to describe how to purge the HTTP cache, diagnose Hishel cache collisions, and interpret new telemetry fields. Note the deprecation of `create_session` and the migration timetable.
- [x] 4.3 Coordinate release notes and the changelog: flag the HTTPX/Hishel introduction, mention dependency updates, advise operators to clear legacy session caches, and document any observable retry/caching behaviour changes.
