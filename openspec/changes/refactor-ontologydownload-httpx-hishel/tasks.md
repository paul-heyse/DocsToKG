## 1. Establish HTTPX + Hishel transport layer (`src/DocsToKG/OntologyDownload/net.py`)
- [x] 1.1 Create a new module `net.py` containing:
  - Module-level constants for `HTTP_CACHE_DIR = CACHE_DIR / "http"` and `_CLIENT_LOCK = threading.RLock()`.
  - Private slot `_HTTP_CLIENT: httpx.Client | None = None`.
  - Public helpers `get_http_client()`, `configure_http_client(client: httpx.Client | None = None, *, factory: Callable[[], httpx.Client | None] | None = None)`, and `reset_http_client()` that all synchronise on `_CLIENT_LOCK`.
- [x] 1.2 Implement `_build_http_client(cache_root: Path, config: DownloadConfiguration | None) -> httpx.Client` with explicit settings:
  - Use `httpx.Timeout(connect=config.connect_timeout_sec, read=config.timeout_sec, write=config.timeout_sec, pool=config.pool_timeout_sec)`; fall back to sane defaults (e.g., 5/30 seconds) when fields are absent.
  - Use `httpx.Limits(max_connections=config.max_httpx_connections, max_keepalive_connections=config.max_keepalive_connections, keepalive_expiry=config.keepalive_expiry_sec)`.
  - Instantiate `ssl.create_default_context(cafile=certifi.where())` and pass as `verify=ssl_context`.
  - Set `http2=config.http2_enabled`, `trust_env=True`, `follow_redirects=False`.
- [x] 1.3 Construct a Hishel cache transport:
  - Import `hishel.CacheOptions` and `hishel.CacheTransport`.
  - Ensure `cache_root` exists (`mkdir(parents=True, exist_ok=True)`).
  - Configure cache options with `cacheable_methods={"GET", "HEAD"}`, `cacheable_status_codes={200, 203, 300, 301, 308, 404, 410, 416}`, `allow_incomplete=False`, `heuristic=None`, `respect_cache_headers=True`, and `allow_mutable_headers={"Date"}`.
  - Wrap the base transport using `CacheTransport(base_transport=httpx.HTTPTransport(retries=0), cache_options=options, storage=hishel.FileStorage(cache_root))` and pass it to `httpx.Client(transport=CacheTransport(...))`.
- [x] 1.4 Register event hooks on the client:
  - `event_hooks["request"]` should merge polite headers. Allow callers to place override headers in `request.extensions["ontology_headers"]` and reapply `DownloadConfiguration.polite_http_headers(correlation_id=...)`.
  - `event_hooks["response"]` should call `response.raise_for_status()`, log cache metadata (`response.extensions.get("hishel", {})`), and attach `response.extensions["ontology_cache_status"]` for downstream consumers.
- [x] 1.5 Update `configure_http_client` logic:
  - Accept either a concrete `httpx.Client` or a `factory` callable. Only one may be provided; raise `ValueError` when both are set.
  - If neither is provided, rebuild using `_build_http_client`.
  - Persist the installed factory on `DownloadConfiguration` so `DownloadConfiguration.get_session_factory()` invokes it and returns the HTTPX client; ensure legacy factories returning `requests.Session` raise a descriptive error.
- [x] 1.6 Re-export `get_http_client`, `configure_http_client`, and `reset_http_client` from `DocsToKG.OntologyDownload.io.__init__` (and optionally the package `__init__`) so existing imports adopt the new API.

## 2. Migrate networking helpers off `SessionPool` (`io/network.py`)
- [x] 2.1 Delete the `SessionPool` class definition, the module-level `SESSION_POOL`, and any references to `requests`. Replace them with `from ..net import get_http_client` and `import httpx`.
- [x] 2.2 Extend `is_retryable_error` to treat `httpx.ConnectError`, `httpx.ReadTimeout`, `httpx.WriteTimeout`, `httpx.PoolTimeout`, `httpx.HTTPStatusError`, and generic `httpx.TransportError` as retryable according to status code (`exc.response.status_code`) and to surface `DownloadFailure.retryable` unchanged.
- [x] 2.3 Rewrite `request_with_redirect_audit`:
  - Accept an `httpx.Client` parameter rather than a `requests.Session`.
  - Build requests with `client.build_request(method, url, headers=headers, timeout=httpx.Timeout(...), follow_redirects=False, stream=stream)`.
  - Loop calling `client.send(request, stream=stream)` while manually validating `response.headers["Location"]` for redirects using `validate_url_security`.
  - Ensure the function closes response objects (`response.close()`) in every branch and continues to attach `validated_url`.
- [x] 2.4 Update `StreamingDownloader`:
  - Replace `session` usage with `http_client = get_http_client()` (or injected client for tests).
  - Use `request_with_redirect_audit(client=http_client, ...)` for HEAD and GET requests.
  - Convert chunks using `response.iter_bytes()`; maintain rate-limit (`bucket.consume()`), resume headers, `expected_media_type`, checksum digests, and cancellation checks.
  - Keep `pooch.retrieve` integration but pass a downloader that internally uses the HTTPX client; ensure no code path instantiates `requests.Session`.
- [x] 2.5 Modify `download_stream`:
  - Acquire the HTTPX client once per retry loop (`http_client = get_http_client()`).
  - Supply per-call polite headers via `ontology_headers` extension so hooks inject them.
  - Remove references to `SESSION_POOL`, `requests`, and adapt exception handling to catch `httpx` errors.
  - Preserve manifest-driven `status` logic and cached-path materialisation.
- [x] 2.6 Remove `SessionPool` from `io/__all__`, ensure NAVMAP docs highlight the HTTPX client, and add import/export of `get_http_client`, `configure_http_client`, `reset_http_client`.

## 3. Update dependent modules to use the shared client
- [x] 3.1 In `checksums._fetch_checksum_from_url`:
  - Replace `with SESSION_POOL.lease(...): session.get(...)` with `client = get_http_client()` and `with client.stream("GET", secure_url, headers=headers, extensions={"ontology_headers": polite_headers}, timeout=timeout) as response:`.
  - Adjust retry helpers so `_parse_retry_after` reads `response.headers.get("Retry-After")` from `httpx.Response`.
  - Catch `httpx.HTTPError` subclasses alongside `DownloadFailure`.
- [x] 3.2 In `planning.planner_http_probe`:
  - Fetch the HTTPX client once (`client = get_http_client()`).
  - Build requests via `client.build_request(...)`, pass polite headers through `ontology_headers`, and call `client.send()`.
  - Replace direct status/headers access with `response.status_code` / `response.headers`.
  - Ensure retry/backoff uses `httpx.HTTPStatusError` and `response.next_request` is not relied on (redirects handled in audit helper).
- [ ] 3.3 Audit the entire package (`rg "SESSION_POOL"`) and migrate remaining references, including `checksums`, `planning`, any CLI utilities, documentation strings, and tests. Remove stale imports and comments referencing `SessionPool`.
- [x] 3.4 Update `DocsToKG.OntologyDownload.testing.reset_state()` (and similar fixtures) to call `reset_http_client()` so each test gets a fresh client/transport/cache state.

## 4. Configuration & injection bridges
- [x] 4.1 Extend `DownloadConfiguration` with HTTPX tuning knobs:
  - Add optional fields `connect_timeout_sec: float = 5.0`, `pool_timeout_sec: float = 5.0`, `max_httpx_connections: int = 100`, `max_keepalive_connections: int = 20`, `keepalive_expiry_sec: float = 30.0`, and `http2_enabled: bool = True`.
  - Document each field in the model docstring and ensure `model_copy` preserves them.
- [x] 4.2 Rename/repurpose `DownloadConfiguration.set_session_factory` to clarify that the factory must return an `httpx.Client`. Validate that any legacy factory returning a `requests.Session` raises a `TypeError` with guidance.
- [x] 4.3 Ensure `DownloadConfiguration.get_session_factory()` is used by `get_http_client()`: when present, call it once, memoise the resulting client, invoke `configure_http_client(client=...)`, and log with module logger that a custom client is active (include module/name for observability).
- [x] 4.4 Add a convenience helper in `DocsToKG.OntologyDownload.testing`:
  - `@contextmanager def use_mock_http_client(transport: httpx.MockTransport, **client_kwargs): ...` that installs a temporary client via `configure_http_client(client=httpx.Client(transport=transport, **client_kwargs))` and restores the prior client on exit.

## 5. Tests & QA
- [x] 5.1 Update all tests referencing `SESSION_POOL`:
  - `tests/ontology_download/test_download_behaviour.py` should assert `net.get_http_client()` reuse by inspecting transport call counts instead of pool size.
  - Replace `requests` monkeypatches with `httpx.MockTransport` that returns deterministic responses (including redirect and 304 cases).
- [ ] 5.2 Add focused unit tests for `net.py` covering:
  - Client singleton behaviour (`configure_http_client` swap + restore).
  - Timeouts/limits derived from `DownloadConfiguration`.
  - Polite header injection via `request.extensions`.
  - Cache directory creation and that repeated GET triggers a Hishel cache hit.
- [ ] 5.3 Extend download behaviour tests to include:
  - A scenario where the manifest contains `etag`/`last_modified` and the server returns 304, asserting `DownloadResult.status == "cached"` and no new bytes are written.
  - A scenario using redirects to confirm `request_with_redirect_audit` blocks disallowed redirects and only follows validated targets.
- [ ] 5.4 Update retry/backoff tests so `is_retryable_error` responds to `httpx` exceptions (connect/read/timeouts) and that planner probes still propagate `Retry-After` delays into the token bucket registry.
- [ ] 5.5 Execute `pytest tests/ontology_download -q` and capture results in the rollout notes; document any new fixtures (`use_mock_http_client`) required by test modules.

## 6. Documentation & developer guidance
- [ ] 6.1 Refresh `src/DocsToKG/OntologyDownload/README.md` and `AGENTS.md`:
  - Replace mentions of `SessionPool` with references to `DocsToKG.OntologyDownload.net`.
  - Document how to configure custom transports and how Hishel manages the cache directory.
- [ ] 6.2 Update library briefs (`LibraryDocumentation/httpx.md`, `LibraryDocumentation/hishel.md`) to include a “DocsToKG integration” section describing the new helper APIs, cache layout, and redirect policy.
- [ ] 6.3 Review NAVMAP headers and docstrings in `io/network.py` and `net.py` to ensure they explain the HTTPX + Hishel architecture, explicitly noting that `requests` is no longer part of the stack.
