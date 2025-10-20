## ADDED Requirements
### Requirement: Pyrate-Limiter Manager
`DocsToKG.OntologyDownload.io.rate_limit` SHALL wrap `pyrate_limiter.Limiter` objects behind the existing `get_bucket(...).consume()` contract. The manager SHALL:
- derive limiter keys as `f"{(service or '_').lower()}:{(host or 'default').lower()}"`;
- build one or more `Rate` objects from the configuration string (mapping `/second`, `/minute`, `/hour` onto `Duration.SECOND`, `Duration.MINUTE`, `Duration.HOUR`) and validate the list before constructing the limiter;
- call `Limiter.try_acquire(name=key, weight=max(1, ceil(tokens)), raise_when_fail=False, max_delay=None)` and block until the limiter admits the call;
- cache limiters and rebuild them if the effective rate changes or if the backing bucket path differs from the cached entry;
- expose `reset()` that drops all cached limiters for tests and diagnostics.

#### Scenario: Acquire limiter for host without service override
- **WHEN** `DownloadConfiguration.rate_limiter="pyrate"` and `get_bucket(http_config=cfg, service=None, host="purl.obolibrary.org")` is called twice within one second
- **THEN** the returned bucket SHALL reuse a cached limiter keyed `_:purl.obolibrary.org`
- **AND** the second `.consume()` SHALL wait until one full token is available before returning.

#### Scenario: Resolver shares limiter with downloader
- **WHEN** the resolver pipeline calls `get_bucket(http_config=cfg, service="ols", host="www.ebi.ac.uk")`
- **THEN** the same limiter instance SHALL be reused later when `io.network` downloads from that host with `service="ols"`
- **AND** consuming tokens in one location SHALL throttle the other.

#### Scenario: Service override rebuilds limiter
- **WHEN** `cfg.rate_limits["bioportal"]` changes from `"5/second"` to `"1/second"` and `reset()` has not been called
- **THEN** the next `get_bucket` call for `service="bioportal"` SHALL detect the new rate, rebuild the limiter, and enforce the tighter quota.

### Requirement: Shared Rate Limit Persistence
When `DownloadConfiguration.shared_rate_limit_dir` is set and the mode is `"pyrate"`, the limiter manager SHALL persist counters in `pyrate_limiter.SQLiteBucket` stored at `<shared_rate_limit_dir>/ratelimit.sqlite`, reusing a single bucket instance across all limiters and creating the directory if it does not exist.

#### Scenario: Shared limiter creates SQLite backing file
- **WHEN** two worker processes call `get_bucket(http_config=cfg, service="obo", host="purl.obolibrary.org")` with `cfg.shared_rate_limit_dir="/tmp/rates"`
- **THEN** `/tmp/rates/ratelimit.sqlite` SHALL be created on the first call
- **AND** the second process SHALL observe throttling caused by tokens consumed in the first process.

### Requirement: Retry-After Alignment
`apply_retry_after` SHALL parse the `Retry-After` header value, return the numeric delay (seconds), and leave limiter state untouched. Callers in `io.network`, `planning`, `resolvers`, and `checksums` SHALL sleep/back off for that duration before invoking `.consume()` again.

#### Scenario: 429 response delays the next acquire
- **WHEN** `_download_once` receives a `429` response with header `Retry-After: 3`
- **THEN** `_apply_retry_after_from_response` SHALL return `3.0`
- **AND** the retry loop SHALL wait roughly three seconds before calling `bucket.consume()` again.

#### Scenario: Resolver honours Retry-After
- **WHEN** a resolver HTTP call raises `httpx.HTTPStatusError` for status 503 with `Retry-After: 5`
- **THEN** the resolver retry helper SHALL pause for roughly five seconds before the next attempt
- **AND** the limiter manager SHALL not mutate internal token counts during that pause.

### Requirement: Rate Limiter Mode and Observability
`DownloadConfiguration` SHALL expose `rate_limiter` with allowed values `"pyrate"` (default) and `"legacy"`. The CLI doctor output and telemetry SHALL include the active mode, and the first limiter creation in each process SHALL log the selected backend and bucket type.

#### Scenario: Doctor command reports limiter mode
- **WHEN** `./.venv/bin/python -m DocsToKG.OntologyDownload.cli doctor --json` runs with defaults
- **THEN** the JSON output SHALL contain `"rate_limiter": "pyrate"` inside the HTTP diagnostics.

#### Scenario: Legacy mode warns and bypasses pyrate
- **WHEN** `cfg.rate_limiter="legacy"` and `get_bucket(...)` is invoked
- **THEN** the legacy token-bucket implementation SHALL be returned
- **AND** a single warning SHALL indicate that legacy mode bypasses pyrate-limiter.

### Requirement: Custom Provider Compatibility
Custom providers returned by `DownloadConfiguration.get_bucket_provider()` SHALL continue to override `get_bucket`. When a provider is present, the pyrate manager SHALL not intercept the call, preserving downstream extensions.

#### Scenario: Custom provider short-circuits limiter
- **WHEN** `get_bucket_provider()` returns a callable that yields a stub bucket
- **THEN** `get_bucket(...)` SHALL return that stub without constructing a pyrate limiter
- **AND** `reset()` SHALL leave custom-provider instances untouched.
