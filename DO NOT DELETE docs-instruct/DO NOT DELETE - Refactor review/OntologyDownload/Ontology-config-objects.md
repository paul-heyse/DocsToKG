Absolutely—here’s a **code-free, agent-ready implementation plan** for **PR 6 — `OntologyDownload/settings.py`**. It bridges **environment → Pydantic (v2) settings → typed config objects** that your existing exports and modules can consume without changing public call-sites.

---

# North-star (what “done” means)

* A **single `Settings` entry point** (Pydantic v2 + `pydantic-settings`) loads configuration from **env / .env / config file(s)** with strict typing and validation, then exposes **domain-specific sub-configs** (Network/HTTP, URL security, Rate Limits, Extraction Policy, Storage, DuckDB, Logging, Observability).
* **Public exports stay stable**: your `config builders` keep the same names but now return views of the new settings or the domain objects derived from them (no caller churn).
* **Precedence is explicit**: CLI flags (if any) > config file(s) > env vars > defaults.
* **Normalization is guaranteed**: helper methods return normalized allowlists, ports, include globs, parsed rate limits, etc., so downstream code never re-parses strings.
* **Immutable at runtime**: once built, settings are read-only (fail on accidental mutation).

---

# Scope & non-goals

**In**: settings module, environment variable mapping, validation/normalization, domain sub-models, compatibility shims in your existing `exports` (builders), tests & docs.

**Out**: no business logic changes in HTTP clients, downloaders, validators, or storage—only how they **obtain** their config.

---

# Module structure (files & responsibilities)

* `src/DocsToKG/OntologyDownload/settings.py`

  * `Settings` (top-level BaseSettings)
  * Domain sub-models:

    * `HttpSettings`, `CacheSettings`, `RetrySettings`, `RateLimitSettings`
    * `SecuritySettings` (URL/IDN/ports/private nets/plain HTTP flags)
    * `ExtractionSettings` (policy knobs: encapsulation, limits, formats, timestamps, hashing, IO buffers)
    * `StorageSettings` (local FS; still supports URL for future FSSPEC)
    * `DuckDBSettings` (local catalog DB)
    * `LoggingSettings` / `TelemetrySettings`
  * **Derived/normalized** helpers (no network I/O): `normalized_allowed_hosts()`, `allowed_port_set()`, `parse_service_rate_limit()`, `include_globs_compiled()`, `config_hash()`
  * Source ordering (`SettingsConfigDict` / `settings_customise_sources`)
* `src/DocsToKG/OntologyDownload/exports.py` (or where you currently export builders)

  * Keep public builder names; internally call `settings.get()` or `settings.build_*()`
* `src/DocsToKG/OntologyDownload/__init__.py`

  * Re-export unchanged symbols if you already do so

---

# Configuration domains & fields (authoritative list)

> All env vars use the **`ONTOFETCH_*`** prefix. Nested fields use **double underscores** (Pydantic convention), e.g., `ONTOFETCH_HTTP__TIMEOUT_CONNECT=5.0`.

## A) HTTP / Network

* `ONTOFETCH_HTTP__HTTP2` (bool; default **true**)
* `ONTOFETCH_HTTP__TIMEOUT_CONNECT` (float seconds; default **5.0**)
* `ONTOFETCH_HTTP__TIMEOUT_READ` (float; default **30.0**)
* `ONTOFETCH_HTTP__TIMEOUT_WRITE` (float; default **30.0**)
* `ONTOFETCH_HTTP__TIMEOUT_POOL` (float; default **5.0**)
* `ONTOFETCH_HTTP__POOL_MAX_CONNECTIONS` (int; default **64**)
* `ONTOFETCH_HTTP__POOL_KEEPALIVE_MAX` (int; default **20**)
* `ONTOFETCH_HTTP__KEEPALIVE_EXPIRY` (float secs; default **30.0**)
* `ONTOFETCH_HTTP__TRUST_ENV` (bool; default **true**)
* `ONTOFETCH_HTTP__USER_AGENT` (str; default `DocsToKG/OntoFetch (+https://…)`)

**Cache (Hishel)**

* `ONTOFETCH_CACHE__ENABLED` (bool; default **true**)
* `ONTOFETCH_CACHE__DIR` (path; default: `<root>/.cache/http`)
* `ONTOFETCH_CACHE__BYPASS` (bool; default **false**)

**Retry (lightweight)**

* `ONTOFETCH_RETRY__CONNECT_RETRIES` (int; default **2**)
* `ONTOFETCH_RETRY__BACKOFF_BASE` (float; default **0.1**)
* `ONTOFETCH_RETRY__BACKOFF_MAX` (float; default **2.0**)

## B) URL Security & DNS

* `ONTOFETCH_SECURITY__ALLOWED_HOSTS` (CSV; supports exact domains, `*.suffix`, IP literals)
* `ONTOFETCH_SECURITY__ALLOWED_PORTS` (CSV ints; default `80,443`)
* `ONTOFETCH_SECURITY__ALLOW_PRIVATE_NETWORKS_FOR_HOST_ALLOWLIST` (bool; default **false**)
* `ONTOFETCH_SECURITY__ALLOW_PLAIN_HTTP_FOR_HOST_ALLOWLIST` (bool; default **false**)
* `ONTOFETCH_SECURITY__STRICT_DNS` (bool; default **true**)

## C) Rate Limits

* `ONTOFETCH_RATELIMIT__DEFAULT` (string; e.g., `10/second`, default **null** = unlimited)
* `ONTOFETCH_RATELIMIT__PER_SERVICE` (JSON or CSV of `service:rate`, e.g., `ols:5/second;bioportal:2/second`)
* `ONTOFETCH_RATELIMIT__SHARED_DIR` (path for shared SQLite bucket; default **null**)
* `ONTOFETCH_RATELIMIT__ENGINE` (enum: `pyrate`; default **pyrate**)

## D) Extraction Policy (Safety + Throughput + Integrity; defaults are strict)

Safety

* `ONTOFETCH_EXTRACT__ENCAPSULATE` (bool; default **true**)
* `ONTOFETCH_EXTRACT__ENCAPSULATION_NAME` (`sha256`|`basename`; default **sha256**)
* Limits:

  * `ONTOFETCH_EXTRACT__MAX_DEPTH` (int; default **32**)
  * `ONTOFETCH_EXTRACT__MAX_COMPONENTS_LEN` (int bytes; default **240**)
  * `ONTOFETCH_EXTRACT__MAX_PATH_LEN` (int bytes; default **4096**)
  * `ONTOFETCH_EXTRACT__MAX_ENTRIES` (int; default **50000**)
  * `ONTOFETCH_EXTRACT__MAX_FILE_SIZE_BYTES` (int; default **2147483648**)
  * `ONTOFETCH_EXTRACT__MAX_TOTAL_RATIO` (float; default **10.0**)
  * `ONTOFETCH_EXTRACT__MAX_ENTRY_RATIO` (float; default **100.0**)
* Name normalization:

  * `ONTOFETCH_EXTRACT__UNICODE_FORM` (`NFC`|`NFD`; default **NFC**)
  * `ONTOFETCH_EXTRACT__CASEFOLD_COLLISION_POLICY` (`reject`|`allow`; default **reject**)
* Overwrites & duplicates:

  * `ONTOFETCH_EXTRACT__OVERWRITE` (`reject`|`replace`|`keep_existing`; default **reject**)
  * `ONTOFETCH_EXTRACT__DUPLICATE_POLICY` (`reject`|`first_wins`|`last_wins`; default **reject**)

Throughput

* `ONTOFETCH_EXTRACT__SPACE_SAFETY_MARGIN` (float; default **1.10**)
* `ONTOFETCH_EXTRACT__PREALLOCATE` (bool; default **true**)
* `ONTOFETCH_EXTRACT__COPY_BUFFER_MIN` (int bytes; default **65536**)
* `ONTOFETCH_EXTRACT__COPY_BUFFER_MAX` (int bytes; default **1048576**)
* `ONTOFETCH_EXTRACT__GROUP_FSYNC` (int; default **32**)
* `ONTOFETCH_EXTRACT__MAX_WALL_TIME_SECONDS` (int; default **120**)

Integrity

* `ONTOFETCH_EXTRACT__HASH_ENABLE` (bool; default **true**)
* `ONTOFETCH_EXTRACT__HASH_ALGORITHMS` (CSV; default `sha256`)
* `ONTOFETCH_EXTRACT__INCLUDE_GLOBS` (CSV; e.g., `*.ttl,*.rdf,*.owl,*.obo`; default empty = include all)
* `ONTOFETCH_EXTRACT__EXCLUDE_GLOBS` (CSV; default empty)
* `ONTOFETCH_EXTRACT__TIMESTAMPS_MODE` (`preserve`|`normalize`|`source_date_epoch`; default **preserve**)
* `ONTOFETCH_EXTRACT__TIMESTAMPS_NORMALIZE_TO` (`archive_mtime`|`now`; default **archive_mtime**)

## E) Storage (local only for now; keep URL as future-proof)

* `ONTOFETCH_STORAGE__ROOT` (path; default `<project_root>/ontologies`)
* `ONTOFETCH_STORAGE__LATEST_NAME` (str; default `LATEST.json`)
* *(Optional future)* `ONTOFETCH_STORAGE__URL` (s3://… etc.); if set, your factory can switch to FSSPEC backend but you’ve chosen local-only for now—keep the field for forward compatibility.

## F) DuckDB Catalog

* `ONTOFETCH_DB__PATH` (path; default `<root>/.catalog/ontofetch.duckdb`)
* `ONTOFETCH_DB__THREADS` (int; default `min(8, CPU)`)
* `ONTOFETCH_DB__READONLY` (bool; default **false**)
* `ONTOFETCH_DB__WLOCK` (bool; default **true**)  — enable writer file lock
* `ONTOFETCH_DB__PARQUET_EVENTS` (bool; default **false**)

## G) Logging / Telemetry

* `ONTOFETCH_LOG__LEVEL` (`DEBUG`|`INFO`|`WARN`|`ERROR`; default **INFO**)
* `ONTOFETCH_LOG__JSON` (bool; default **true**)
* `ONTOFETCH_TELEMETRY__RUN_ID` (UUID; default auto)
* `ONTOFETCH_TELEMETRY__EMIT_EVENTS` (bool; default **true**)

---

# Source precedence & loading order

Implement `settings_customise_sources`:

1. **CLI overlay** (dict passed in by your CLI layer; optional)
2. **Env file(s)**: `.env.ontofetch`, `.env`
3. **Environment variables** (`ONTOFETCH_*`)
4. **Config file** (optional path `ONTOFETCH_CONFIG` supporting TOML/YAML; if present, insert above env)
5. **Defaults** baked into models

> Document precedence in README. Always log **which sources** contributed (not values), and emit a **config hash** for provenance.

---

# Validation & normalization (must-haves)

* **CSV → lists/sets**:

  * `ALLOWED_HOSTS` → `(exact_domains, wildcard_suffixes, per_host_ports, ip_literals)`
  * `ALLOWED_PORTS` → `set[int]` (add 80/443 if empty)
  * `INCLUDE_GLOBS`, `EXCLUDE_GLOBS` → compiled patterns (store both raw + compiled)
* **Rate strings**: parse `N/second|minute|hour` → canonical `(limit, Duration)` and derived **RPS**; maintain map per service.
* **Paths**: normalize to absolute paths, POSIX semantics, create directories if required (`cache.dir`, `storage.root`, `db.path`’s parent).
* **Bounds**: enforce sensible min/max (e.g., buffer sizes, timeouts, margins, ratios).
* **Enum coercion**: case-insensitive parsing for modes and policies.
* **Derived properties**:

  * `normalized_allowed_hosts()`: returns 4-tuple for validator
  * `allowed_port_set()`: returns merged global/per-host sets
  * `include_filters()`: returns a callable `path -> bool` based on include/exclude
  * `config_hash()`: stable SHA-256 hash of the **normalized** dict (exclude secrets), used in run context

> All normalization happens **once** at construction; domain objects (`HttpSettings`, `ExtractionSettings`…) expose ready-to-use methods so consumers never re-parse strings.

---

# Backward compatibility & exports

* Keep existing public builder names, e.g., `build_http_config()`, `build_security_config()`, `build_extraction_policy()`, etc.

  * Internally: `s = settings.get()` then `return s.http` (or a converted legacy object if callers expect a specific type).
* If prior code reached into environment directly, **remove** those reads and route through `Settings`.

---

# Wiring: where each domain is used

* **Network client** (`net`/`HttpxClient`): `settings.http`, `settings.cache`, `settings.retry`, plus `settings.ratelimit` for bucket creation.
* **URL validator**: `settings.security.normalized_allowed_hosts()`, `allowed_port_set()`, `allow_private_networks_for_host_allowlist`, `allow_plain_http_for_host_allowlist`, `strict_dns`.
* **Downloader/Extraction**: `settings.extraction` (encapsulation root, limits, buffer sizes, hashing, timestamps).
* **Storage**: `settings.storage.root` (and later `storage.url` to switch to FSSPEC).
* **DuckDB**: `settings.db` (path, threads, readonly, writer lock).
* **Logging**: `settings.log` + `settings.telemetry`.

---

# Example env (minimal)

```
# HTTP
ONTOFETCH_HTTP__TIMEOUT_CONNECT=5
ONTOFETCH_HTTP__TIMEOUT_READ=30
ONTOFETCH_CACHE__DIR=/var/tmp/ontofetch/http-cache

# Security
ONTOFETCH_SECURITY__ALLOWED_HOSTS=ebi.ac.uk,*.purl.org,141.0.0.0/8,10.0.0.7
ONTOFETCH_SECURITY__ALLOW_PLAIN_HTTP_FOR_HOST_ALLOWLIST=false
ONTOFETCH_SECURITY__STRICT_DNS=true

# Rate limits
ONTOFETCH_RATELIMIT__DEFAULT=8/second
ONTOFETCH_RATELIMIT__PER_SERVICE=ols:4/second;bioportal:2/second

# Extraction
ONTOFETCH_EXTRACT__ENCAPSULATE=true
ONTOFETCH_EXTRACT__INCLUDE_GLOBS=*.ttl,*.rdf,*.owl,*.obo
ONTOFETCH_EXTRACT__MAX_TOTAL_RATIO=10.0

# Storage & DB
ONTOFETCH_STORAGE__ROOT=/data/ontologies
ONTOFETCH_DB__PATH=/data/ontologies/.catalog/ontofetch.duckdb
```

---

# Test plan (unit + integration)

## Unit (settings only)

* **Defaults** load correctly; derived values match expectations (ports set contains 80,443; include filter returns True for `a.ttl`).
* **Env mapping**: set env vars; assert all fields updated; nested `__` mapping works.
* **Validation errors**: invalid rate strings, negative limits, malformed hosts → raise with precise messages.
* **Normalization**: wildcard suffixes parsed; IP literals recognized; paths normalized; enums case-insensitive.
* **Config hash**: stable across order changes; changes when any underlying normalized value changes.

## Integration (lightweight)

* **Network client** picks up timeouts/pool sizes and sets user-agent.
* **URL validator** enforces allowlist/ports/plain HTTP flags.
* **Downloader/Extractor** sees include globs, limits, buffer sizes.
* **DuckDB** opens with configured threads; writer lock engaged when WLOCK=true.
* **Logging** switches to JSON when configured; run_id propagated into event payloads.

---

# Roll-in sequence (small PR steps)

1. Add `settings.py` with all domain models, normalization, and sources.
2. Add a **singleton accessor** (internally cached) so modules call `settings.get()` without expensive re-parsing.
3. Update `exports` to return objects from `settings`.
4. Replace any direct `os.environ[...]` reads with the typed settings.
5. Add tests (unit first, then integration bindings).
6. Update README: env var matrix, precedence, and examples.

---

# Acceptance checklist

* [ ] `Settings` loads from env/.env/config with the precedence described; sensitive values never logged.
* [ ] All domain sub-models validated and normalized on construction; consumers never parse raw strings.
* [ ] Public builder functions **unchanged** in name/signature, now backed by `Settings`.
* [ ] URL validator works solely off `SecuritySettings` helpers (`normalized_allowed_hosts()`, `allowed_port_set()`); no duplicate parsing elsewhere.
* [ ] Downloader/extractor obtains **one** `ExtractionSettings` object; limits & policies applied consistently.
* [ ] DuckDB & Storage read settings once at init; writer lock honored when enabled.
* [ ] Test suite covers defaults, env mapping, validation failures, derived helpers, and a basic end-to-end settings flow.
* [ ] Docs updated with **env var matrix**, precedence diagram, and a minimal `.env` sample.

---

If you want, I can produce a **field-by-field env matrix** (CSV/Markdown table) and a **settings reference page** you can drop into your repo docs.
