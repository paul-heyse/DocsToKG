awesome — here’s a **complete, field-by-field ENV matrix** followed by a **Settings Reference page** you can drop into your repo. It matches the PR-6 plan (Pydantic v2 + `pydantic-settings`) and is organized by domain so an agent can wire it directly.

---

# ENV Matrix (field-by-field)

> Convention: nested settings use **double underscores** (`ONTOFETCH_HTTP__TIMEOUT_READ=30`).
> Types are validated at load; invalid values produce clear errors with the env key that failed.
> All timestamps are UTC. All sizes are **bytes**. All paths are **absolute POSIX** (normalized on load).

## A) HTTP / Network (HTTPX + Hishel)

| Env Var                                | Type      |              Default | Allowed / Format | Description                         | Used by                            |                |
| -------------------------------------- | --------- | -------------------: | ---------------- | ----------------------------------- | ---------------------------------- | -------------- |
| `ONTOFETCH_HTTP__HTTP2`                | bool      |               `true` | `true            | false`                              | Enable HTTP/2 for HTTPX client.    | Network client |
| `ONTOFETCH_HTTP__TIMEOUT_CONNECT`      | float (s) |                `5.0` | `>0`             | Connect timeout per attempt.        | Network client                     |                |
| `ONTOFETCH_HTTP__TIMEOUT_READ`         | float (s) |               `30.0` | `>0`             | Read timeout.                       | Network client                     |                |
| `ONTOFETCH_HTTP__TIMEOUT_WRITE`        | float (s) |               `30.0` | `>0`             | Write timeout.                      | Network client                     |                |
| `ONTOFETCH_HTTP__TIMEOUT_POOL`         | float (s) |                `5.0` | `>0`             | Acquire-from-pool timeout.          | Network client                     |                |
| `ONTOFETCH_HTTP__POOL_MAX_CONNECTIONS` | int       |                 `64` | `>=1`            | Max concurrent connections.         | Network client                     |                |
| `ONTOFETCH_HTTP__POOL_KEEPALIVE_MAX`   | int       |                 `20` | `>=0`            | Keepalive pool size.                | Network client                     |                |
| `ONTOFETCH_HTTP__KEEPALIVE_EXPIRY`     | float (s) |               `30.0` | `>=0`            | Idle connection expiry.             | Network client                     |                |
| `ONTOFETCH_HTTP__TRUST_ENV`            | bool      |               `true` | `true            | false`                              | Honor `HTTP(S)_PROXY`, `NO_PROXY`. | Network client |
| `ONTOFETCH_HTTP__USER_AGENT`           | str       | `DocsToKG/OntoFetch` | any              | Stable UA string (add contact URL). | Network client                     |                |
| `ONTOFETCH_CACHE__ENABLED`             | bool      |               `true` | `true            | false`                              | Enable Hishel RFC-9111 cache.      | Cache layer    |
| `ONTOFETCH_CACHE__DIR`                 | path      | `<root>/.cache/http` | abs path         | Cache directory.                    | Cache layer                        |                |
| `ONTOFETCH_CACHE__BYPASS`              | bool      |              `false` | `true            | false`                              | Force bypass (no revalidation).    | Cache layer    |
| `ONTOFETCH_RETRY__CONNECT_RETRIES`     | int       |                  `2` | `>=0`            | Retries for **connect** errors.     | Network client                     |                |
| `ONTOFETCH_RETRY__BACKOFF_BASE`        | float (s) |                `0.1` | `>=0`            | Backoff start.                      | Network client                     |                |
| `ONTOFETCH_RETRY__BACKOFF_MAX`         | float (s) |                `2.0` | `>=0`            | Backoff cap.                        | Network client                     |                |

## B) URL Security & DNS

| Env Var                                                         | Type     |   Default | Allowed / Format                                       | Description                                                   | Used by                                                    |               |
| --------------------------------------------------------------- | -------- | --------: | ------------------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------- | ------------- |
| `ONTOFETCH_SECURITY__ALLOWED_HOSTS`                             | CSV      | *(empty)* | `host`, `*.suffix`, `ip`, `ip/cidr`, optional `:ports` | Registrable allowlist + IP literals; supports per-host ports. | URL validator                                              |               |
| `ONTOFETCH_SECURITY__ALLOWED_PORTS`                             | CSV<int> |  `80,443` | ints                                                   | Global allowed ports (merged with per-host ports).            | URL validator                                              |               |
| `ONTOFETCH_SECURITY__ALLOW_PRIVATE_NETWORKS_FOR_HOST_ALLOWLIST` | bool     |   `false` | `true                                                  | false`                                                        | Permit private/loopback **if allowlisted**.                | URL validator |
| `ONTOFETCH_SECURITY__ALLOW_PLAIN_HTTP_FOR_HOST_ALLOWLIST`       | bool     |   `false` | `true                                                  | false`                                                        | Allow `http://` **if allowlisted**; else upgrade to HTTPS. | URL validator |
| `ONTOFETCH_SECURITY__STRICT_DNS`                                | bool     |    `true` | `true                                                  | false`                                                        | Fail if DNS cannot resolve.                                | URL validator |

**Examples**
`ONTOFETCH_SECURITY__ALLOWED_HOSTS="ebi.ac.uk,*.purl.org,10.0.0.7,141.0.0.0/8,example.org:8443"`

## C) Rate Limits (pyrate-limiter)

| Env Var                            | Type    |  Default | Allowed / Format            | Description                                     | Used by     |                                             |             |
| ---------------------------------- | ------- | -------: | --------------------------- | ----------------------------------------------- | ----------- | ------------------------------------------- | ----------- |
| `ONTOFETCH_RATELIMIT__DEFAULT`     | str     | *(null)* | `N/second                   | minute                                          | hour`       | Global quota when service-specific not set. | Ratelimiter |
| `ONTOFETCH_RATELIMIT__PER_SERVICE` | str map | *(null)* | `service:rate;service:rate` | Per-service quotas.                             | Ratelimiter |                                             |             |
| `ONTOFETCH_RATELIMIT__SHARED_DIR`  | path    | *(null)* | abs path                    | Enables **SQLite** shared buckets across procs. | Ratelimiter |                                             |             |
| `ONTOFETCH_RATELIMIT__ENGINE`      | enum    | `pyrate` | `pyrate`                    | Engine selector (future-proof).                 | Ratelimiter |                                             |             |

**Example**
`ONTOFETCH_RATELIMIT__PER_SERVICE="ols:4/second;bioportal:2/second"`

## D) Extraction Policy (Safety / Throughput / Integrity)

**Safety & Structure**

| Env Var                                        | Type  |      Default | Description                                      |                                    |                 |
| ---------------------------------------------- | ----- | -----------: | ------------------------------------------------ | ---------------------------------- | --------------- |
| `ONTOFETCH_EXTRACT__ENCAPSULATE`               | bool  |       `true` | Extract inside deterministic root subdir.        |                                    |                 |
| `ONTOFETCH_EXTRACT__ENCAPSULATION_NAME`        | enum  |     `sha256` | `sha256` or `basename`.                          |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_DEPTH`                 | int   |         `32` | Max path depth (components).                     |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_COMPONENTS_LEN`        | int   |        `240` | Max bytes per path component (UTF-8).            |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_PATH_LEN`              | int   |       `4096` | Max bytes per full path (UTF-8).                 |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_ENTRIES`               | int   |      `50000` | Max extractable entries per archive.             |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_FILE_SIZE_BYTES`       | int   | `2147483648` | Per-file size cap (2 GiB).                       |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_TOTAL_RATIO`           | float |       `10.0` | Global zip-bomb ratio (uncompressed/compressed). |                                    |                 |
| `ONTOFETCH_EXTRACT__MAX_ENTRY_RATIO`           | float |      `100.0` | Per-entry ratio cap (when available).            |                                    |                 |
| `ONTOFETCH_EXTRACT__UNICODE_FORM`              | enum  |        `NFC` | `NFC` or `NFD` normalization.                    |                                    |                 |
| `ONTOFETCH_EXTRACT__CASEFOLD_COLLISION_POLICY` | enum  |     `reject` | `reject                                          | allow` for case-insensitive dupes. |                 |
| `ONTOFETCH_EXTRACT__OVERWRITE`                 | enum  |     `reject` | `reject                                          | replace                            | keep_existing`. |
| `ONTOFETCH_EXTRACT__DUPLICATE_POLICY`          | enum  |     `reject` | In-archive duplicates: `reject                   | first_wins                         | last_wins`.     |

**Throughput**

| Env Var                                    | Type  |   Default | Description                        |
| ------------------------------------------ | ----- | --------: | ---------------------------------- |
| `ONTOFETCH_EXTRACT__SPACE_SAFETY_MARGIN`   | float |    `1.10` | Required free-space headroom.      |
| `ONTOFETCH_EXTRACT__PREALLOCATE`           | bool  |    `true` | Preallocate files when size known. |
| `ONTOFETCH_EXTRACT__COPY_BUFFER_MIN`       | int   |   `65536` | Min copy buffer bytes (64 KiB).    |
| `ONTOFETCH_EXTRACT__COPY_BUFFER_MAX`       | int   | `1048576` | Max copy buffer bytes (1 MiB).     |
| `ONTOFETCH_EXTRACT__GROUP_FSYNC`           | int   |      `32` | fsync directory every N files.     |
| `ONTOFETCH_EXTRACT__MAX_WALL_TIME_SECONDS` | int   |     `120` | Soft time budget per archive.      |

**Integrity & Filtering**

| Env Var                                      | Type |         Default | Description                               |           |                     |
| -------------------------------------------- | ---- | --------------: | ----------------------------------------- | --------- | ------------------- |
| `ONTOFETCH_EXTRACT__HASH_ENABLE`             | bool |          `true` | Compute file digests during write.        |           |                     |
| `ONTOFETCH_EXTRACT__HASH_ALGORITHMS`         | CSV  |        `sha256` | e.g., `sha256,sha1` (prefer sha256 only). |           |                     |
| `ONTOFETCH_EXTRACT__INCLUDE_GLOBS`           | CSV  |       *(empty)* | Only extract matching paths.              |           |                     |
| `ONTOFETCH_EXTRACT__EXCLUDE_GLOBS`           | CSV  |       *(empty)* | Skip matching paths.                      |           |                     |
| `ONTOFETCH_EXTRACT__TIMESTAMPS_MODE`         | enum |      `preserve` | `preserve                                 | normalize | source_date_epoch`. |
| `ONTOFETCH_EXTRACT__TIMESTAMPS_NORMALIZE_TO` | enum | `archive_mtime` | When `normalize`: `archive_mtime          | now`.     |                     |

## E) Storage (Local)

| Env Var                               | Type |                Default | Allowed / Format | Description                                       |
| ------------------------------------- | ---- | ---------------------: | ---------------- | ------------------------------------------------- |
| `ONTOFETCH_STORAGE__ROOT`             | path | `<project>/ontologies` | abs path         | Blob root for archives/extractions.               |
| `ONTOFETCH_STORAGE__LATEST_NAME`      | str  |          `LATEST.json` | filename         | Marker name in root.                              |
| `ONTOFETCH_STORAGE__URL` *(optional)* | url  |               *(null)* | fsspec URL       | Future remote backend (ignored if staying local). |

## F) DuckDB Catalog

| Env Var                        | Type |                            Default | Description                               |
| ------------------------------ | ---- | ---------------------------------: | ----------------------------------------- |
| `ONTOFETCH_DB__PATH`           | path | `<root>/.catalog/ontofetch.duckdb` | DB file path.                             |
| `ONTOFETCH_DB__THREADS`        | int  |                      `min(8, CPU)` | DuckDB threads.                           |
| `ONTOFETCH_DB__READONLY`       | bool |                            `false` | Open DB read-only.                        |
| `ONTOFETCH_DB__WLOCK`          | bool |                             `true` | Writer file-lock enabled.                 |
| `ONTOFETCH_DB__PARQUET_EVENTS` | bool |                            `false` | Store events as Parquet instead of table. |

## G) Logging & Telemetry

| Env Var                            | Type |  Default | Description                                |      |      |         |
| ---------------------------------- | ---- | -------: | ------------------------------------------ | ---- | ---- | ------- |
| `ONTOFETCH_LOG__LEVEL`             | enum |   `INFO` | `DEBUG                                     | INFO | WARN | ERROR`. |
| `ONTOFETCH_LOG__JSON`              | bool |   `true` | JSON logs on/off.                          |      |      |         |
| `ONTOFETCH_TELEMETRY__RUN_ID`      | UUID | *(auto)* | Inject fixed run id (determinism/testing). |      |      |         |
| `ONTOFETCH_TELEMETRY__EMIT_EVENTS` | bool |   `true` | Write `extract.*` events to logs/DB.       |      |      |         |

---

# Settings Reference (drop-in page)

## Overview

`OntologyDownload/settings.py` centralizes configuration using **Pydantic v2** and `pydantic-settings`. It loads values from **CLI overlay → config file → env → defaults**, validates & **normalizes** them once, and exposes typed sub-configs used by the rest of the system.

## Source Precedence

1. **CLI overlay** (dict you pass in)
2. `ONTOFETCH_CONFIG` (**TOML/YAML/JSON** file; optional)
3. `.env.ontofetch` (if present)
4. `.env` (if present)
5. **Environment variables** (`ONTOFETCH_*`)
6. **Defaults** (in code)

> We **log sources**, not values, and compute a **config hash** for provenance.

## Sub-models (what consumers read)

* `HttpSettings`: timeouts, pool, HTTP/2, UA, proxy trust
* `CacheSettings`: Hishel path, enabled/bypass
* `RetrySettings`: connect retries/backoff
* `SecuritySettings`: allowlist, ports, DNS/HTTP flags; helpers:

  * `normalized_allowed_hosts() -> (exact, suffixes, per_host_ports, ip_literals)`
  * `allowed_port_set() -> set[int]`
* `RateLimitSettings`: default & per-service rates; helpers:

  * `parse_service_rate_limit(service) -> (limit, unit, rps)`
* `ExtractionSettings`: safety limits, overwrites/dupes, throughput knobs, hashing & timestamp policy; helpers:

  * `include_filters() -> Callable[path->bool]`
* `StorageSettings`: local root, latest marker name (remote URL reserved)
* `DuckDBSettings`: DB path, threads, readonly, writer lock, parquet events
* `LoggingSettings` & `TelemetrySettings`: log level/format, run id, event emission

### Derived Helpers (ready-to-use)

* `config_hash() -> str`: SHA-256 of normalized, non-secret config
* `include_filters()` returns a fast predicate combining include/exclude globs
* `user_agent()` builds the final UA once (base + version + contact URL)

## Validation & Normalization Rules

* **Hosts/Ports**: parse `ALLOWED_HOSTS` into exacts/suffixes/IPs + per-host ports (`host:443`), merge with global ports set.
* **Rate strings**: `N/second|minute|hour` parsed to `(N, unit, rps)` with exact unit enum.
* **Paths**: normalized to **absolute POSIX**; parent dirs created as needed for cache, DB path, storage root.
* **Enums**: case-insensitive; invalid → crisp error message showing allowed values.
* **Numeric bounds**: timeouts > 0; ratios ≥ 1; sizes ≥ 0; wall time ≥ 0.

## Typical Usage (internals)

```text
# Pseudocode (no code in PR text)
s = settings.get()
http = s.http
cache = s.cache
sec  = s.security
rl   = s.ratelimit
ext  = s.extraction
store= s.storage
db   = s.db
```

* **Network client** reads `http`, `cache`, `retry`, `rl` to build the HTTPX+Hishel adapter.
* **Validator** uses `sec.normalized_allowed_hosts()` and `sec.allowed_port_set()`.
* **Extractor** uses `ext` for encapsulation, limits, buffers, hashing, timestamps.
* **Storage** uses `store.root` & `store.latest_name`.
* **DuckDB** uses `db.path`, `db.threads`, `db.readonly`, `db.wlock`.
* **Logs** use `log.level`, `log.json`; Telemetry uses `telemetry.run_id`, `emit_events`.

## Example `.env`

```dotenv
# HTTP
ONTOFETCH_HTTP__TIMEOUT_CONNECT=5
ONTOFETCH_HTTP__POOL_MAX_CONNECTIONS=64
ONTOFETCH_CACHE__DIR=/var/tmp/ontofetch/http-cache

# Security
ONTOFETCH_SECURITY__ALLOWED_HOSTS=ebi.ac.uk,*.purl.org,10.0.0.7
ONTOFETCH_SECURITY__ALLOWED_PORTS=80,443,8443
ONTOFETCH_SECURITY__ALLOW_PLAIN_HTTP_FOR_HOST_ALLOWLIST=false
ONTOFETCH_SECURITY__STRICT_DNS=true

# Rate limits
ONTOFETCH_RATELIMIT__DEFAULT=8/second
ONTOFETCH_RATELIMIT__PER_SERVICE=ols:4/second;bioportal:2/second

# Extraction
ONTOFETCH_EXTRACT__ENCAPSULATE=true
ONTOFETCH_EXTRACT__INCLUDE_GLOBS=*.ttl,*.rdf,*.owl,*.obo
ONTOFETCH_EXTRACT__MAX_TOTAL_RATIO=10.0
ONTOFETCH_EXTRACT__OVERWRITE=reject

# Storage & DB
ONTOFETCH_STORAGE__ROOT=/data/ontologies
ONTOFETCH_DB__PATH=/data/ontologies/.catalog/ontofetch.duckdb

# Logging
ONTOFETCH_LOG__LEVEL=INFO
ONTOFETCH_LOG__JSON=true
```

## FAQs / Guidance

* **Local-only storage?** Set `ONTOFETCH_STORAGE__ROOT` and leave `ONTOFETCH_STORAGE__URL` unset.
* **Per-service quotas?** Use `ONTOFETCH_RATELIMIT__PER_SERVICE="ols:4/second;bioportal:2/second"`.
* **Allow certain HTTP endpoints?** Add them to `ALLOWED_HOSTS`, then set `ALLOW_PLAIN_HTTP_FOR_HOST_ALLOWLIST=true` (still discouraged; prefer HTTPS).
* **Private networks?** Only allowed when host/IP is on the allowlist **and** `ALLOW_PRIVATE_NETWORKS_FOR_HOST_ALLOWLIST=true`.
* **Determinism for builds**: set `TELEMETRY__RUN_ID` and `EXTRACT__TIMESTAMPS_MODE=source_date_epoch` with `SOURCE_DATE_EPOCH`.

---

If you’d like, I can generate a **ready-to-commit `SETTINGS.md`** with these sections formatted for your docs, or a **CSV** version of the ENV matrix for spreadsheets.
