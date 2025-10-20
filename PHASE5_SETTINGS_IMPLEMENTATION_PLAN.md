# Phase 5: Pydantic v2 Settings Configuration System - Implementation Plan

**Status**: Planning Phase
**Target Completion**: 4-6 weeks
**Risk Level**: MEDIUM (complex refactoring, but well-scoped by documentation)
**Breaking Changes**: NONE (backward compatible via exports and builders)

---

## 1. Executive Summary

Phase 5 implements a **production-grade Pydantic v2 + `pydantic-settings` configuration system** that centralizes all OntologyDownload configuration (currently scattered across code, environment reads, and manual parsing) into a single, typed, validated source-of-truth.

### What this solves

- **Configuration fragmentation**: Settings scattered across modules; ad-hoc environment reads
- **Validation gaps**: No early validation of rate limits, hosts, paths, numeric bounds
- **Normalization complexity**: Consumers re-parse CSV strings, compute allowlists, etc.
- **Testing friction**: Hard to inject configs; no principled defaults
- **Observability**: Config provenance and hashing not available for auditing

### What we build

- 1 central `Settings` model (Pydantic v2 `BaseSettings`)
- 10 domain sub-models (HTTP, Cache, Retry, Security, RateLimit, Extraction, Storage, DuckDB, Logging, Telemetry)
- **50+ configuration fields** with validation, normalization, and derived helpers
- Full **source precedence**: CLI → config file → .env → environment → defaults
- **Non-intrusive rollout**: Existing call-sites unchanged via builder compatibility layer

---

## 2. Architecture Overview

### 2.1 Design Principles

1. **Single Responsibility**: Each sub-model owns one domain (HTTP, Security, etc.)
2. **Validation @ Entry**: All normalization happens once on construction
3. **Immutability**: Settings frozen after load (via `frozen=True` on BaseSettings)
4. **No Secrets in Logs**: Sensitive fields excluded from `__repr__` and config hashing
5. **Backward Compat**: Public builders kept unchanged; route to settings internally
6. **Composability**: Sub-models can be used independently or within the root model
7. **Testability**: Fixtures override via `CliRunner.invoke(..., env={...})` or context managers

### 2.2 Domain Decomposition

```
Settings (BaseSettings)
├── http: HttpSettings
│   ├── http2: bool
│   ├── timeout_connect: float
│   ├── timeout_read: float
│   ├── timeout_write: float
│   ├── timeout_pool: float
│   ├── pool_max_connections: int
│   ├── pool_keepalive_max: int
│   ├── keepalive_expiry: float
│   ├── trust_env: bool
│   └── user_agent: str
├── cache: CacheSettings
│   ├── enabled: bool
│   ├── dir: Path
│   └── bypass: bool
├── retry: RetrySettings
│   ├── connect_retries: int
│   ├── backoff_base: float
│   └── backoff_max: float
├── security: SecuritySettings
│   ├── allowed_hosts: List[str]
│   ├── allowed_ports: Set[int]
│   ├── allow_private_networks: bool
│   ├── allow_plain_http: bool
│   ├── strict_dns: bool
│   └── [helpers] normalized_allowed_hosts(), allowed_port_set(), ...
├── ratelimit: RateLimitSettings
│   ├── default: Optional[str]
│   ├── per_service: Dict[str, str]
│   ├── shared_dir: Optional[Path]
│   ├── engine: str
│   └── [helpers] parse_service_rate_limit(), ...
├── extraction: ExtractionSettings
│   ├── [safety] encapsulate, max_depth, max_entries, ...
│   ├── [throughput] space_safety_margin, preallocate, buffer_sizes, ...
│   ├── [integrity] hash_enable, hash_algorithms, include_globs, ...
│   └── [helpers] include_filters(), file_passes_extraction(), ...
├── storage: StorageSettings
│   ├── root: Path
│   ├── latest_name: str
│   └── url: Optional[str]  # future
├── db: DuckDBSettings
│   ├── path: Path
│   ├── threads: int
│   ├── readonly: bool
│   ├── wlock: bool
│   └── parquet_events: bool
├── logging: LoggingSettings
│   ├── level: str
│   ├── json: bool
│   └── [helpers] level_int(), ...
├── telemetry: TelemetrySettings
│   ├── run_id: UUID
│   └── emit_events: bool
└── [module-level helpers]
    ├── config_hash() -> str
    ├── resolve_sources() -> Dict[str, str]
    └── [singleton getter] get() -> Settings
```

### 2.3 Source Precedence & Loading

```
┌─────────────────────────────────────────────────────────┐
│  CLI Overlay (dict passed by argparse/typer)            │  ← Highest priority
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Config File (TOML/YAML/JSON via $ONTOFETCH_CONFIG)     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Dotenv Files (.env.ontofetch, .env)                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Environment Variables (ONTOFETCH_*)                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Baked-in Defaults (in code)                            │  ← Lowest priority
└─────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Field Specifications

### 3.1 A) HTTP / Network (11 fields + 3 cache + 3 retry = 17 total)

**HttpSettings**

| Field | Type | Default | Validation | Notes |
|-------|------|---------|-----------|-------|
| `http2` | bool | `true` | - | Enable HTTP/2 |
| `timeout_connect` | float | `5.0` | `> 0` | Connect timeout (seconds) |
| `timeout_read` | float | `30.0` | `> 0` | Read timeout (seconds) |
| `timeout_write` | float | `30.0` | `> 0` | Write timeout (seconds) |
| `timeout_pool` | float | `5.0` | `> 0` | Acquire-from-pool timeout |
| `pool_max_connections` | int | `64` | `>= 1` | Max concurrent connections |
| `pool_keepalive_max` | int | `20` | `>= 0` | Keepalive pool size |
| `keepalive_expiry` | float | `30.0` | `>= 0` | Idle connection expiry (seconds) |
| `trust_env` | bool | `true` | - | Honor HTTP(S)_PROXY env vars |
| `user_agent` | str | `DocsToKG/OntoFetch` | - | User-Agent header |

**CacheSettings**

| Field | Type | Default | Validation | Notes |
|-------|------|---------|-----------|-------|
| `enabled` | bool | `true` | - | Enable Hishel RFC-9111 cache |
| `dir` | Path | `~/.cache/http` | normalization | Cache directory (auto-created) |
| `bypass` | bool | `false` | - | Force bypass (no revalidation) |

**RetrySettings**

| Field | Type | Default | Validation | Notes |
|-------|------|---------|-----------|-------|
| `connect_retries` | int | `2` | `>= 0` | Retries for connect errors |
| `backoff_base` | float | `0.1` | `>= 0` | Backoff start (seconds) |
| `backoff_max` | float | `2.0` | `>= 0` | Backoff cap (seconds) |

**Environment Variables:**

```bash
ONTOFETCH_HTTP__HTTP2=true
ONTOFETCH_HTTP__TIMEOUT_CONNECT=5
ONTOFETCH_HTTP__TIMEOUT_READ=30
ONTOFETCH_HTTP__TIMEOUT_WRITE=30
ONTOFETCH_HTTP__TIMEOUT_POOL=5
ONTOFETCH_HTTP__POOL_MAX_CONNECTIONS=64
ONTOFETCH_HTTP__POOL_KEEPALIVE_MAX=20
ONTOFETCH_HTTP__KEEPALIVE_EXPIRY=30
ONTOFETCH_HTTP__TRUST_ENV=true
ONTOFETCH_HTTP__USER_AGENT=DocsToKG/OntoFetch

ONTOFETCH_CACHE__ENABLED=true
ONTOFETCH_CACHE__DIR=/var/tmp/ontofetch/http-cache
ONTOFETCH_CACHE__BYPASS=false

ONTOFETCH_RETRY__CONNECT_RETRIES=2
ONTOFETCH_RETRY__BACKOFF_BASE=0.1
ONTOFETCH_RETRY__BACKOFF_MAX=2.0
```

### 3.2 B) URL Security & DNS (5 fields)

**SecuritySettings**

| Field | Type | Default | Validation | Notes |
|-------|------|---------|-----------|-------|
| `allowed_hosts` | List[str] | `[]` | parsing | CSV with `host`, `*.suffix`, IP/CIDR, `:port` |
| `allowed_ports` | Set[int] | `{80,443}` | `1-65535` | Global + per-host ports merged |
| `allow_private_networks` | bool | `false` | - | Permit private/loopback if allowlisted |
| `allow_plain_http` | bool | `false` | - | Allow `http://` if allowlisted |
| `strict_dns` | bool | `true` | - | Fail on DNS resolution failure |

**Helpers:**

- `normalized_allowed_hosts() -> (exact_domains, wildcard_suffixes, per_host_ports, ip_literals)`
- `allowed_port_set() -> Set[int]`
- `is_host_allowed(host: str, port: int, use_https: bool) -> bool`

**Environment Variables:**

```bash
ONTOFETCH_SECURITY__ALLOWED_HOSTS="ebi.ac.uk,*.purl.org,10.0.0.7,141.0.0.0/8,example.org:8443"
ONTOFETCH_SECURITY__ALLOWED_PORTS="80,443,8443"
ONTOFETCH_SECURITY__ALLOW_PRIVATE_NETWORKS=false
ONTOFETCH_SECURITY__ALLOW_PLAIN_HTTP=false
ONTOFETCH_SECURITY__STRICT_DNS=true
```

### 3.3 C) Rate Limits (4 fields)

**RateLimitSettings**

| Field | Type | Default | Validation | Notes |
|-------|------|---------|-----------|-------|
| `default` | Optional[str] | `null` | regex `^(\d+\.?\d*)/(second\|minute\|hour)$` | Global quota |
| `per_service` | Dict[str, str] | `{}` | per-entry regex | Service-specific quotas |
| `shared_dir` | Optional[Path] | `null` | normalization | SQLite shared bucket dir |
| `engine` | str | `pyrate` | enum | Engine selector (future-proof) |

**Helpers:**

- `parse_service_rate_limit(service: str) -> Optional[Tuple[float, str, float]]` → `(limit, unit, rps)`
- `get_bucket_for_service(service: Optional[str]) -> Bucket`

**Environment Variables:**

```bash
ONTOFETCH_RATELIMIT__DEFAULT="8/second"
ONTOFETCH_RATELIMIT__PER_SERVICE="ols:4/second;bioportal:2/second"
ONTOFETCH_RATELIMIT__SHARED_DIR="/tmp/ontofetch-buckets"
ONTOFETCH_RATELIMIT__ENGINE="pyrate"
```

### 3.4 D) Extraction Policy (23 fields: safety + throughput + integrity)

**ExtractionSettings (Safety)**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `encapsulate` | bool | `true` | Extract in deterministic root |
| `encapsulation_name` | enum | `sha256` | `sha256` or `basename` |
| `max_depth` | int | `32` | Max path components |
| `max_components_len` | int | `240` | Max bytes per component |
| `max_path_len` | int | `4096` | Max bytes per full path |
| `max_entries` | int | `50000` | Max entries per archive |
| `max_file_size_bytes` | int | `2147483648` | Per-file size cap (2 GiB) |
| `max_total_ratio` | float | `10.0` | Zip-bomb ratio (uncompressed/compressed) |
| `max_entry_ratio` | float | `100.0` | Per-entry ratio cap |
| `unicode_form` | enum | `NFC` | `NFC` or `NFD` normalization |
| `casefold_collision_policy` | enum | `reject` | `reject` or `allow` for case dupes |
| `overwrite` | enum | `reject` | `reject`, `replace`, `keep_existing` |
| `duplicate_policy` | enum | `reject` | `reject`, `first_wins`, `last_wins` |

**ExtractionSettings (Throughput)**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `space_safety_margin` | float | `1.10` | Required free-space headroom |
| `preallocate` | bool | `true` | Preallocate files when size known |
| `copy_buffer_min` | int | `65536` | Min copy buffer bytes (64 KiB) |
| `copy_buffer_max` | int | `1048576` | Max copy buffer bytes (1 MiB) |
| `group_fsync` | int | `32` | fsync directory every N files |
| `max_wall_time_seconds` | int | `120` | Soft time budget per archive |

**ExtractionSettings (Integrity)**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `hash_enable` | bool | `true` | Compute file digests during write |
| `hash_algorithms` | List[str] | `['sha256']` | e.g., `sha256,sha1` |
| `include_globs` | List[str] | `[]` | Only extract matching paths |
| `exclude_globs` | List[str] | `[]` | Skip matching paths |
| `timestamps_mode` | enum | `preserve` | `preserve`, `normalize`, `source_date_epoch` |
| `timestamps_normalize_to` | enum | `archive_mtime` | When normalizing: `archive_mtime` or `now` |

**Helpers:**

- `include_filters() -> Callable[[str], bool]` (combines include/exclude globs)
- `file_passes_extraction(path: str) -> bool`
- `compute_extraction_root(url: str) -> Path`

### 3.5 E) Storage (3 fields, local-only for now)

**StorageSettings**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `root` | Path | `./ontologies` | Blob root for archives/extractions |
| `latest_name` | str | `LATEST.json` | Marker name in root |
| `url` | Optional[str] | `null` | Future remote backend (fsspec) |

### 3.6 F) DuckDB Catalog (5 fields)

**DuckDBSettings**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `path` | Path | `~/.catalog/ontofetch.duckdb` | DB file path |
| `threads` | int | `min(8, CPU)` | Query execution threads |
| `readonly` | bool | `false` | Open DB read-only |
| `wlock` | bool | `true` | Writer file-lock enabled |
| `parquet_events` | bool | `false` | Store events as Parquet |

### 3.7 G) Logging & Telemetry (5 fields)

**LoggingSettings**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `level` | enum | `INFO` | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `json` | bool | `true` | JSON logs on/off |

**TelemetrySettings**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `run_id` | UUID | `uuid4()` | Fixed run id for determinism/testing |
| `emit_events` | bool | `true` | Write events to logs/DB |

**Helpers:**

- `level_int() -> int` (convert to logging level int)

---

## 4. Implementation Roadmap (5 Phases)

### Phase 5.1: Domain Models Foundation (Days 1-5)

**Scope**: Build HttpSettings, CacheSettings, RetrySettings, LoggingSettings, TelemetrySettings

**Tasks:**

1. Create domain model base classes (validation patterns, common helpers)
2. Implement HttpSettings with timeout/pool validators
3. Implement CacheSettings with path normalization
4. Implement RetrySettings with backoff validation
5. Implement LoggingSettings with level enum
6. Implement TelemetrySettings with UUID auto-generation
7. Write unit tests for all domain models (field validation, normalization, serialization)

**Deliverables:**

- `src/DocsToKG/OntologyDownload/settings_domains.py` (or inline in settings.py)
- `tests/ontology_download/test_settings_domain_models.py`
- **Test Count**: 40-50 tests (defaults, env mapping, validation errors, normalization)

### Phase 5.2: Complex Domain Models (Days 6-12)

**Scope**: Build SecuritySettings, RateLimitSettings, ExtractionSettings, StorageSettings, DuckDBSettings

**Tasks:**

1. Implement SecuritySettings with host/port parsing, CIDR support, normalization helpers
2. Implement RateLimitSettings with rate string parsing, per-service breakdown
3. Implement ExtractionSettings with 3-part domain (safety/throughput/integrity), globs compilation
4. Implement StorageSettings with path normalization
5. Implement DuckDBSettings with thread auto-detection
6. Write unit tests for each (parsing, normalization, helpers, edge cases)

**Deliverables:**

- Complex domain models in settings.py
- `tests/ontology_download/test_settings_security.py`
- `tests/ontology_download/test_settings_ratelimit.py`
- `tests/ontology_download/test_settings_extraction.py`
- `tests/ontology_download/test_settings_storage_db.py`
- **Test Count**: 80-100 tests

### Phase 5.3: Root Settings Model & Loading (Days 13-18)

**Scope**: Build root Settings (BaseSettings), source precedence, singleton getter

**Tasks:**

1. Implement root `Settings(BaseSettings)` with all 10 sub-models
2. Implement `settings_customise_sources()` for precedence (CLI → config → .env → env → defaults)
3. Implement config file loader (TOML/YAML/JSON support)
4. Implement `.env` file discovery and loading
5. Implement singleton `get()` accessor with caching
6. Implement `config_hash()` for provenance
7. Implement `resolve_sources()` for debugging
8. Implement `model_rebuild()` for forward refs if needed

**Deliverables:**

- Root Settings model and helpers in settings.py
- `tests/ontology_download/test_settings_loading.py`
- **Test Count**: 30-40 tests (defaults, env mapping, config file precedence, hashing)

### Phase 5.4: Integration Layer & Backward Compat (Days 19-24)

**Scope**: Wire settings into existing builders, maintain public API, update exports

**Tasks:**

1. Update exports.py to add Settings, domain models to public API
2. Update existing builder functions to use `settings.get()` internally
3. Ensure no breaking changes to call-sites
4. Update cli.py to inject Settings into subcommands via context
5. Update planning.py to read from Settings instead of direct env reads
6. Update validation.py, resolvers.py, io/* to use Settings
7. Create integration tests that verify end-to-end flow (CLI → Settings → usage)

**Deliverables:**

- Integration layer in settings.py and exports.py
- CLI context wiring in cli.py
- `tests/ontology_download/test_settings_integration.py`
- **Test Count**: 20-30 integration tests

### Phase 5.5: Testing, Docs, Finalization (Days 25-30)

**Scope**: Comprehensive test suite, documentation, code review prep

**Tasks:**

1. Run full test suite to ensure no regressions
2. Create `.env.example` with all fields
3. Create `SETTINGS.md` reference page (env matrix, examples, FAQs)
4. Create migration guide for code that currently reads environment directly
5. Code review and refinement
6. Performance validation (settings loading time)

**Deliverables:**

- Comprehensive test coverage (>90% for settings module)
- `.env.example` file
- `docs/06-operations/SETTINGS.md`
- Migration guide in comments
- **Test Count**: 150-200 total tests

---

## 5. Key Design Decisions

### 5.1 Why Pydantic v2?

- **Speed**: Rust-backed validation via pydantic-core
- **Features**: Discriminated unions, computed fields, mode-based validators
- **Ecosystem**: pydantic-settings for BaseSettings
- **Future-proof**: Modern, maintained, clear API

### 5.2 Why BaseSettings (not just BaseModel)?

- Built-in environment variable loading
- Dotenv support
- Custom source precedence via `settings_customise_sources()`
- Caching and lazy-loading hooks

### 5.3 Why domain sub-models?

- **Separation of concerns**: Each domain owns its validation
- **Reusability**: Sub-models can be tested/used independently
- **Clarity**: Consumers read from clearly-named sub-models, not flat field list
- **Extensibility**: Future additions don't pollute root Settings

### 5.4 Why singleton `get()`?

- Settings should be loaded once, reused everywhere
- Caching prevents re-parsing expensive operations (path normalization, host parsing)
- Testing can override via fixtures that `monkeypatch.setattr(...get.cache_clear())`
- No global state (getter is a function, not a module-level var)

### 5.5 Why no secrets in config_hash()?

- `BIOPORTAL_API_KEY`, `EUROPE_PMC_API_KEY`, etc. excluded from hash
- Hash used for provenance tracking (e.g., "why did we choose this resolver?")
- Secrets shouldn't leak into logs/telemetry

### 5.6 Why CSV parsing in validators?

- CSV is human-friendly (e.g., `ALLOWED_HOSTS="host1,host2,host3"`)
- Pydantic validators can coerce strings → typed collections
- Still strict validation (e.g., port range checks, CIDR validation)

### 5.7 Why frozen Settings?

- Configuration shouldn't change mid-run (source of bugs)
- `model_config = ConfigDict(frozen=True)` prevents accidental mutation
- Tests can create new Settings instances per test

---

## 6. Test Strategy

### 6.1 Test Categories

**Unit Tests** (150-160 tests)

- Domain models: defaults, validation, normalization
- Field validators: edge cases, error messages
- Helpers: rate parsing, host normalization, glob compilation
- Serialization/deserialization round-trips

**Integration Tests** (30-40 tests)

- Settings loading from each source (CLI, config, .env, env, defaults)
- Source precedence verification
- Singleton getter caching
- Config hashing

**End-to-End Tests** (20-30 tests)

- CLI receives settings via typer context
- Planning layer reads from settings
- Validation layer applies extraction policy
- Downloader respects rate limits

### 6.2 Fixture Strategy

```python
@pytest.fixture
def temp_env(monkeypatch, tmp_path):
    """Temporarily override environment variables and paths."""
    monkeypatch.setenv("ONTOFETCH_HTTP__TIMEOUT_CONNECT", "10")
    monkeypatch.setenv("ONTOFETCH_SECURITY__ALLOWED_HOSTS", "example.com")
    # Clear singleton cache
    from DocsToKG.OntologyDownload.settings import get
    get.cache_clear()
    yield
    get.cache_clear()

@pytest.fixture
def settings_from_dict(monkeypatch):
    """Create Settings from dict (for testing CLI overlay)."""
    def _make(**kwargs):
        from DocsToKG.OntologyDownload.settings import Settings
        return Settings.model_validate(kwargs)
    return _make
```

---

## 7. Backward Compatibility & Migration Path

### 7.1 Phase-in Strategy

**Week 1-2**: Settings module only

- Implement all domain models and root Settings
- No changes to existing code
- Existing code continues to read env directly

**Week 3-4**: Optional adoption

- Update builders (e.g., `build_download_config()`) to route through settings
- Existing callers see no change (API stable)
- New code can opt-in via `settings.get()`

**Week 5-6**: Full integration

- CLI wires Settings into context
- Planning, validation, io modules read from settings
- Env reads gradually removed
- Tests updated to use settings fixtures

### 7.2 No Breaking Changes

- All existing public builder functions kept
- All existing function signatures unchanged
- Existing tests pass without modification
- New tests alongside old (gradual rollout)

---

## 8. Success Criteria

- [ ] All 50+ configuration fields implemented with validation
- [ ] 150-200 tests passing (>90% coverage for settings module)
- [ ] Source precedence working correctly (CLI → config → .env → env → defaults)
- [ ] Settings loading time < 100ms (with caching < 1μs on re-access)
- [ ] Config hashing stable and reproducible
- [ ] All domain models immutable (frozen)
- [ ] Zero breaking changes to public API
- [ ] Migration guide complete and clear
- [ ] Documentation (SETTINGS.md) comprehensive and easy to follow
- [ ] Code review approval from 2+ team members

---

## 9. Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Circular imports during settings loading | MEDIUM | Lazy imports in validators; TYPE_CHECKING guards |
| Settings not thread-safe | MEDIUM | Use `functools.lru_cache` for singleton getter; frozen models |
| Performance regression (settings loading slow) | LOW | Profile load time; cache normalization results |
| Breaking changes to CLI | HIGH | Extensive backward-compat tests; keep all builders |
| Validation too strict breaks existing configs | MEDIUM | Lenient defaults; clear error messages; migration guide |

---

## 10. File Structure (New & Modified)

### New Files

- `src/DocsToKG/OntologyDownload/settings.py` — MAJOR EXPANSION (add domain models, root Settings, loaders)
- `tests/ontology_download/test_settings_domain_models.py`
- `tests/ontology_download/test_settings_loading.py`
- `tests/ontology_download/test_settings_security.py`
- `tests/ontology_download/test_settings_ratelimit.py`
- `tests/ontology_download/test_settings_extraction.py`
- `tests/ontology_download/test_settings_storage_db.py`
- `tests/ontology_download/test_settings_integration.py`
- `.env.example` (at repo root)
- `docs/06-operations/SETTINGS.md`

### Modified Files

- `src/DocsToKG/OntologyDownload/exports.py` — Add Settings classes to exports
- `src/DocsToKG/OntologyDownload/cli.py` — Wire Settings into typer context
- `src/DocsToKG/OntologyDownload/planning.py` — Read from settings instead of env
- `src/DocsToKG/OntologyDownload/validation.py` — Read from settings
- `src/DocsToKG/OntologyDownload/resolvers.py` — Read rate limits from settings
- `src/DocsToKG/OntologyDownload/io/network.py` — Read HTTP config from settings

---

## 11. Estimation & Timeline

| Phase | Days | Confidence | Notes |
|-------|------|------------|-------|
| 5.1 (Domain Models Foundation) | 5 | HIGH | Straightforward Pydantic models + tests |
| 5.2 (Complex Domain Models) | 7 | HIGH | More complex parsing, but well-specified |
| 5.3 (Root Settings & Loading) | 6 | MEDIUM | Precedence logic can be tricky; needs thorough testing |
| 5.4 (Integration & Backward Compat) | 6 | MEDIUM | Requires careful refactoring of existing code |
| 5.5 (Testing, Docs, Finalization) | 6 | HIGH | Straightforward; mostly documentation and polish |
| **Total** | **30 days** | **HIGH** | ~4-6 weeks with normal velocity |

---

## 12. Next Steps

1. **Review & Approval**: Stakeholders review this plan
2. **Kick-off Phase 5.1**: Implement domain models
3. **Daily Standups**: Brief daily sync on progress, blockers
4. **Mid-phase Check-in**: After Phase 5.3, review source precedence and adjust if needed
5. **Code Review**: Each phase includes peer review before merging

---

## Appendix A: Example Configuration Flow

```python
# 1. Settings load from sources (auto on first access)
from DocsToKG.OntologyDownload.settings import get

s = get()  # Loads from: CLI override → config file → .env → ONTOFETCH_* → defaults

# 2. Each domain is typed and validated
print(s.http.timeout_connect)  # float: 5.0
print(s.security.normalized_allowed_hosts())  # Tuple[Set, Set, Dict, Set]
print(s.extraction.include_filters())  # Callable[[str], bool]

# 3. Consumers read from settings
def download_stream(url: str):
    s = get()
    timeout = s.http.timeout_read
    rate_limit = s.ratelimit.parse_service_rate_limit("ols")
    extraction_policy = s.extraction
    # ... proceed with download

# 4. Tests inject custom settings
def test_download_with_low_rate_limit(monkeypatch):
    from DocsToKG.OntologyDownload.settings import get, Settings
    custom_settings = Settings(
        ratelimit=RateLimitSettings(default="1/second")
    )
    monkeypatch.setattr("DocsToKG.OntologyDownload.settings.get", lambda: custom_settings)
    # ... test proceeds with low rate limit
```

---

**Document Created**: 2025-10-20
**Last Updated**: 2025-10-20
**Status**: Ready for Implementation
