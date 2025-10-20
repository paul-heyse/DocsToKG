# Phase 5: Quick Reference for Developers

**When you start implementing Phase 5.1-5.5, use this as your quick lookup.**

---

## File Locations

### Planning Documents

- **Full Plan**: `PHASE5_SETTINGS_IMPLEMENTATION_PLAN.md` (637 lines — detailed specs)
- **Executive Summary**: `PHASE5_PLANNING_EXECUTIVE_SUMMARY.md` (quick overview)
- **This Quick Ref**: `PHASE5_PLANNING_QUICK_REFERENCE.md` (you are here)

### Code to Modify/Create

- **Main settings module**: `src/DocsToKG/OntologyDownload/settings.py` (MAJOR EXPANSION)
- **Exports**: `src/DocsToKG/OntologyDownload/exports.py` (add Settings classes)
- **CLI**: `src/DocsToKG/OntologyDownload/cli.py` (wire Settings into context)
- **Planning**: `src/DocsToKG/OntologyDownload/planning.py` (read from settings)
- **Validation**: `src/DocsToKG/OntologyDownload/validation.py` (read from settings)
- **Resolvers**: `src/DocsToKG/OntologyDownload/resolvers.py` (read rate limits)
- **IO Network**: `src/DocsToKG/OntologyDownload/io/network.py` (read HTTP config)

### Test Files (Create These)

- `tests/ontology_download/test_settings_domain_models.py` (40-50 tests)
- `tests/ontology_download/test_settings_security.py` (20-30 tests)
- `tests/ontology_download/test_settings_ratelimit.py` (15-20 tests)
- `tests/ontology_download/test_settings_extraction.py` (25-35 tests)
- `tests/ontology_download/test_settings_storage_db.py` (10-15 tests)
- `tests/ontology_download/test_settings_loading.py` (30-40 tests)
- `tests/ontology_download/test_settings_integration.py` (20-30 tests)

### Documentation (Create These)

- `.env.example` — Example config file with all 50+ fields
- `docs/06-operations/SETTINGS.md` — Reference page (env matrix, examples, FAQs)

---

## The 10 Domain Models (in order of implementation)

### Phase 5.1 (Foundation)

1. **HttpSettings** — 10 fields
   - Timeouts, pool sizes, HTTP/2, user agent, proxy trust
2. **CacheSettings** — 3 fields
   - Hishel cache enabled, directory, bypass flag
3. **RetrySettings** — 3 fields
   - Connect retries, backoff base/max
4. **LoggingSettings** — 2 fields
   - Log level, JSON mode
5. **TelemetrySettings** — 2 fields
   - Run ID, emit events flag

### Phase 5.2 (Complex)

6. **SecuritySettings** — 5 fields
   - Allowed hosts (with host/port/CIDR parsing)
   - Allowed ports, private networks flag, plain HTTP flag, strict DNS flag
   - Helpers: `normalized_allowed_hosts()`, `allowed_port_set()`, `is_host_allowed()`
7. **RateLimitSettings** — 4 fields
   - Global default, per-service quotas, shared bucket dir, engine
   - Helpers: `parse_service_rate_limit()`, `get_bucket_for_service()`
8. **ExtractionSettings** — 23 fields
   - Safety (encapsulation, max depth/entries/size, ratios, unicode form, dupes)
   - Throughput (space margin, preallocate, buffer sizes, fsync, wall time)
   - Integrity (hashing, include/exclude globs, timestamps mode)
   - Helpers: `include_filters()`, `file_passes_extraction()`, `compute_extraction_root()`
9. **StorageSettings** — 3 fields
   - Root path, latest marker name, future URL for fsspec
10. **DuckDBSettings** — 5 fields
    - DB path, threads, readonly, writer lock, parquet events

### Phase 5.3 (Root + Loading)

- **Settings** (root model)
  - Composes all 10 sub-models
  - Implements `settings_customise_sources()` for precedence
  - Adds: `config_hash()`, `resolve_sources()`, singleton `get()`, `model_rebuild()`

---

## Key Patterns to Follow

### Pydantic v2 Pattern: Field Validator

```python
from pydantic import BaseModel, Field, field_validator

class MySettings(BaseModel):
    timeout: float = Field(default=5.0, gt=0.0)

    @field_validator("timeout", mode="after")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v < 0.1 or v > 60.0:
            raise ValueError("Timeout must be 0.1-60 seconds")
        return v
```

### Pydantic v2 Pattern: Model Config (Immutability)

```python
from pydantic import BaseModel, ConfigDict

class HttpSettings(BaseModel):
    model_config = ConfigDict(frozen=True)  # Make immutable
    # fields...
```

### Pydantic v2 Pattern: Nested Model (Domain Sub-model)

```python
class Settings(BaseSettings):
    http: HttpSettings = Field(default_factory=HttpSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    # ... all 10 domains
```

### Pydantic Settings Pattern: Source Precedence

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=('.env.ontofetch', '.env'),
        env_prefix='ONTOFETCH_',
        nested_delimiter='__',  # for ONTOFETCH_HTTP__TIMEOUT_CONNECT
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type['Settings'],
        init_settings: SettingsSource,
        env_settings: SettingsSource,
        dotenv_settings: SettingsSource,
        file_settings: SettingsSource,
    ) -> tuple[SettingsSource, ...]:
        # Precedence: CLI → config file → .env → env → defaults
        return (
            init_settings,        # CLI overlay
            file_settings,        # TOML/YAML/JSON config
            dotenv_settings,      # .env files
            env_settings,         # ONTOFETCH_* env vars
        )
```

### Helper Pattern: Rate Limit Parsing

```python
import re
from typing import Optional, Tuple

_RATE_LIMIT_PATTERN = re.compile(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$")

def parse_rate_limit(limit_str: Optional[str]) -> Optional[float]:
    """Convert 'N/unit' to requests-per-second."""
    if not limit_str:
        return None
    match = _RATE_LIMIT_PATTERN.match(limit_str)
    if not match:
        raise ValueError(f"Invalid rate limit: {limit_str}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"second", "sec", "s"}:
        return value
    elif unit in {"minute", "min", "m"}:
        return value / 60.0
    elif unit in {"hour", "h"}:
        return value / 3600.0
    return None
```

### Helper Pattern: Host Normalization

```python
def normalized_allowed_hosts(self) -> Optional[Tuple[Set[str], Set[str], Dict[str, Set[int]], Set[str]]]:
    """Parse allowed_hosts into (exact, wildcard_suffixes, per_host_ports, ip_literals)."""
    # Parse CSV entries
    # Split "host.com:port" into host + port
    # Handle "*.example.com" for wildcard
    # Handle "10.0.0.0/8" for CIDR
    # Handle "[::1]:8443" for IPv6
    return (exact_domains, wildcard_suffixes, per_host_ports, ip_literals)
```

### Singleton Pattern: Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get() -> Settings:
    """Load settings once; subsequent calls return cached instance."""
    return Settings()
```

### Test Fixture Pattern: Temp Env

```python
@pytest.fixture
def temp_env(monkeypatch, tmp_path):
    """Override env vars + clear singleton cache."""
    from DocsToKG.OntologyDownload.settings import get

    monkeypatch.setenv("ONTOFETCH_HTTP__TIMEOUT_CONNECT", "10")
    get.cache_clear()  # Force reload
    yield
    get.cache_clear()  # Reset for next test
```

---

## Environment Variable Naming (50+ Fields)

### HTTP Domain (17 env vars)

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
ONTOFETCH_HTTP__USER_AGENT="DocsToKG/OntoFetch"

ONTOFETCH_CACHE__ENABLED=true
ONTOFETCH_CACHE__DIR=/var/tmp/ontofetch/http-cache
ONTOFETCH_CACHE__BYPASS=false

ONTOFETCH_RETRY__CONNECT_RETRIES=2
ONTOFETCH_RETRY__BACKOFF_BASE=0.1
ONTOFETCH_RETRY__BACKOFF_MAX=2.0
```

### Security Domain (5 env vars)

```bash
ONTOFETCH_SECURITY__ALLOWED_HOSTS="ebi.ac.uk,*.purl.org,10.0.0.7,141.0.0.0/8"
ONTOFETCH_SECURITY__ALLOWED_PORTS="80,443,8443"
ONTOFETCH_SECURITY__ALLOW_PRIVATE_NETWORKS=false
ONTOFETCH_SECURITY__ALLOW_PLAIN_HTTP=false
ONTOFETCH_SECURITY__STRICT_DNS=true
```

### Rate Limit Domain (4 env vars)

```bash
ONTOFETCH_RATELIMIT__DEFAULT="8/second"
ONTOFETCH_RATELIMIT__PER_SERVICE="ols:4/second;bioportal:2/second"
ONTOFETCH_RATELIMIT__SHARED_DIR="/tmp/ontofetch-buckets"
ONTOFETCH_RATELIMIT__ENGINE="pyrate"
```

### Extraction Domain (23 env vars)

```bash
# Safety (13 vars)
ONTOFETCH_EXTRACT__ENCAPSULATE=true
ONTOFETCH_EXTRACT__ENCAPSULATION_NAME="sha256"  # sha256 or basename
ONTOFETCH_EXTRACT__MAX_DEPTH=32
ONTOFETCH_EXTRACT__MAX_COMPONENTS_LEN=240
ONTOFETCH_EXTRACT__MAX_PATH_LEN=4096
ONTOFETCH_EXTRACT__MAX_ENTRIES=50000
ONTOFETCH_EXTRACT__MAX_FILE_SIZE_BYTES=2147483648
ONTOFETCH_EXTRACT__MAX_TOTAL_RATIO=10.0
ONTOFETCH_EXTRACT__MAX_ENTRY_RATIO=100.0
ONTOFETCH_EXTRACT__UNICODE_FORM="NFC"  # NFC or NFD
ONTOFETCH_EXTRACT__CASEFOLD_COLLISION_POLICY="reject"  # reject or allow
ONTOFETCH_EXTRACT__OVERWRITE="reject"  # reject, replace, keep_existing
ONTOFETCH_EXTRACT__DUPLICATE_POLICY="reject"  # reject, first_wins, last_wins

# Throughput (6 vars)
ONTOFETCH_EXTRACT__SPACE_SAFETY_MARGIN=1.10
ONTOFETCH_EXTRACT__PREALLOCATE=true
ONTOFETCH_EXTRACT__COPY_BUFFER_MIN=65536
ONTOFETCH_EXTRACT__COPY_BUFFER_MAX=1048576
ONTOFETCH_EXTRACT__GROUP_FSYNC=32
ONTOFETCH_EXTRACT__MAX_WALL_TIME_SECONDS=120

# Integrity (4 vars)
ONTOFETCH_EXTRACT__HASH_ENABLE=true
ONTOFETCH_EXTRACT__HASH_ALGORITHMS="sha256"
ONTOFETCH_EXTRACT__INCLUDE_GLOBS="*.ttl,*.rdf,*.owl,*.obo"
ONTOFETCH_EXTRACT__EXCLUDE_GLOBS=""
ONTOFETCH_EXTRACT__TIMESTAMPS_MODE="preserve"  # preserve, normalize, source_date_epoch
ONTOFETCH_EXTRACT__TIMESTAMPS_NORMALIZE_TO="archive_mtime"  # archive_mtime or now
```

### Storage Domain (3 env vars)

```bash
ONTOFETCH_STORAGE__ROOT="/home/user/ontologies"
ONTOFETCH_STORAGE__LATEST_NAME="LATEST.json"
ONTOFETCH_STORAGE__URL=""  # Future fsspec backend
```

### DuckDB Domain (5 env vars)

```bash
ONTOFETCH_DB__PATH="/home/user/.catalog/ontofetch.duckdb"
ONTOFETCH_DB__THREADS=8
ONTOFETCH_DB__READONLY=false
ONTOFETCH_DB__WLOCK=true
ONTOFETCH_DB__PARQUET_EVENTS=false
```

### Logging Domain (2 env vars)

```bash
ONTOFETCH_LOG__LEVEL="INFO"  # DEBUG, INFO, WARN, ERROR
ONTOFETCH_LOG__JSON=true
```

### Telemetry Domain (2 env vars)

```bash
ONTOFETCH_TELEMETRY__RUN_ID="550e8400-e29b-41d4-a716-446655440000"  # UUID auto-generated
ONTOFETCH_TELEMETRY__EMIT_EVENTS=true
```

**Total: 63 environment variables** (some for sub-model nesting, some for actual config)

---

## Testing Checklist

### For Each Domain Model

- [ ] Defaults apply correctly
- [ ] Environment variable loading works (`ONTOFETCH_*` parsing)
- [ ] Field validators catch invalid inputs with clear error messages
- [ ] Normalization happens once (not repeatedly)
- [ ] Helper methods work correctly
- [ ] Models are frozen (immutable)
- [ ] Serialization round-trip works (dumps → loads → equal)

### For Root Settings

- [ ] Source precedence: CLI override → config file → .env → env → defaults
- [ ] Config hashing is stable and reproducible
- [ ] Singleton getter caches correctly
- [ ] Cache clears when needed (for tests)
- [ ] All 10 sub-models accessible via `.http`, `.cache`, etc.

### For Integration

- [ ] CLI receives Settings via typer context
- [ ] Planning layer reads from settings
- [ ] Validation layer applies extraction policy from settings
- [ ] Downloader respects HTTP timeouts and rate limits from settings
- [ ] All existing tests pass (zero regressions)

---

## Common Gotchas (Watch Out!)

### ❌ Gotcha 1: Circular Imports

- Don't import planning.py in settings.py at module level
- Use `TYPE_CHECKING` for type hints only
- Lazy imports in validators if needed

### ❌ Gotcha 2: Mutable Defaults

- DON'T: `default=[]` or `default={}`
- DO: `default_factory=list` or `default_factory=dict`

### ❌ Gotcha 3: CSV Parsing in env vars

- Remember: `"1,2,3"` needs to be split in the validator
- Use `pydantic.field_validator` mode="before" to coerce strings → lists

### ❌ Gotcha 4: Path Normalization

- Always normalize paths to absolute + POSIX
- Use `Path(...).expanduser().resolve()` in validators

### ❌ Gotcha 5: Secrets in Config

- API keys, tokens should be excluded from `config_hash()`
- Use `@computed_field` with `exclude=True` or mark field as private

### ❌ Gotcha 6: Settings Mutability

- Use `frozen=True` in `model_config`
- This prevents accidental mutation mid-run

### ❌ Gotcha 7: Testing Isolation

- Always clear singleton cache between tests (`get.cache_clear()`)
- Use `monkeypatch` to override env vars
- Yield in fixtures for cleanup

---

## Implementation Order (Strict)

1. **Phase 5.1** → Domain models (HttpSettings, ..., TelemetrySettings)
2. **Phase 5.2** → Complex domains (SecuritySettings, ..., DuckDBSettings)
3. **Phase 5.3** → Root Settings + loading logic
4. **Phase 5.4** → Integration with existing code (builders, CLI, etc.)
5. **Phase 5.5** → Tests, docs, finalization

**Don't skip phases or reorder.** Each phase depends on the previous one.

---

## How to Run Tests During Development

```bash
# Run only settings tests
python -m pytest tests/ontology_download/test_settings*.py -v

# Run with coverage
python -m pytest tests/ontology_download/test_settings*.py --cov=src/DocsToKG/OntologyDownload/settings --cov-report=term-missing

# Run single test file
python -m pytest tests/ontology_download/test_settings_domain_models.py -v

# Run single test
python -m pytest tests/ontology_download/test_settings_domain_models.py::TestHttpSettings::test_timeout_validation -v
```

---

## Progress Tracking

As you implement each phase:

- [ ] Phase 5.1 complete → Create `PHASE5.1_COMPLETE.md` summary
- [ ] Phase 5.2 complete → Create `PHASE5.2_COMPLETE.md` summary
- [ ] Phase 5.3 complete → Create `PHASE5.3_COMPLETE.md` summary
- [ ] Phase 5.4 complete → Create `PHASE5.4_COMPLETE.md` summary
- [ ] Phase 5.5 complete → Create `PHASE5.5_COMPLETE.md` (final sign-off)

Each summary should include:

- What was built
- Test counts and coverage %
- Known limitations or deferred items
- Next phase kickoff checklist

---

## Key Files to Reference

- **Pydantic v2 Guide**: `DO NOT DELETE docs-instruct/DO NOT DELETE - Refactor review/PythonLibraryReference/pydantic.md` (400 lines)
- **Typer Guide**: `DO NOT DELETE docs-instruct/DO NOT DELETE - Refactor review/PythonLibraryReference/typer.md` (425 lines)
- **Settings Spec**: `DO NOT DELETE docs-instruct/DO NOT DELETE - Refactor review/OntologyDownload/Ontology-config-objects.md` (286 lines)
- **Settings Matrix**: `DO NOT DELETE docs-instruct/DO NOT DELETE - Refactor review/OntologyDownload/Ontology-config-objects-matrix.md` (243 lines)

---

## Contact/Questions

- **For architectural questions**: Refer to `PHASE5_SETTINGS_IMPLEMENTATION_PLAN.md` (sections 2, 5, 9)
- **For field specs**: Refer to `PHASE5_SETTINGS_IMPLEMENTATION_PLAN.md` (section 3)
- **For testing patterns**: Refer to `PHASE5_SETTINGS_IMPLEMENTATION_PLAN.md` (section 6)
- **For env vars**: Refer to this file (above) or full plan (section 3.1-3.7)

---

**Last Updated**: October 20, 2025
**Next Review**: Before starting Phase 5.2
