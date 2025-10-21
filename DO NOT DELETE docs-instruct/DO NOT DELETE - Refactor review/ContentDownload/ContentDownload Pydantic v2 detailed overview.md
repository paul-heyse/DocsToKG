# Pydantic v2 in **ContentDownload** — Consolidated Architecture & Review

> Save as `src/DocsToKG/ContentDownload/ARCHITECTURE_pydantic_v2.md`.
> This is the single source-of-truth document describing **where** Pydantic v2 is used, **how** we use it, **why** certain choices were made (and where we intentionally *don’t* use it), plus an audit checklist and examples.

---

## 0) Scope & intent

Pydantic v2 is the **configuration backbone** of ContentDownload. We use it to:

* Define **strict, typed, composable** config models for every subsystem.
* Merge **file ⊕ env ⊕ CLI** into a single, validated `ContentDownloadConfig`.
* Emit **JSON Schema** for editor/CI validation and versioning.
* Provide **clear, actionable errors** (v2 `ValidationError`) to operators.

We **do not** use Pydantic in hot-path runtime objects (download plans, outcomes, stream results): those stay as **frozen `dataclass` types** for speed and low coupling.

---

## 1) Where Pydantic v2 lives (files & models)

```
src/DocsToKG/ContentDownload/
  config/
    models.py      ← all pydantic v2 models (this is the “contract”)
    loader.py      ← merge file/env/CLI → ContentDownloadConfig
    schema.py      ← optional: export model_json_schema()
  cli/
    app.py         ← uses models via load_config(); provides `print-config`, `validate-config`, `explain`
```

### 1.1 Top-level aggregate: `ContentDownloadConfig`

The **one** object we pass around at bootstrap. It aggregates the rest:

* `http: HttpClientConfig`
* `robots: RobotsPolicy`
* `download: DownloadPolicy`
* `telemetry: TelemetryConfig` (later extended with `otel_*`)
* `hishel: HishelConfig` (cache transport/controller/storage)
* `resolvers: ResolversConfig` (order + per-resolver knobs)
* `orchestrator: OrchestratorConfig` & `queue: QueueConfig` (PR #8)
* `storage: StorageConfig` & `catalog: CatalogConfig` (PR #9)

All models use v2’s `ConfigDict(extra="forbid")` to **reject unknown keys**.

> **Why:** One place to reason about behavior; v2 validation gives fast feedback when config drifts.

---

## 2) The major models (and v2 features used)

Below are representative snippets showing how v2 features are applied. (Field lists elided for brevity.)

```python
# config/models.py (representative)
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal, Dict, List

class HttpClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_agent: str
    mailto: Optional[str] = None
    timeout_connect_s: float = 10.0
    timeout_read_s: float = 60.0
    verify_tls: bool = True
    proxies: Dict[str, str] = Field(default_factory=dict)

class RobotsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    ttl_seconds: int = 3600

class DownloadPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    atomic_write: bool = True
    verify_content_length: bool = True
    chunk_size_bytes: int = 1 << 20
    max_bytes: Optional[int] = None

class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sinks: List[str] = Field(default_factory=lambda: ["csv"])
    csv_path: str = "attempts.csv"
    manifest_path: str = "manifest.jsonl"
    # OTel (PR #7)
    otel_enabled: bool = False
    otel_service_name: str = "DocsToKG-ContentDownload"
    otel_exporter: Literal["otlp-grpc","otlp-http","console","inmemory"] = "otlp-grpc"
    otel_endpoint: Optional[str] = None
    otel_sample_ratio: float = 1.0
    otel_metrics_enabled: bool = True
    otel_resource_attrs: Dict[str, str] = Field(default_factory=dict)

class HishelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    backend: Literal["file","sqlite","redis","s3"] = "file"
    base_path: str = "state/hishel-cache"
    sqlite_path: str = "state/hishel-cache.sqlite"
    redis_url: Optional[str] = None
    s3_bucket: Optional[str] = None
    ttl_seconds: int = 30*24*3600
    check_ttl_every_seconds: int = 600
    force_cache: bool = False
    allow_heuristics: bool = False
    allow_stale: bool = False
    always_revalidate: bool = False
    cache_private: bool = True
    cacheable_methods: List[str] = Field(default_factory=lambda: ["GET"])

class ResolverCommonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # Retry policy
    class RetryPolicy(BaseModel):
        model_config = ConfigDict(extra="forbid")
        retry_statuses: List[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
        max_attempts: int = 4
        base_delay_ms: int = 200
        max_delay_ms: int = 4000
        jitter_ms: int = 100
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

    # Rate limit policy
    class RateLimitPolicy(BaseModel):
        model_config = ConfigDict(extra="forbid")
        capacity: int = 5
        refill_per_sec: float = 1.0
        burst: int = 2
    rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)

    timeout_read_s: Optional[float] = None

class ResolversConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: List[str] = Field(default_factory=lambda: [
        "unpaywall","crossref","arxiv","europe_pmc","core","doaj","s2","landing","wayback"
    ])
    # One entry per resolver; each extends ResolverCommonConfig
    unpaywall: ResolverCommonConfig = Field(default_factory=ResolverCommonConfig)
    crossref: ResolverCommonConfig = Field(default_factory=ResolverCommonConfig)
    # ... others analogous

class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_workers: int = 8
    max_per_resolver: Dict[str, int] = Field(default_factory=dict)
    max_per_host: int = 4
    lease_ttl_seconds: int = 600
    heartbeat_seconds: int = 30
    max_job_attempts: int = 3
    retry_backoff_seconds: int = 60
    jitter_seconds: int = 15

class QueueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = "sqlite"
    path: str = "state/workqueue.sqlite"
    wal_mode: bool = True

class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["fs","s3"] = "fs"
    root_dir: str = "data/docs"
    layout: Literal["policy_path","cas"] = "policy_path"
    cas_prefix: str = "sha256"
    hardlink_dedup: bool = True
    s3_bucket: Optional[str] = None
    s3_prefix: str = "docs/"
    s3_storage_class: str = "STANDARD"

class CatalogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["sqlite"] = "sqlite"
    path: str = "state/catalog.sqlite"
    wal_mode: bool = True
    compute_sha256: bool = True
    verify_on_register: bool = False
    retention_days: int = 0
    orphan_ttl_days: int = 7

class ContentDownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: Optional[str] = None
    http: HttpClientConfig = Field(default_factory=HttpClientConfig)
    robots: RobotsPolicy = Field(default_factory=RobotsPolicy)
    download: DownloadPolicy = Field(default_factory=DownloadPolicy)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    hishel: HishelConfig = Field(default_factory=HishelConfig)
    resolvers: ResolversConfig = Field(default_factory=ResolversConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
```

**v2 features in use:**

* `ConfigDict(extra="forbid")` → **reject unknown keys** (safety).
* `Literal[...]` → restrict enumerations (backends, layouts, exporters).
* Nested `BaseModel`s for hierarchical policy groups.
* `Field(default_factory=...)` for lists/dicts/policies to avoid shared mutable defaults.
* (Optional across models) v2 **strictness** toggles per field or per model if needed.

---

## 3) Loader & precedence (file ⊕ env ⊕ CLI)

**File:** YAML/JSON parsed to Python dict → `ContentDownloadConfig.model_validate(data)`
**Env:** overlay accepts double-underscore nesting (`DTKG_HTTP__USER_AGENT`)
**CLI:** overlay is last-wins (already typed when possible)

```python
# config/loader.py (essentials)
def load_config(path: Optional[str], env_prefix="DTKG_", cli_overrides: Optional[Mapping[str, Any]]=None) -> ContentDownloadConfig:
    data: dict[str, Any] = {}
    if path: data = read_yaml_or_json(path)  # returns dict

    # ENV overlay: DTKG_SECTION__KEY=VALUE → data['section']['key']=coerced(VALUE)
    for k, v in os.environ.items():
        if not k.startswith(env_prefix): continue
        dotted = k[len(env_prefix):].lower().replace("__",".")
        _assign(data, dotted, v)  # supports int/float/bool/list coercion via TypeAdapter

    if cli_overrides:
        data = deep_merge(data, cli_overrides)

    return ContentDownloadConfig.model_validate(data)
```

**v2-specific** considerations:

* Use `TypeAdapter` to coerce env strings to list/bool/ints when `_assign` writes values.
* Fail fast on wrong types → v2 `ValidationError` is surfaced verbatim by `cli/app.py: validate-config`.

---

## 4) JSON Schema & CLI

* `ContentDownloadConfig.model_json_schema()` is exposed in `config/schema.py` and optionally via `cli app.py` → `contentdownload config-schema`.
* `contentdownload print-config` shows the **merged** config as `model_dump(mode="json")`.
* `contentdownload validate-config -c cd.yaml` triggers a v2 validation round-trip.

---

## 5) Where we intentionally **don’t** use Pydantic

Runtime “hot path” objects remain **frozen dataclasses** (not Pydantic models):

```
src/DocsToKG/ContentDownload/api/types.py
  DownloadPlan
  DownloadStreamResult
  DownloadOutcome
  ResolverResult
```

**Why:**

* These are created per URL/attempt and live on the tightest loops (prepare/stream/finalize).
* Dataclasses with `slots=True, frozen=True` give **lower overhead** and help avoid accidental mutation.
* The contract is **intentionally narrow**, and we validate Semantics in tests, not at runtime.

---

## 6) Validations & policies (v2 idioms)

* **Enums with `Literal`**: backends, exporters, layouts. Failing values → instant, clear errors.
* **Numeric ranges**: keep in code (e.g., max must be ≥ base), or add `@field_validator` (v2) for richer checks:

  ```python
  from pydantic import field_validator
  class RetryPolicy(BaseModel):
      # fields...
      @field_validator("max_delay_ms")
      @classmethod
      def _max_ge_base(cls, v, info):
          base = info.data.get("base_delay_ms", 0)
          if v < base: raise ValueError("max_delay_ms must be >= base_delay_ms")
          return v
  ```

* **Cross-field (model) checks** use v2 `@model_validator(mode="after")` if we add invariants later (e.g., S3 keys require bucket).
* **Aliases**: if we need alternative input keys, v2 supports `validation_alias` / `serialization_alias` per field (not used now; easy to add).

---

## 7) How v2 models are used downstream (bootstrap seams)

All bootstrap functions accept the **pydantic models** and **read their fields** to construct runtime dependencies:

* **HTTP & hishel:** `cfg.http`, `cfg.hishel` → build `httpx.Client` + `hishel.CacheTransport`.
* **Rate/Retry per resolver:** `cfg.resolvers.<name>.rate_limit`/`retry` → build per-resolver `RateRetryClient`.
* **Robots/Download policies:** `cfg.robots`, `cfg.download` passed into prepare/stream/finalize.
* **Telemetry:** `cfg.telemetry` configures **CSV/JSONL** sinks and optional **OTel** exporter.
* **Orchestrator/Queue:** `cfg.orchestrator`, `cfg.queue` configure worker pool & persistent queue (PR #8).
* **Storage/Catalog:** `cfg.storage`, `cfg.catalog` drive CAS/policy paths, SHA-256, and catalog registration (PR #9).

**Pattern:** *read from pydantic → construct runtime objects (dataclasses & classes).* We don’t pass Pydantic models deep into hot code paths.

---

## 8) v2 method cheat-sheet (what to expect in code)

| Concern                         | v2 API used                                                 |
| ------------------------------- | ----------------------------------------------------------- |
| Validate Python dict → model    | `ContentDownloadConfig.model_validate(data)`                |
| Dump merged config for logs/CLI | `cfg.model_dump(mode="json")`                               |
| Produce JSON Schema             | `ContentDownloadConfig.model_json_schema()`                 |
| Validate env overlays           | `TypeAdapter(T).validate_python(value)` (in loader helpers) |
| Custom checks                   | `@field_validator`, `@model_validator`                      |
| Config metadata                 | `ConfigDict(extra="forbid")` (per model)                    |

---

## 9) Example end-to-end

**cd.yaml**

```yaml
run_id: "2025-10-21T23:12:45Z-abc123"
http:
  user_agent: "DocsToKG/ContentDownload (+mailto:data@example.org)"
  timeout_connect_s: 10
  timeout_read_s: 60
robots:
  enabled: true
  ttl_seconds: 3600
download:
  atomic_write: true
  verify_content_length: true
  chunk_size_bytes: 1048576
hishel:
  enabled: true
  backend: file
  base_path: state/hishel-cache
resolvers:
  order: ["unpaywall","crossref","landing"]
  unpaywall:
    enabled: true
    retry: { max_attempts: 4, retry_statuses: [429,500,502,503,504], base_delay_ms: 200, max_delay_ms: 4000 }
    rate_limit: { capacity: 5, refill_per_sec: 1.0, burst: 2 }
telemetry:
  sinks: [csv]
  csv_path: logs/attempts.csv
  manifest_path: logs/manifest.jsonl
```

**Env overlay**

```
export DTKG_TELEMETRY__OTEL_ENABLED=true
export DTKG_TELEMETRY__OTEL_EXPORTER=otlp-http
export DTKG_TELEMETRY__OTEL_ENDPOINT=http://localhost:4318
```

**CLI overlay**

```
contentdownload run -c cd.yaml --resolver-order unpaywall,landing --chunk-size 2097152
```

**Result in code**

```python
cfg = load_config("cd.yaml", cli_overrides={"resolvers": {"order": ["unpaywall","landing"]},
                                            "download": {"chunk_size_bytes": 2_097_152}})
# cfg is fully validated; wrong types/keys would have failed before this point
```

---

## 10) Audit checklist (to confirm coverage)

* [ ] **All config-bearing subsystems** have a v2 model under `config/models.py`:

  * [ ] `HttpClientConfig`, `RobotsPolicy`, `DownloadPolicy`
  * [ ] `TelemetryConfig` (+ OTel fields)
  * [ ] `HishelConfig`
  * [ ] `ResolversConfig` + per-resolver `ResolverCommonConfig`
  * [ ] `OrchestratorConfig`, `QueueConfig`
  * [ ] `StorageConfig`, `CatalogConfig`
  * [ ] `ContentDownloadConfig` aggregates all
* [ ] **`extra="forbid"`** on every model (no silent typos).
* [ ] Loader merges **file ⊕ env ⊕ CLI** in that order; env uses `DTKG_` + `__` nesting.
* [ ] `print-config` dumps `cfg.model_dump(mode="json")`.
* [ ] `validate-config` surfaces v2 `ValidationError` messages on bad input.
* [ ] `config-schema` (optional) exports `model_json_schema` for tooling.
* [ ] Bootstrap reads **only** from pydantic models, never mutates them.
* [ ] Hot path (plans/outcomes/streams) remains dataclasses (`api/types.py`).
* [ ] Tests exist for: parsing precedence, unknown keys rejection, env coercion (bool/int/list), CLI overrides.

---

## 11) Known gaps & future niceties

* **Stricter numeric guards** (e.g., `max_delay_ms ≥ base_delay_ms`) via v2 validators across all policies.
* **URL validation** for proxies or base paths via a tiny `HttpUrl` adapter (v2-friendly).
* **Alias support** if we ever rename keys; v2 `validation_alias` avoids breaking older configs.
* **pydantic-settings**: optional swap-in for loader if we want built-in env handling, though our current loader is simple and fast.
* **Config hashing**: we already hash the merged config for provenance in telemetry; consider adding `cfg_hash` to the model dump for persistent runs.

---

## 12) Migration notes (v1 → v2 mental map)

* v1 `dict()/json()` → v2 `model_dump()/model_dump_json()`
* v1 `parse_obj()` → v2 `model_validate()`
* v1 `schema()` → v2 `model_json_schema()`
* v1 `BaseSettings` (in core) → v2 lives in **`pydantic-settings`** (we used a custom loader instead)

---

## 13) Why this division of labor works

* Pydantic v2 gives us **defensive, explicit config** with excellent error messages and schema tooling.
* Dataclasses keep the hot loop **lean and predictable**.
* The boundary between them is **bootstrap**: read once, construct dependencies, pass **plain typed objects** to execution.

---

### TL;DR

* **Where**: `config/models.py` + `config/loader.py` (and CLI helpers).
* **How**: v2 models with `extra="forbid"`, `Literal[...]`, nested policy groups, merged by loader; JSON schema & validate CLI.
* **Not used**: hot-path runtime types (keep dataclasses).
* **Outcome**: A robust, versionable configuration surface that’s easy for both humans and agents to reason about.

If you want, I can generate **schema snapshots** (the JSON of `model_json_schema()`) and a short **“breaking change” policy** for evolving config fields safely over time.
