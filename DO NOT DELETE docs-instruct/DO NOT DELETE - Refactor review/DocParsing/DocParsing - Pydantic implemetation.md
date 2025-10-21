Awesome — let’s tackle the **next item (P2)** you asked for: **Config, API shape, and maintainability**. Below is a detailed, code-level implementation plan you can hand to a PR queue. It’s split into mergeable steps with clear acceptance criteria and tests.

---

# P2 Objectives

1. Move all ContentDownload configuration to **Pydantic v2** (strict, typed, env-aware).
2. **Modularize resolvers** and load them via a **registry** with explicit ordering/enable/override controls.
3. **Stabilize the download API surface** (types & contracts that helpers/pipeline share).
4. **Polish CLI** to assemble an “effective config” (file + env + CLI) and expose introspection commands.

---

# Directory Changes (at a glance)

```
src/DocsToKG/ContentDownload/
  config/
    __init__.py
    models.py            # Pydantic v2 config models
    loader.py            # merges file/env/CLI -> models
    schema.py            # json schema export (optional)
  resolvers/
    __init__.py          # registry + @register decorator
    base.py              # Resolver Protocol / base class
    unpaywall.py
    crossref.py
    landing.py
    arxiv.py
    europe_pmc.py
    core.py
    doaj.py
    s2.py
    wayback.py
  api/
    __init__.py
    types.py             # DownloadPlan/Outcome/... (unified)
  cli/
    __init__.py
    app.py               # Typer commands
```

---

# Step 1 — Pydantic v2 Config Models

## 1.1 Create config models

```python
# src/DocsToKG/ContentDownload/config/models.py
from __future__ import annotations
from typing import Annotated, Literal, Optional, List, Dict
from datetime import timedelta
from pydantic import BaseModel, Field, ConfigDict, HttpUrl, TypeAdapter

# Shared primitives
class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
    retry_statuses: List[int] = Field(default=[429, 500, 502, 503, 504])
    max_attempts: int = 4
    base_delay_ms: int = 200
    max_delay_ms: int = 4000
    jitter_ms: int = 100

class BackoffPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["exponential", "constant"] = "exponential"
    factor: float = 2.0

class RateLimitPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capacity: int = 5                 # tokens
    refill_per_sec: float = 1.0       # tokens/sec
    burst: int = 2

class RobotsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    ttl_seconds: int = 3600

class DownloadPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    atomic_write: bool = True
    verify_content_length: bool = True
    chunk_size_bytes: int = 1 << 20   # 1 MiB
    max_bytes: Optional[int] = None   # content-size cap (None = unlimited)

class HttpClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_agent: str = "DocsToKG/ContentDownload"
    mailto: Optional[str] = None        # appended to UAs where appropriate
    timeout_connect_s: float = 10.0
    timeout_read_s: float = 60.0
    verify_tls: bool = True
    proxies: Dict[str, str] = Field(default_factory=dict)  # {"https": "...", "http": "..."}

class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sinks: List[str] = Field(default_factory=lambda: ["csv"])  # future: "otlp", "console"
    csv_path: str = "attempts.csv"
    manifest_path: str = "manifest.jsonl"

# Resolver-specific config blocks (keep minimal; override per resolver)
class ResolverCommonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)
    timeout_read_s: Optional[float] = None  # override of HttpClientConfig

class UnpaywallConfig(ResolverCommonConfig):
    email: Optional[str] = None

class CrossrefConfig(ResolverCommonConfig):
    mailto: Optional[str] = None

# Top-level
class ResolversConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: List[str] = Field(default_factory=lambda: [
        "unpaywall", "crossref", "arxiv", "europe_pmc", "core", "doaj",
        "s2", "landing", "wayback"
    ])
    unpaywall: UnpaywallConfig = Field(default_factory=UnpaywallConfig)
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig)
    # ... add others analogously with ResolverCommonConfig defaults

class ContentDownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    run_id: Optional[str] = None
    http: HttpClientConfig = Field(default_factory=HttpClientConfig)
    robots: RobotsPolicy = Field(default_factory=RobotsPolicy)
    download: DownloadPolicy = Field(default_factory=DownloadPolicy)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    resolvers: ResolversConfig = Field(default_factory=ResolversConfig)
```

**Design choices**

* `extra="forbid"` everywhere (typos become immediate errors).
* Use nested models so overrides are clear and scoped.
* Keep resolver-specific configs thin; most knobs live in `ResolverCommonConfig`.

## 1.2 Loader (file/env/CLI precedence)

```python
# src/DocsToKG/ContentDownload/config/loader.py
from __future__ import annotations
import json, os, pathlib
from typing import Any, Mapping, Optional
from pydantic import TypeAdapter
from .models import ContentDownloadConfig

def _read_yaml_or_json(path: str) -> Mapping[str, Any]:
    p = pathlib.Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("Install PyYAML to load YAML configs") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)

def load_config(
    path: Optional[str],
    env_prefix: str = "DTKG_",
    cli_overrides: Optional[Mapping[str, Any]] = None,
) -> ContentDownloadConfig:
    data: dict[str, Any] = {}
    if path:
        data = dict(_read_yaml_or_json(path))

    # env overlay (flat dot-notation to nested dict; e.g., DTKG_HTTP__USER_AGENT)
    for k, v in os.environ.items():
        if not k.startswith(env_prefix):
            continue
        dotted = k[len(env_prefix):].lower().replace("__", ".")  # HTTP__USER_AGENT -> http.user_agent
        _assign(data, dotted, v)  # implement: split on '.'; coerce types best-effort

    # CLI overrides win last (already parsed types)
    if cli_overrides:
        data = deep_merge(data, cli_overrides)  # implement shallow recursive merge

    return ContentDownloadConfig.model_validate(data)
```

**Precedence:** *file* < *env* < *CLI*.
**Env pattern:** `DTKG_HTTP__USER_AGENT="..."`, `DTKG_RESOLVERS__ORDER='["arxiv","landing"]'`.

## 1.3 JSON Schema (optional but handy for docs)

```python
# src/DocsToKG/ContentDownload/config/schema.py
from .models import ContentDownloadConfig
def export_json_schema() -> dict:
    return ContentDownloadConfig.model_json_schema()
```

### Acceptance

* Invalid keys are rejected with actionable error messages.
* You can load from JSON/YAML, override with env, and final overrides via CLI.
* Provide `--print-config`/`--validate-config` (see CLI in Step 4).

---

# Step 2 — Resolver Registry & Modularization

## 2.1 Base protocol & register decorator

```python
# src/DocsToKG/ContentDownload/resolvers/base.py
from __future__ import annotations
from typing import Protocol, ClassVar, Optional
from ..api.types import ResolverResult, Artifact, DownloadContext
from ..telemetry import AttemptSink

class Resolver(Protocol):
    name: ClassVar[str]
    priority: ClassVar[int]  # optional, default 50
    def resolve(
        self,
        artifact: Artifact,
        session,
        ctx: DownloadContext,
        telemetry: Optional[AttemptSink],
        run_id: Optional[str],
    ) -> ResolverResult: ...
```

```python
# src/DocsToKG/ContentDownload/resolvers/__init__.py
from __future__ import annotations
from typing import Callable, Dict, List, Type, Optional
from .base import Resolver

_REGISTRY: Dict[str, Type[Resolver]] = {}

def register(name: str):
    def deco(cls: Type[Resolver]) -> Type[Resolver]:
        cls.name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = cls
        return cls
    return deco

def get_registry() -> Dict[str, Type[Resolver]]:
    return dict(_REGISTRY)

def build_resolvers(order: List[str], config) -> List[Resolver]:
    out: List[Resolver] = []
    for name in order:
        cls = _REGISTRY.get(name)
        if not cls:
            raise ValueError(f"Unknown resolver '{name}'")
        # pull per-resolver config: config.resolvers.<name>
        rcfg = getattr(config.resolvers, name)
        if not rcfg.enabled:
            continue
        # Prefer a classmethod factory if provided:
        if hasattr(cls, "from_config"):
            inst = cls.from_config(rcfg, config)
        else:
            inst = cls()   # type: ignore[call-arg]
        out.append(inst)
    return out
```

## 2.2 Move resolvers into modules and register

```python
# src/DocsToKG/ContentDownload/resolvers/unpaywall.py
from . import register
from .base import Resolver
@register("unpaywall")
class UnpaywallResolver:
    priority = 10
    # optional: @classmethod def from_config(cls, rcfg, root_cfg): ...
    def resolve(...): ...
```

Repeat for `crossref.py`, `landing.py`, etc.
**Pipeline** now asks the registry to materialize an ordered list using `config.resolvers.order`.

### Acceptance

* Resolvers are importable as modules; registry builds instances using `order` and `enabled` flags.
* Unknown resolver names fail fast with a helpful error.
* Existing order-override flags continue to work by updating `config.resolvers.order`.

---

# Step 3 — Stabilize the Download API Surface

Create a single source of truth for the types used between helpers and pipeline.

```python
# src/DocsToKG/ContentDownload/api/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass(frozen=True)
class DownloadPlan:
    url: str
    resolver_name: str
    referer: Optional[str] = None
    expected_mime: Optional[str] = None
    # future: conditional headers, ETag, IMS, max_bytes override

@dataclass(frozen=True)
class DownloadStreamResult:
    path_tmp: str
    bytes_written: int
    http_status: int
    content_type: Optional[str]

@dataclass(frozen=True)
class DownloadOutcome:
    ok: bool
    path: Optional[str]
    classification: str   # "success" | "skip" | "error"
    reason: Optional[str] = None
    meta: Dict[str, Any] = None

# Optional: common return container for resolvers
@dataclass(frozen=True)
class ResolverResult:
    plans: List[DownloadPlan]          # often one; some resolvers may yield many
    notes: Dict[str, Any] = None
```

**Action:** Update download helpers and pipeline to use these types consistently (rename existing equivalents to these or alias them).

### Acceptance

* All helpers accept/return these types.
* Pipeline → helpers use only these types (no ad-hoc dicts).
* Telemetry attaches to these fields (resolver_name/url/reason).

---

# Step 4 — CLI polish (Typer)

Add an app with 3 subcommands:

```python
# src/DocsToKG/ContentDownload/cli/app.py
import typer
from ..config.loader import load_config
from ..config.models import ContentDownloadConfig
from ..resolvers import get_registry, build_resolvers

app = typer.Typer(help="DocsToKG ContentDownload")

@app.command("run")
def run(
    config: str = typer.Option(None, "--config", "-c"),
    resolver_order: str = typer.Option(None, "--resolver-order",
        help='Comma list, e.g., "unpaywall,crossref,landing"'),
    no_robots: bool = typer.Option(False, "--no-robots"),
    no_atomic: bool = typer.Option(False, "--no-atomic-write"),
    chunk_size: int = typer.Option(None, "--chunk-size"),
):
    overrides = {}
    if resolver_order:
        overrides = {"resolvers": {"order": resolver_order.split(",")}}
    if no_robots:
        overrides.setdefault("robots", {})["enabled"] = False
    if no_atomic:
        overrides.setdefault("download", {})["atomic_write"] = False
    if chunk_size is not None:
        overrides.setdefault("download", {})["chunk_size_bytes"] = chunk_size
    cfg = load_config(config, cli_overrides=overrides)
    # assemble pipeline & run (existing entry point)

@app.command("print-config")
def print_config(config: str = typer.Option(None, "--config", "-c")):
    cfg = load_config(config)
    import json
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2))

@app.command("validate-config")
def validate_config(config: str):
    _ = load_config(config)
    typer.echo("OK")

@app.command("explain")
def explain(config: str = typer.Option(None, "--config", "-c")):
    cfg = load_config(config)
    reg = get_registry()
    typer.echo("Order: " + ", ".join(cfg.resolvers.order))
    missing = [n for n in cfg.resolvers.order if n not in reg]
    if missing:
        typer.echo(f"Missing resolver(s): {missing}")
```

**Expose:** `--config`, `--resolver-order`, `--no-robots`, `--no-atomic-write`, `--chunk-size`.
**Future:** `--only`, `--max-per-resolver` (easy to add once config is in).

### Acceptance

* `print-config` shows merged effective config (file/env/CLI).
* `validate-config` returns non-zero on bad configs.
* `explain` shows resolver order and missing modules.

---

# Step 5 — Wire Config into Pipeline/Runner

* Replace legacy dataclass config objects with `ContentDownloadConfig`.
* At pipeline construction:

  * Build `requests.Session` (or your HTTP client) using `cfg.http` (timeouts, agent, proxies, TLS).
  * Build resolver list: `build_resolvers(cfg.resolvers.order, cfg)`.
  * Pass `cfg.download` flags to download helpers (atomic write, chunk size, max_bytes).
  * Pass `cfg.robots` to robots cache (enabled, ttl).
  * Build telemetry/manifest sinks from `cfg.telemetry`.

**Guardrails**

* Keep backwards-compatible defaults so existing tests don’t change.
* Keep dependency injection explicit (pass `cfg` fields; no globals).

### Acceptance

* Runner accepts either `--config` or programmatic `ContentDownloadConfig` and produces the same outcomes with legacy defaults.

---

# Step 6 — Tests

## 6.1 Config parsing & precedence

* YAML vs JSON both parse; unknown key in any block raises.
* Env override wins over file; CLI wins over env.
* Types are coerced correctly (ints/bools/lists).

## 6.2 Registry loading

* Unknown resolver in order → fails with helpful error.
* Disabled resolver is skipped.
* `from_config` paths get per-resolver overrides (e.g., `timeout_read_s`).

## 6.3 API types

* Helpers accept/return `DownloadPlan/DownloadOutcome` and pipeline composes them.
* Telemetry calls reference `plan.resolver_name`, `plan.url`, not ad-hoc fields.

## 6.4 CLI

* `print-config` prints merged config; `validate-config` returns 0; `explain` lists order and missing names.

---

# Step 7 — Migration & Backwards Compatibility

* Provide a compatibility shim in old config paths:

  * A factory `from_legacy_dataclass()` that creates `ContentDownloadConfig` from the old dataclasses.
  * Deprecation warnings if legacy constructors are used.
* Keep resolver behavior identical; only module locations change.
* Document the new dotted env syntax: `DTKG_RESOLVERS__ORDER`, etc.

---

# Step 8 — Observability Hooks (bonus, low effort)

* Emit a single **config_loaded** telemetry event at run start with:

  * sanitized UA, enabled resolvers, chunk size, robots.enabled, run_id.
* Include `config_hash` (e.g., SHA256 of `model_dump_json(sort_keys=True)`) in that event so runs are attributable to exact config.

---

# PR Breakdown (mergeable increments)

1. **Config models + loader** (no pipeline changes yet)

   * Add models, loader, schema; unit tests for parsing & precedence.

2. **Resolver modularization + registry**

   * Move resolvers to modules; add `@register` usage; unit tests for order/enable.

3. **API types stabilization**

   * Introduce `api/types.py`; refactor helpers/pipeline to use them; green tests.

4. **CLI polish**

   * Add Typer app; `print-config`, `validate-config`, `explain`; tests.

5. **Wire config end-to-end**

   * Replace legacy config injection in pipeline/runner with `ContentDownloadConfig`.

---

# Acceptance Checklist

* [ ] Config is fully Pydantic v2, strict, and validates deeply.
* [ ] Effective config = file ⊕ env ⊕ CLI (in that order).
* [ ] Resolvers live in `resolvers/`, are loaded by name via registry, and respect `order` and `enabled`.
* [ ] Download helpers and pipeline share one set of types (`DownloadPlan/Outcome/...`).
* [ ] CLI exposes `run`, `print-config`, `validate-config`, `explain`.
* [ ] All existing tests green; new tests for config/registry/API/CLI pass.
* [ ] No reliance on globals; all dependencies injected explicitly.

---

If you want, I can turn this into **ready-to-apply patch files** for PR 1 (models + loader) and PR 2 (registry + module split), including unit tests and minimal stubs for each resolver file to keep diffs focused.
