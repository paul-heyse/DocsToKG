Perfect — here’s a **code-level plan for PR #3: API Types Stabilization** (from the P2 breakdown). It makes the download/pipeline contracts crystal clear, removes dict-shaped ad-hoc payloads, and gives you a single source of truth for types the whole subsystem shares.

---

# PR #3 Objectives

1. Introduce a **canonical types module** for the ContentDownload API: `DownloadPlan`, `DownloadStreamResult`, `DownloadOutcome`, `ResolverResult`.
2. Refactor download helpers, resolvers, and the pipeline to **use those types exclusively**.
3. Provide **stable token vocabularies** (Literal/Enum) for `classification`, `status`, `reason`.
4. Add **adapters** for any legacy results (minimal shim so we don’t break PR #1/#2 work).
5. Ship **tests** and minimal docs; keep behavior identical.

Non-goals: changing algorithms, move/rename telemetry fields, or altering config. This PR only locks **shapes**, not behavior.

---

# 0) File scaffolding (new)

```
src/DocsToKG/ContentDownload/
  api/
    __init__.py
    types.py          # NEW: canonical API types (dataclasses + Literals/Enums)
  ...
```

---

# 1) Canonical types

Create `src/DocsToKG/ContentDownload/api/types.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

# ----- Stable token vocabularies (public contract) -----
from typing import Literal

OutcomeClass = Literal["success", "skip", "error"]

AttemptStatus = Literal[
    "http-head", "http-get", "http-200", "http-304",
    "robots-fetch", "robots-disallowed",
    "retry", "size-mismatch", "content-policy-skip", "download-error"
]

ReasonCode = Literal[
    "ok", "not-modified", "retry-after", "backoff",
    "robots", "policy-type", "policy-size",
    "timeout", "conn-error", "tls-error",
    "too-large", "unexpected-ct", "size-mismatch"
]

# ----- Core API payloads used between resolvers, pipeline, and execution -----

@dataclass(frozen=True, slots=True)
class DownloadPlan:
    url: str
    resolver_name: str
    referer: Optional[str] = None
    expected_mime: Optional[str] = None
    # Future-proofing (optional headers etc.)
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    max_bytes_override: Optional[int] = None

@dataclass(frozen=True, slots=True)
class DownloadStreamResult:
    # Path to the temp file (pre-rename)
    path_tmp: str
    bytes_written: int
    http_status: int
    content_type: Optional[str]

@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    ok: bool
    classification: OutcomeClass          # "success" | "skip" | "error"
    path: Optional[str] = None            # final path (post-rename) if any
    reason: Optional[ReasonCode] = None
    # Attach any small, structured bits (don’t dump huge blobs here)
    meta: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ResolverResult:
    # Typically a single plan; sometimes more (e.g., multiple mirrors)
    plans: Sequence[DownloadPlan]
    # Optional notes for telemetry/diagnostics (small key/values)
    notes: Mapping[str, Any] = field(default_factory=dict)
```

**Guidelines baked in:**

* **Frozen + slots** → immutable, memory-tight, cheap to copy.
* **Literal token types** → consumers can’t invent new values by accident; helps mypy/pyright.
* **Meta** maps are allowed but intended to stay tiny.

Export in `api/__init__.py`:

```python
from .types import (  # noqa: F401
    DownloadPlan, DownloadStreamResult, DownloadOutcome, ResolverResult,
    OutcomeClass, AttemptStatus, ReasonCode,
)
```

---

# 2) Adapters for legacy shapes (compat layer)

If you already have interim types like `DownloadPreflightResult`, `FinalizationResult`, etc., add **one tiny adapter file** so old call sites compile while we refactor.

`src/DocsToKG/ContentDownload/api/adapters.py`:

```python
from __future__ import annotations
from typing import Any, Mapping, Optional
from .types import DownloadPlan, DownloadStreamResult, DownloadOutcome, ResolverResult, OutcomeClass, ReasonCode

# Example shim (adjust to your legacy names)
def to_download_plan(url: str, resolver_name: str, *, referer: Optional[str]=None,
                     expected_mime: Optional[str]=None, **_: Any) -> DownloadPlan:
    return DownloadPlan(url=url, resolver_name=resolver_name, referer=referer, expected_mime=expected_mime)

def to_outcome_success(path: str, **meta: Any) -> DownloadOutcome:
    return DownloadOutcome(ok=True, classification="success", path=path, meta=meta)

def to_outcome_skip(reason: ReasonCode, **meta: Any) -> DownloadOutcome:
    return DownloadOutcome(ok=False, classification="skip", reason=reason, meta=meta)

def to_outcome_error(reason: ReasonCode, **meta: Any) -> DownloadOutcome:
    return DownloadOutcome(ok=False, classification="error", reason=reason, meta=meta)
```

Mark any legacy classes as **deprecated aliases** if they must stay public for a release (optional):

```python
# src/DocsToKG/ContentDownload/api/legacy.py
# from .types import DownloadPlan as DownloadPreflightPlan  # example alias
```

---

# 3) Signature refactors (mechanical, low-risk)

## 3.1 Download execution helpers

Change signatures to consume/produce the canonical types.

```python
# src/DocsToKG/ContentDownload/download_execution.py (example names)
from .api.types import DownloadPlan, DownloadStreamResult, DownloadOutcome
from .telemetry import AttemptSink  # PR#1 introduced Protocol
from typing import Optional

def prepare_candidate_download(
    plan: DownloadPlan, *,
    telemetry: Optional[AttemptSink] = None,
    run_id: Optional[str] = None,
) -> DownloadPlan:
    """Return the (maybe adjusted) plan; or raise Skip/Error by returning an Outcome?"""
    # Keep this simple: return plan or raise a local SkipError/Error that
    # the pipeline translates into DownloadOutcome (skip/error) to keep
    # the function’s return type stable.
    return plan

def stream_candidate_payload(
    plan: DownloadPlan, *,
    telemetry: Optional[AttemptSink] = None,
    run_id: Optional[str] = None,
) -> DownloadStreamResult:
    ...

def finalize_candidate_download(
    plan: DownloadPlan,
    stream: DownloadStreamResult, *,
    telemetry: Optional[AttemptSink] = None,
    run_id: Optional[str] = None,
) -> DownloadOutcome:
    ...
```

> If your existing `prepare_*` returns a special preflight result, replace it with either:
> (a) **returned plan** (pass-through), or
> (b) **raising small local exceptions** (`SkipDownload(reason)`, `DownloadError(reason)`) that the pipeline catches and converts to `DownloadOutcome`. This keeps type signatures **pure** and simplifies tests.

## 3.2 Resolver interface

Make resolvers return `ResolverResult`. (PR #2 provided `resolvers/base.py`; update there.)

```python
# src/DocsToKG/ContentDownload/resolvers/base.py (Protocol)
from ..api.types import ResolverResult, DownloadPlan
class Resolver(Protocol):
    name: str
    def resolve(...) -> ResolverResult: ...
```

Then in each resolver module (e.g., `resolvers/unpaywall.py`):

```python
from ..api.types import ResolverResult, DownloadPlan
def resolve(...)-> ResolverResult:
    ...
    if not url:
        return ResolverResult(plans=())  # empty = nothing to do
    return ResolverResult(plans=[DownloadPlan(url=url, resolver_name="unpaywall")])
```

## 3.3 Pipeline orchestration

Change pipeline internals to consume these types:

```python
for resolver in resolvers:
    rres = resolver.resolve(artifact, session, ctx, telemetry, run_id)
    for plan in rres.plans:
        # Preflight (may raise skip/error for robots/policy)
        adj_plan = prepare_candidate_download(plan, telemetry=telemetry, run_id=run_id)
        # Stream
        stream = stream_candidate_payload(adj_plan, telemetry=telemetry, run_id=run_id)
        # Finalize
        outcome = finalize_candidate_download(adj_plan, stream, telemetry=telemetry, run_id=run_id)
        record_pipeline_result(..., outcome=outcome, plan=adj_plan, ...)   # existing telemetry helper
        if outcome.ok:
            return outcome  # stop once we have a good artifact
# nothing succeeded
return DownloadOutcome(ok=False, classification="error", reason="download-error")
```

---

# 4) Telemetry call sites (light touch)

PR #1 already added attempt emission. Make sure any `log_attempt(...)` pulls **resolver name** and **url** from `DownloadPlan` rather than ad-hoc params:

```python
_emit(telemetry,
      run_id=run_id, resolver=plan.resolver_name, url=plan.url,
      verb="HEAD", status="http-head", ...)
```

No telemetry shape changes are introduced by this PR — only **where the data comes from**.

---

# 5) Tests

## 5.1 New unit tests for the types module

`tests/contentdownload/test_api_types.py`

* Instantiate each dataclass; assert immutability (setting an attribute raises `FrozenInstanceError`).
* mypy/pyright (if in CI): simple typed function that only accepts `OutcomeClass`/`ReasonCode` fails on random strings.
* `asdict` serialization is stable (optional snapshot).

## 5.2 Resolver contract tests

`tests/contentdownload/test_resolver_contract.py`

* A minimal fake resolver returns `ResolverResult(plans=[DownloadPlan(...)])`.
* Pipeline picks up `resolver_name` and `url` from the plan (assert used in telemetry events).
* Empty `plans` means the pipeline tries the next resolver.

## 5.3 Download execution contract tests

* **Happy path**: `prepare` passthrough, `stream` yields `DownloadStreamResult`, `finalize` returns success `DownloadOutcome`.
* **Skip path**: `prepare` raises `SkipDownload("robots")`; pipeline converts to `DownloadOutcome(classification="skip", reason="robots")`.
* **Error path**: `finalize` returns error `DownloadOutcome`; pipeline records final manifest once.

## 5.4 Backwards-compatible adapters

If any legacy helpers are still referenced in tests, add a small compatibility test that simulates a legacy result → adapter → canonical outcome.

---

# 6) Documentation & dev-XP

* **ARCHITECTURE.md**: Add a brief “API Types” section with a one-screen summary and a diagram showing `ResolverResult → DownloadPlan → (prepare/stream/finalize) → DownloadOutcome`.
* **Docstrings**: Ensure each dataclass docstring describes its field semantics and stability.
* **Changelog**: “PR #3 — Canonical API types introduced; legacy helpers shimmable via `api.adapters` for one minor release.”

---

# 7) Migration strategy (low churn)

1. Land `api/types.py` and **imports only** (no behavior change).
2. Convert **download_execution** to canonical types (smallest surface).
3. Convert **pipeline**.
4. Convert **resolvers**.
5. Remove/alias legacy types; keep adapters for one release behind a deprecation warning (optional).

> This order keeps the code compiling after each commit and makes review diff-friendly.

---

# 8) Acceptance criteria

* All download helpers, resolvers, and pipeline **only** use `DownloadPlan`, `DownloadStreamResult`, `DownloadOutcome`, `ResolverResult`.
* No dict-shaped payloads remain in these paths.
* Telemetry call sites derive data from `DownloadPlan` consistently.
* Tests passing: new unit tests + existing integration tests unchanged (behavior is identical).
* (Optional) mypy/pyright run clean on `src/DocsToKG/ContentDownload/api`.

---

# 9) Risk & mitigations

* **Hidden legacy imports**: Use a repo-wide search for type names (e.g., `Download*Result`) and add temporary aliases in `api/legacy.py`.
* **Third-party agent scripts** (if any) importing old names: ship one minor-release deprecation window with alias + `warnings.warn`.
* **Drift in status/reason strings**: Literals make drift compile-visible; if a new token is needed, add it to the union in one PR.

---

# 10) Suggested commit structure

1. `feat(types): add canonical API types (DownloadPlan/Outcome/…) + exports`
2. `refactor(execution): use canonical types in prepare/stream/finalize`
3. `refactor(pipeline): orchestrate with canonical types`
4. `refactor(resolvers): return ResolverResult(plans=[DownloadPlan])`
5. `test: add unit tests for types + resolver contract`
6. `docs: ARCHITECTURE.md API types notes`
7. `chore: add temporary legacy adapters (if required)`

---

If you want, I can turn this plan into **ready-to-apply patches** for steps (1)–(3) with stubbed adapters so you get a compiling tree immediately, then follow with resolvers in a second commit for an easy review.
