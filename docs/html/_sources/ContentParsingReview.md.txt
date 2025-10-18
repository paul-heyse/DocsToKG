Absolutely — here’s a **high-fidelity narrative** for each change to `src/DocsToKG/DocParsing`, written so an engineer (or reviewer) understands the *why*, the *shape* of the solution, its *blast radius*, and *how we’ll know it worked*. No patches here—this is the “design intent + implementation notes” you can hand off.

---

# P0 — Correctness & run-stoppers (do these first)

## 1) Decompose the “god module” (`core.py`)

**Problem.** `core.py` mixes unrelated concerns (HTTP helpers, CLI glue, manifest logging, JSONL I/O, config types). This creates import-time side effects, brittle couplings (e.g., logging depends on HTTP globals), and makes targeted testing painful.

**Change (what).** Split `core.py` into focused modules:

* `io.py` — JSONL readers/writers, atomic file writes, path utilities.
* `config.py` — `StageConfigBase` and stage configs; env/file/CLI overlay; `finalize()` invariants.
* `env.py` (or `paths.py`) — data_root detection + path builders **without touching the filesystem** at import.
* `logging.py` — structured `log_event`, manifest formatting, summary composition.

**How it works.** Each module exposes a small API; `core.py` can temporarily re-export to keep imports stable during migration. The CLI imports narrow slices, avoiding heavy imports when not needed.

**Why it reduces risk.**

* No accidental behavior during module import.
* Unit tests can focus on one surface (e.g., JSONL streaming) without spinning up everything else.
* Clear ownership (bugs don’t hide “somewhere in core.py”).

**Blast radius.** Medium: call sites importing `core` symbols will need to swap to new modules (or rely on temporary re-exports).

**Definition of done.** `core.py` ≤20% of current size; tests run with only `io.py` imported; no unexpected network/filesystem I/O on import.

---

## 2) Make JSONL **streaming** (no full-file materialization)

**Problem.** `jsonl_load` reads whole files into memory; large corpora cause spikes and GC churn even though we already have internal iterators.

**Change (what).** Publicly expose a streaming iterator (and an optional batched iterator). Deprecate eager loaders in stage code paths.

**How it works.** Callers iterate records (or batches) as generators. Error handling skips malformed lines without aborting the run; backpressure is natural (downstream speed controls memory).

**Why it reduces risk.**

* Predictable memory even for huge inputs.
* Long-running jobs don’t degrade over time.

**Blast radius.** Low/medium: places calling `list(jsonl_load(...))` will switch to loops; tests update to iterate.

**Definition of done.** Chunking/embedding stages process identical inputs with flat memory usage profile; fuzzer test with junk lines doesn’t crash.

---

## 3) Harden `resolve_hash_algorithm` (env override guard)

**Problem.** A typo like `DOCSTOKG_HASH_ALG=sha-1` bubbles into `hashlib.new`, failing late with opaque stack traces.

**Change (what).** Validate env override against `hashlib.algorithms_available`. If unknown/NaN/blank, log a warning and fall back to a safe default (e.g., `sha256`).

**How it works.** All hasher construction flows through a single helper that enforces the guard.

**Why it reduces risk.**

* Fail-safe behavior; production jobs don’t die from simple typos.
* Observability: the warning tells ops exactly what happened.

**Blast radius.** Very low; centralized.

**Definition of done.** Unit test: bogus env → warning + `sha256`; valid env → selected algo.

---

## 4) Recompute `in_dir` / `doctags_dir` on `finalize()` (no import-time caching)

**Problem.** Defaults are captured when modules import. Later environment/CLI updates to `data_root` don’t propagate, so stage runs read/write the wrong directories.

**Change (what).** Treat `data_root` as the single source of truth. In each config’s `finalize()`, compute derived paths **only if the user didn’t explicitly set them**.

**How it works.** `finalize()` resolves: `data_root := CLI > env > default`. If `in_dir/doctags_dir` are `None`, set them relative to the resolved `data_root`; otherwise leave them as user-overridden.

**Why it reduces risk.**

* “Late” env/CLI overrides always work.
* No invisible shadowing from import-time constants.

**Blast radius.** Low; only path resolution code.

**Definition of done.** Changing `DOCSTOKG_DATA_ROOT` between runs changes actual IO locations without restarting the process.

---

## 5) Move configuration **invariants into `finalize()`**

**Problem.** CLI guards exist, but non-CLI code paths (env/config file) can feed invalid values (e.g., `min_tokens > max_tokens`, shard index out of bounds).

**Change (what).** Validate invariants in `finalize()` for each stage config:

* Token windows: non-negative; `min ≤ max`.
* Shards: `0 ≤ shard_index < shard_count`.
* Parallelism: integers ≥1 where required (e.g., `files_parallel`).
* Any derived fields computed after validation.

**How it works.** Uniform, early failure with clear error messages regardless of entry point.

**Why it reduces risk.**

* Fewer “works on CLI; breaks in batch” incidents.
* Errors point to the cause (not downstream stack traces).

**Blast radius.** Low; validation only.

**Definition of done.** Config file with bad values fails fast; equivalent CLI runs behave identically.

---

# P1 — Performance & memory

## 6) Remove redundant `chunk_manifest_index` loads

**Problem.** The index is loaded twice back-to-back in resume mode; it costs an extra filesystem scan for no benefit.

**Change (what).** Load once, reuse. If there’s a fast path (e.g., content hash unchanged), branch early.

**How it works.** Store the loaded index in a local variable and pass it along the call chain.

**Why it reduces risk.**

* Fewer disk hits; faster cold starts.
* Clearer control flow in resume logic.

**Blast radius.** Very low.

**Definition of done.** Trace shows a single load per run; timings improve marginally on large manifests.

---

## 7) Avoid re-tokenizing **unchanged** records in `coalesce_small_runs`

**Problem.** The final re-tokenization pass hits every output record, even those never merged. With large docs and HF tokenizers, this dominates runtime.

**Change (what).** Track which records actually changed during merge decisions; only re-tokenize those. Cache token counts as you merge.

**How it works.** The merge loop maintains token counts and a `changed` set. Post-processing only touches changed indices.

**Why it reduces risk.**

* Big runtime reduction on large files.
* Identical output; fewer tokenizer calls.

**Blast radius.** Low; confined to the coalesce step.

**Definition of done.** Profiling shows significant drop in tokenizer calls and wall-clock time for merge-heavy docs.

---

## 8) Make `tqdm` truly **optional**

**Problem.** The module docstring says `tqdm` is optional, but it’s imported unconditionally—so “light” environments fail to import.

**Change (what).** Wrap the import in `try/except`; provide a no-op fallback function.

**How it works.** In dev, you see progress bars; in slim environments, the code still runs, just without progress output.

**Why it reduces risk.**

* No unnecessary dependency failures in minimal containers.
* More flexible deployment targets.

**Blast radius.** Very low.

**Definition of done.** Uninstall `tqdm`; module still imports and runs.

---

## 9) Bound the embedding model cache (LRU) + explicit flush

**Problem.** `_QWEN_LLM_CACHE` grows unbounded with distinct configs (prompt templates, precision, quantization, etc.); memory won’t be reclaimed until process exit.

**Change (what).** Swap to a small LRU (e.g., size 2–3), and expose `flush_llm_cache()` for long-running workers.

**How it works.** New entries evict oldest; evicted models are closed if they expose a `close()`/`shutdown()`. A management signal can clear all entries before switching cohorts.

**Why it reduces risk.**

* Prevents slow memory/VRAM growth on varied workloads.
* Operators can flush between batches to reclaim resources.

**Blast radius.** Low; localized to embedding loader.

**Definition of done.** Memory profile plateaus with alternating configs; explicit flush reclaims memory.

---

# P2 — API/typing & modularity

## 10) Replace “stringly-typed” context dicts with **typed dataclasses**

**Problem.** Many paths pass big dicts (e.g., `download_context`) with magic keys and mutations; typos are silent, and validation is fragmented.

**Change (what).** Introduce a `@dataclass` (or `TypedDict`) for the context shared between CLI and pipeline. Move defaulting/validation into the dataclass.

**How it works.** Callers construct `DownloadContext(...)`; unknown fields are obvious; IDEs/types catch mistakes. The pipeline reads attributes—not magic strings.

**Why it reduces risk.**

* Eliminates whole classes of bugs (misspelled keys, wrong types).
* Centralizes defaults (one place to read).

**Blast radius.** Medium: all call sites touching the context convert to the dataclass. Doable incrementally with an adapter.

**Definition of done.** No direct `dict['some_key']` lookups for context in the pipeline; typing hints pass.

---

## 11) Verify serializer provider **contract** on load

**Problem.** `_resolve_serializer_provider("module:Class")` only checks that the attribute exists; misconfigured classes fail later (at first method call) with confusing errors.

**Change (what).** After importing, assert `issubclass` against the expected base/protocol (e.g., `ChunkingSerializerProvider`). Fail fast with a descriptive error.

**How it works.** Misconfigurations show up at stage start, not during mid-pipeline operation.

**Why it reduces risk.**

* Fewer runtime “mystery” failures.
* Better UX for config authors.

**Blast radius.** Low.

**Definition of done.** Bad provider spec produces an immediate `TypeError` with actionable text.

---

## 12) Remove **import-time** filesystem side effects

**Problem.** Importing `chunking.py` computes defaults and may create directories. That’s surprising for library consumers inspecting types or running dry tests.

**Change (what).** Move directory creation and path resolution into `finalize()` or the stage runner. Importing the module should be a no-op beyond definitions.

**How it works.** Defaults are stored as callables or `default_factory`; only when a config is materialized (and a stage invoked) do we make directories.

**Why it reduces risk.**

* No hidden I/O on import; safer to embed into other systems.
* Cleaner test environments.

**Blast radius.** Low/medium: remove `os.makedirs(...)` and similar from module top level.

**Definition of done.** Importing DocParsing modules never creates folders on disk; stage start does.

---

# P3 — Logging/observability & deduplication

## 13) Prefer structured logging; no bare `print()`

**Problem.** `_validate_chunk_files()` prints a summary to stdout while the rest of the stage uses structured `log_event`. Mixed styles complicate capture and correlation.

**Change (what).** Emit summaries via the logging façade with consistent event names/keys, or return the summary object for the caller to log.

**How it works.** Every message includes run id, stage, file id, shard, and any durations/sizes; collectors can parse uniformly.

**Why it reduces risk.**

* Clean logs; easier to filter/analyze.
* No interleaving mess with other prints.

**Blast radius.** Very low.

**Definition of done.** No `print` in stage code (except explicit user-facing CLI banners).

---

## 14) Deduplicate helpers and schema-version normalization

**Problem.** We have multiple near-copies of `dedupe_preserve_order`, and both `ensure_chunk_schema` and `process_pass_a` normalize schema versions independently.

**Change (what).** Keep one canonical helper per concern:

* One `dedupe_preserve_order` in `io.py`/`utils.py`.
* One `ensure_chunk_schema(rec)` used everywhere.
* Remove accidental double decorators (e.g., `@classmethod` twice).

**How it works.** Every code path that needs schema defaulting calls the single helper; future schema upgrades touch one place.

**Why it reduces risk.**

* Prevents drift when taxonomy evolves.
* Simpler reviews (fewer places to check).

**Blast radius.** Low.

**Definition of done.** Grep shows a single definition for each helper; callers updated accordingly.

---

## 15) Clarify who owns **manifest logging and aggregation**

**Problem.** The CLI currently does manifest writes and run summaries while the pipeline already emits structured attempts. Responsibility is blurred, and reuse outside the CLI is harder.

**Change (what).** Move manifest composition and aggregation into telemetry (or a small orchestrator module). The CLI coordinates: parse args → build config → call runner → render summaries **from telemetry’s outputs**.

**How it works.** Pipeline sends attempt records; telemetry sinks produce per-work manifests and aggregation; the CLI is a thin façade.

**Why it reduces risk.**

* Single source of truth for manifest shape.
* The pipeline can be reused by other entry points (service/API) with consistent outputs.

**Blast radius.** Medium: the CLI gets slimmer; telemetry takes on more responsibility.

**Definition of done.** Turning off the CLI still yields manifests if the pipeline + telemetry are invoked from another program.

---

# Cross-cutting test plan (what proves the changes worked)

* **Config invariants**: invalid token/shard values fail in `finalize()` regardless of entry path (env/file/CLI).
* **JSONL streaming**: huge JSONL files process at a flat memory footprint; malformed lines don’t abort the run.
* **Hash guard**: invalid env algorithms produce warnings and fall back; valid ones are honored.
* **No import side effects**: importing modules does not create directories; running stages does.
* **Serializer contract**: bad provider raises a clear error on start.
* **Coalesce optimization**: mock tokenizer indicates fewer calls; wall-clock reduces on large docs.
* **Embedding cache**: alternating configs keep memory bounded; `flush_llm_cache()` drops allocations.
* **Structured logging**: log stream includes canonical events for validation/summary; no stray prints.

---

# Roll-out plan (low thrash)

1. **Day 0 (P0):** JSONL streaming; hash guard; `finalize()` reroute + invariants.
2. **Day 1 (P1):** Drop duplicate index load; coalesce optimization; optional `tqdm`; LRU cache + flush.
3. **Day 2 (P2):** Extract `io.py`, `config.py`, `env.py`, `logging.py`; introduce typed context; keep re-exports in `core.py`.
4. **Day 3 (P3):** Structured logging for validators; dedupe helpers; centralize schema normalization; plugin contract check.
5. **Day 4:** Tests/CI gates; remove temporary re-exports once call sites migrate.

---

# TL;DR (prioritized)

**P0 – Correctness & run-stoppers (do first)**

1. Split `core.py` responsibilities that currently couple unrelated behaviors (HTTP, CLI, logging, JSONL, config).
2. Make JSONL **streaming**; don’t materialize whole files.
3. Guard `resolve_hash_algorithm` against bad env overrides (e.g., `sha-1`).
4. Recompute `in_dir`/`doctags_dir` in `finalize()` when `data_root` changes; stop caching import-time defaults.
5. Add config **invariant checks in `finalize()`** (e.g., `min_tokens ≤ max_tokens`, shard bounds).

**P1 – Performance & memory**
6) Fix redundant work: drop duplicate `chunk_manifest_index` loads; avoid re-tokenizing unchanged records in `coalesce_small_runs`.
7) Make progress/throughput bits robust and lightweight: optional `tqdm` import, and bounded/evictable caches for LLMs.

**P2 – API/typing & modularity**
8) Replace ad-hoc context dicts with typed dataclasses; centralize “overlay” logic (env→file→CLI) in a base config.
9) Verify plugin contracts (serializer provider class really implements the expected interface).
10) Remove import-time filesystem side-effects; create directories lazily.

**P3 – Logging/observability & duplication**
11) Route prints through structured logging; keep summaries consistent.
12) Deduplicate helpers (e.g., `dedupe_preserve_order`) and schema-version normalization in one place.

---

You got it. Below are **drop-in Python snippets** in the exact order I just described (P0 → P3, items 1–15). They’re written to be easy to paste into your tree under `src/DocsToKG/DocParsing/…`. Names are conservative so they won’t collide; adjust imports to match your package layout.

---

## P0 — Correctness & run-stoppers

### 1) Decompose `core.py` into focused modules (shim + stubs)

```python
# src/DocsToKG/DocParsing/core.py  (temporary shim during decomposition)
# Goal: re-export from focused modules so callers don't break during the split.

from .io import iter_jsonl, iter_jsonl_batches, write_jsonl_atomically, dedupe_preserve_order, make_hasher
from .config import StageConfigBase, ChunkerCfg, EmbeddingCfg
from .paths import detect_data_root, path_doctags, path_chunks
from .logging import log_event, ManifestWriter, Summary

__all__ = [
    "iter_jsonl", "iter_jsonl_batches", "write_jsonl_atomically", "dedupe_preserve_order", "make_hasher",
    "StageConfigBase", "ChunkerCfg", "EmbeddingCfg",
    "detect_data_root", "path_doctags", "path_chunks",
    "log_event", "ManifestWriter", "Summary",
]
```

```python
# src/DocsToKG/DocParsing/paths.py
from pathlib import Path
import os

def detect_data_root(env=os.environ) -> Path:
    v = env.get("DOCSTOKG_DATA_ROOT")
    return Path(v).expanduser() if v else Path.cwd()

def path_doctags(root: Path) -> Path:
    return (root / "doctags").resolve()

def path_chunks(root: Path) -> Path:
    return (root / "chunks").resolve()
```

```python
# src/DocsToKG/DocParsing/logging.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, logging, time
logger = logging.getLogger("DocsToKG.DocParsing")

def log_event(event: str, **fields) -> None:
    payload = {"event": event, "ts": time.time(), **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))

@dataclass
class Summary:
    processed: int
    succeeded: int
    failed: int
    duration_s: float

class ManifestWriter:
    def __init__(self, path): self._path = path
    def write_row(self, row: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    def write_summary(self, summary: Summary) -> None:
        self.write_row({"type": "summary", **asdict(summary)})
```

---

### 2) Streaming JSONL (and batched) readers

```python
# src/DocsToKG/DocParsing/io.py
from __future__ import annotations
from typing import Iterator, Iterable
from pathlib import Path
import json, os, hashlib
from contextlib import contextmanager
import tempfile, shutil

def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # optionally: log_event("jsonl_decode_error", path=str(path))
                continue

def iter_jsonl_batches(paths: Iterable[Path], batch_size: int = 1000) -> Iterator[list[dict]]:
    buf: list[dict] = []
    for p in paths:
        for rec in iter_jsonl(p):
            buf.append(rec)
            if len(buf) >= batch_size:
                yield buf
                buf = []
    if buf:
        yield buf

@contextmanager
def _atomic_write(path: Path, mode: str = "w", encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(prefix="atom-", dir=str(path.parent))
    tmp = Path(d) / (path.name + ".part")
    try:
        with tmp.open(mode, encoding=encoding) as f:
            yield f
        shutil.move(str(tmp), str(path))
    finally:
        try: shutil.rmtree(d)
        except Exception: pass

def write_jsonl_atomically(path: Path, records: Iterable[dict]) -> None:
    with _atomic_write(path) as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set(); out = []
    for x in items:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out

def make_hasher(name: str | None = None, *, default: str = "sha256") -> "hashlib._Hash":
    alg = (name or os.getenv("DOCSTOKG_HASH_ALG") or default).lower()
    if alg not in hashlib.algorithms_available:
        # optional: log_event("unknown_hash_alg", alg=alg, fallback=default)
        alg = default
    return hashlib.new(alg)
```

---

### 3) Safe hash-algorithm override (already included above as `make_hasher`)

> See `make_hasher()` in the snippet just above. It guards env overrides and falls back to `sha256`.

---

### 4) Recompute dirs on `finalize()` (no import-time caching)

```python
# src/DocsToKG/DocParsing/config.py
from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Any
import os, yaml
from .paths import detect_data_root, path_doctags, path_chunks

@dataclass
class StageConfigBase:
    data_root: Path | None = None

    @classmethod
    def from_sources(
        cls,
        *,
        defaults: dict[str, Any] | None = None,
        cfg_path: Path | None = None,
        args: Any | None = None,
        env: Mapping[str, str] = os.environ,
    ):
        obj = cls(**(defaults or {}))
        # 1) file
        if cfg_path and cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}
            for k, v in data.items():
                if hasattr(obj, k): setattr(obj, k, v)
        # 2) env for data_root
        dr = env.get("DOCSTOKG_DATA_ROOT")
        if dr and getattr(obj, "data_root", None) in (None, ""):
            obj.data_root = Path(dr).expanduser()
        # 3) CLI args override
        if args is not None:
            for f in fields(obj):
                if hasattr(args, f.name):
                    val = getattr(args, f.name)
                    if val is not None:
                        setattr(obj, f.name, val)
        return obj.finalize()

@dataclass
class ChunkerCfg(StageConfigBase):
    in_dir: Path | None = None
    doctags_dir: Path | None = None
    out_dir: Path | None = None
    min_tokens: int = 70
    max_tokens: int = 260
    shard_index: int = 0
    shard_count: int = 1
    # … other knobs …

    def finalize(self) -> "ChunkerCfg":
        root = self.data_root or detect_data_root()
        if self.in_dir is None:
            self.in_dir = path_doctags(root)
        if self.doctags_dir is None:
            self.doctags_dir = path_doctags(root)
        if self.out_dir is None:
            self.out_dir = path_chunks(root)
        # invariants (see item 5)
        if self.min_tokens < 0 or self.max_tokens < 0:
            raise ValueError("min_tokens/max_tokens must be non-negative")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be ≤ max_tokens")
        if not (0 <= self.shard_index < self.shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        return self

@dataclass
class EmbeddingCfg(StageConfigBase):
    files_parallel: int = 2
    # …
    def finalize(self) -> "EmbeddingCfg":
        root = self.data_root or detect_data_root()
        # derive any paths from root if needed…
        if self.files_parallel < 1:
            raise ValueError("files_parallel must be ≥ 1")
        return self
```

---

### 5) Invariant checks in `finalize()` (already included above)

> See `ChunkerCfg.finalize()` and `EmbeddingCfg.finalize()` in the snippet just above (min/max tokens, shard bounds, parallelism).

---

## P1 — Performance & memory

### 6) Load `chunk_manifest_index` once (resume path)

```python
# src/DocsToKG/DocParsing/chunking.py  (inside your resume orchestration)
def run_chunking(cfg: ChunkerCfg, manifest_path: Path, resume: bool = True) -> None:
    manifest_index = None
    if resume and manifest_path.exists():
        manifest_index = load_chunk_manifest_index(manifest_path)  # ← single load
    # pass manifest_index down; do not call load_chunk_manifest_index again
    process_inputs(cfg, manifest_index=manifest_index)
```

---

### 7) Avoid re-tokenizing unchanged records in `coalesce_small_runs`

```python
# src/DocsToKG/DocParsing/chunking.py
from typing import Sequence

def coalesce_small_runs(chunks: list[dict], tokenizer) -> list[dict]:
    # Assume each chunk has 'text' and 'token_count'
    out = chunks[:]  # shallow copy
    changed: set[int] = set()

    i = 0
    while i < len(out) - 1:
        a, b = out[i], out[i+1]
        # decide if merge is needed (example condition)
        if a["token_count"] < 40:
            merged_text = a["text"] + " " + b["text"]
            # approximate new token_count by adding; we will re-tokenize later
            approx_tokens = a["token_count"] + b["token_count"]
            out[i] = {"text": merged_text, "token_count": approx_tokens}
            del out[i+1]
            changed.add(i)  # mark merged index
            continue
        i += 1

    # re-tokenize ONLY changed indices
    for idx in changed:
        out[idx]["token_count"] = len(tokenizer.encode(out[idx]["text"]))
    return out
```

---

### 8) Make `tqdm` truly optional

```python
# src/DocsToKG/DocParsing/embedding.py (top of file)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(it, **kwargs):  # no-op fallback
        return it
```

---

### 9) Bound the embedding model cache (LRU) + flush

```python
# src/DocsToKG/DocParsing/embedding.py
from collections import OrderedDict
from typing import Callable, Any

class _LRU:
    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self._d: OrderedDict[tuple, Any] = OrderedDict()

    def get(self, key: tuple, factory: Callable[[], Any]) -> Any:
        if key in self._d:
            self._d.move_to_end(key)  # mark as recently used
            return self._d[key]
        obj = factory()
        self._d[key] = obj
        self._d.move_to_end(key)
        if len(self._d) > self.maxsize:
            _, evicted = self._d.popitem(last=False)
            try:
                getattr(evicted, "close", lambda: None)()
            except Exception:
                pass
        return obj

    def clear(self) -> None:
        for _, obj in list(self._d.items()):
            try:
                getattr(obj, "close", lambda: None)()
            except Exception:
                pass
        self._d.clear()

_QWEN_LLM_CACHE = _LRU(maxsize=2)

def get_qwen_model(cfg_tuple: tuple) -> Any:
    def factory():
        # construct the model for this cfg_tuple
        return _make_qwen_model(cfg_tuple)
    return _QWEN_LLM_CACHE.get(cfg_tuple, factory)

def flush_llm_cache() -> None:
    _QWEN_LLM_CACHE.clear()
```

---

## P2 — API/typing & modularity

### 10) Replace context dicts with typed dataclasses

```python
# src/DocsToKG/DocParsing/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ParsingContext:
    run_id: str
    data_root: Path
    # IO/paths
    in_dir: Path
    out_dir: Path
    doctags_dir: Path
    # chunking knobs
    min_tokens: int = 70
    max_tokens: int = 260
    # execution
    files_parallel: int = 2
    # feature toggles / misc
    enable_validation: bool = True
    extra: dict = field(default_factory=dict)
```

> Use `ParsingContext` across CLI → chunking/embedding instead of `dict` lookups; add/adjust fields to match your current usage.

---

### 11) Verify serializer provider contract early

```python
# src/DocsToKG/DocParsing/interfaces.py
from typing import Protocol, Iterable

class ChunkingSerializerProvider(Protocol):
    def serialize(self, chunks: Iterable[dict]) -> bytes: ...
```

```python
# src/DocsToKG/DocParsing/chunking.py
from importlib import import_module
from .interfaces import ChunkingSerializerProvider

def resolve_serializer_provider(spec: str) -> type[ChunkingSerializerProvider]:
    mod_name, sep, cls_name = spec.partition(":")
    if sep != ":" or not mod_name or not cls_name:
        raise ValueError(f"Bad provider spec {spec!r}; expected 'module:Class'")
    cls = getattr(import_module(mod_name), cls_name)
    if not isinstance(cls, type) or not issubclass(cls, ChunkingSerializerProvider):  # type: ignore[arg-type]
        raise TypeError(f"{spec!r} is not a ChunkingSerializerProvider")
    return cls  # type: ignore[return-value]
```

---

### 12) Remove import-time filesystem side-effects

```python
# BAD (remove this pattern from module top-level):
# DEFAULT_ROOT = detect_data_root(); DEFAULT_OUT = path_chunks(DEFAULT_ROOT); DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

# GOOD: create dirs only when running the stage
def ensure_output_dirs(cfg: ChunkerCfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.doctags_dir.mkdir(parents=True, exist_ok=True)
```

Call `ensure_output_dirs(cfg)` in the stage runner just before writing outputs.

---

## P3 — Logging/observability & deduplication

### 13) Structured logging for validators; no prints

```python
# src/DocsToKG/DocParsing/chunking.py
from .logging import log_event

def _validate_chunk_files(paths: list[Path]) -> dict:
    stats = {"files": len(paths), "bytes": sum(p.stat().st_size for p in paths)}
    # return the dict; the caller logs it
    return stats

# CLI / runner:
stats = _validate_chunk_files(input_paths)
log_event("chunk_validation", **stats, stage="chunking", run_id=run_id)
```

---

### 14) Deduplicate helpers + schema-version normalization

```python
# src/DocsToKG/DocParsing/schema.py
def ensure_chunk_schema(rec: dict, *, default_version: str = "v1") -> dict:
    rec.setdefault("schema_version", default_version)
    # additional normalization/validation here…
    return rec
```

```python
# src/DocsToKG/DocParsing/io.py  (already houses dedupe_preserve_order)
# Ensure doctags.py imports and reuses: from .io import dedupe_preserve_order
```

```python
# src/DocsToKG/DocParsing/doctags.py  (fix accidental double decorator)
class DoctagsCfg(StageConfigBase):
    @classmethod
    def from_env(cls):  # ← ensure this appears only once
        # construct from env…
        ...
```

Use `ensure_chunk_schema(rec)` anywhere you previously normalized the schema in ad-hoc loops.

---

### 15) Clarify manifest ownership (telemetry sink owns it)

```python
# src/DocsToKG/DocParsing/telemetry.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, time

@dataclass
class Attempt:
    run_id: str
    file_id: str
    stage: str
    status: str
    reason: str | None
    started_at: float
    finished_at: float
    bytes: int

@dataclass
class ManifestEntry:
    run_id: str
    file_id: str
    stage: str
    output_path: str
    tokens: int
    schema_version: str
    duration_s: float

class TelemetrySink:
    def __init__(self, attempts_path: Path, manifest_path: Path):
        self._attempts_path = attempts_path
        self._manifest_path = manifest_path

    def write_attempt(self, a: Attempt) -> None:
        with open(self._attempts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(a)) + "\n")

    def write_manifest_entry(self, m: ManifestEntry) -> None:
        with open(self._manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(m)) + "\n")
```

```python
# CLI / runner (thin façade):
sink = TelemetrySink(attempts_path, manifest_path)
# pipeline code calls sink.write_attempt(...) and sink.write_manifest_entry(...)
# CLI only renders summaries from what sink produced.
```

---

### Notes

* Keep the **temporary re-exports** in `core.py` until all call-sites update; then delete the shim.
* For **percentiles** and other stats in long runs, prefer bounded deques (like you did for ContentDownload) or streaming quantiles if you need precise p-levels.

If you want these wrapped into specific file-level mini-PRs (e.g., “P0-hotfix set” vs “P2-refactor set”), I can lay them out as commit-ready diffs next.
