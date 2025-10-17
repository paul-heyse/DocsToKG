Absolutely—here’s a single, clean set of recommendations that merges all four DocParsing reviews into one plan. I’ve prioritized fixes, grouped them by subsystem, and included concrete “how to change it” notes and drop-in code patterns so your engineer can implement directly.

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

# Concrete changes (what/why/how)

## A) Refactor the “god module” (`core.py`) into focused pieces (P0/P2)

**Why**: 2k–4k lines mixing HTTP, CLI, logging, JSONL, config makes testing and reasoning difficult; import side-effects leak across stages.

**Target layout**

* `io.py` – JSONL iter/save (atomic writes), manifest helpers.
* `config.py` – `StageConfigBase` + per-stage cfgs; overlay/env helpers; `finalize` checks.
* `env.py` (or `paths.py`) – data_root detection, directory builders (no I/O at import).
* `logging.py` – `log_event`, manifest writers, summary formatting.
* (keep `chunking.py`, `embedding.py`, `doctags.py`, `token_profiles.py` focused)

**Tip**: Start by moving the *pure* helpers first (I/O & config). Keep the re-exports in `core.py` temporarily to avoid wide churn; delete once call-sites are updated.

---

## B) JSONL must be streaming, not eager (P0)

**Why**: Large corpora blow RAM when `list(load_jsonl(...))` is used.

**How**

```python
# io.py
def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    yield json.loads(s)
                except json.JSONDecodeError:
                    # optional: log and continue
                    continue

def iter_jsonl_batches(paths: Iterable[Path], batch_size: int = 1000) -> Iterator[list[dict]]:
    buf: list[dict] = []
    for p in paths:
        for rec in iter_jsonl(p):
            buf.append(rec)
            if len(buf) >= batch_size:
                yield buf; buf = []
    if buf: yield buf
```

Replace all eager loaders with the iterator(s).

---

## C) Hash algorithm override must be safe (P0)

**Why**: A typo in `DOCSTOKG_HASH_ALG` currently explodes late in `hashlib.new`.

**How**

```python
# io.py or utils.py
import hashlib, os, logging
log = logging.getLogger(__name__)

def make_hasher(name: str | None = None, *, default: str = "sha256") -> "hashlib._Hash":
    alg = (name or os.getenv("DOCSTOKG_HASH_ALG") or default).lower()
    if alg not in hashlib.algorithms_available:
        log.warning("Unknown hash alg %r; falling back to %s", alg, default)
        alg = default
    return hashlib.new(alg)
```

---

## D) Stop import-time defaults & recompute in `finalize()` (P0)

**Why**: Modules currently compute `DEFAULT_DATA_ROOT/IN_DIR/OUT_DIR` at import. Changing `DOCSTOKG_DATA_ROOT` later or passing a new `data_root` won’t rewire `in_dir/doctags_dir`.

**How**

```python
# config.py
from dataclasses import dataclass, field
from pathlib import Path
import os

def _env_data_root() -> Path | None:
    v = os.getenv("DOCSTOKG_DATA_ROOT")
    return Path(v).expanduser() if v else None

@dataclass
class ChunkerCfg:
    data_root: Path | None = None
    in_dir: Path | None = None
    doctags_dir: Path | None = None
    min_tokens: int = 70
    max_tokens: int = 260
    shard_index: int = 0
    shard_count: int = 1
    # ... other fields ...

    def finalize(self) -> "ChunkerCfg":
        root = self.data_root or _env_data_root() or Path.cwd()
        if self.in_dir is None:
            self.in_dir = root / "doctags"
        if self.doctags_dir is None:
            self.doctags_dir = root / "doctags"
        # invariants:
        if self.min_tokens < 0 or self.max_tokens < 0:
            raise ValueError("min_tokens/max_tokens must be non-negative")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be ≤ max_tokens")
        if not (0 <= self.shard_index < self.shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        return self
```

**Also**: Replace any global `DEFAULT_* = detect_*()` with lazy accessors or `default_factory` patterns and invoke directory creation only in the runtime path (not at import).

---

## E) Centralize “overlay” logic for configs (P2)

**Why**: Every config class repeats env→file→CLI merging; errors drift.

**How**

```python
# config.py
import dataclasses, yaml

@dataclass
class StageConfigBase:
    data_root: Path | None = None

    @classmethod
    def from_sources(
        cls, *,
        defaults: dict | None = None,
        cfg_path: Path | None = None,
        args: object | None = None,
        env: Mapping[str, str] = os.environ,
    ):
        obj = cls(**(defaults or {}))
        # 1) file
        if cfg_path and cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}
            for k, v in data.items():
                if hasattr(obj, k): setattr(obj, k, v)
        # 2) env
        dr = env.get("DOCSTOKG_DATA_ROOT")
        if dr and getattr(obj, "data_root", None) in (None, ""):
            obj.data_root = Path(dr).expanduser()
        # 3) args (non-None only)
        if args is not None:
            for f in dataclasses.fields(obj):
                if hasattr(args, f.name):
                    val = getattr(args, f.name)
                    if val is not None:
                        setattr(obj, f.name, val)
        return obj.finalize()
```

Have `ChunkerCfg`, `EmbeddingCfg`, etc. inherit `StageConfigBase`.

---

## F) Validate invariants in `finalize()` (P0)

**Why**: Non-CLI entry points silently accept bad values (token ranges, shard bounds, parallelism).

**How**: As shown in (D). Add similar checks to other configs (e.g., `files_parallel > 0` for embedding).

---

## G) Serializer provider contract verification (P2)

**Why**: `_resolve_serializer_provider` only checks importability; wrong classes fail later.

**How**

```python
# chunking.py
from importlib import import_module
from .interfaces import ChunkingSerializerProvider  # your protocol/base

def resolve_serializer_provider(spec: str) -> type[ChunkingSerializerProvider]:
    mod_name, _, cls_name = spec.partition(":")
    if not mod_name or not cls_name:
        raise ValueError(f"Bad provider spec {spec!r}; expected 'module:Class'")
    cls = getattr(import_module(mod_name), cls_name)
    if not issubclass(cls, ChunkingSerializerProvider):
        raise TypeError(f"{spec!r} is not a ChunkingSerializerProvider")
    return cls
```

---

## H) Eliminate redundant work & hotspots (P1)

1. **Duplicate index load**
   If `chunk_manifest_index(...)` is called twice in a row, remove the first call and reuse the result.

2. **Avoid re-tokenizing unchanged records**
   In `coalesce_small_runs`, only re-tokenize indices that actually merged; cache counts during the loop:

```python
changed: set[int] = set()
# during merge decisions:
#   when you merge i→j, add j to 'changed' and carry forward token counts
# after loop:
for idx in changed:
    rec = output[idx]
    rec.token_count = tok.count(rec.text)  # or set from cached value
```

---

## I) Embedding cache & optional dependencies (P1)

1. **Bound the LLM cache**

```python
# embedding.py
from collections import OrderedDict

class _LRU:
    def __init__(self, maxsize=2):
        self.maxsize = maxsize
        self._d: OrderedDict[tuple, object] = OrderedDict()
    def get(self, key, factory):
        if key in self._d:
            self._d.move_to_end(key); return self._d[key]
        obj = factory()
        self._d[key] = obj; self._d.move_to_end(key)
        if len(self._d) > self.maxsize:
            _, old = self._d.popitem(last=False)
            try: getattr(old, "close", lambda: None)()
            except Exception: pass
        return obj
    def clear(self): self._d.clear()

_QWEN_CACHE = _LRU(maxsize=2)

def get_qwen_llm(cfg_tuple):
    return _QWEN_CACHE.get(cfg_tuple, lambda: _make_qwen_llm(cfg_tuple))
def flush_llm_cache():
    _QWEN_CACHE.clear()
```

2. **Make `tqdm` truly optional**

```python
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs): return it  # no-op fallback
```

---

## J) Avoid import-time filesystem side-effects (P2)

**Why**: Library consumers importing types shouldn’t have directories auto-created.

**How**: Replace:

```python
DEFAULT_DATA_ROOT = detect_data_root(); DEFAULT_OUT_DIR = data_chunks(DEFAULT_DATA_ROOT); os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
```

with lazy comp in `finalize()` or:

```python
DEFAULT_DATA_ROOT = None  # sentinel
def _ensure_dirs(cfg: ChunkerCfg):
    cfg.in_dir.mkdir(parents=True, exist_ok=True)
    # … only when actually running a stage …
```

---

## K) Structured logging, no prints; unify schema-version normalization; dedupe helpers (P3)

1. **Use `log_event` (or return summaries) in `_validate_chunk_files`**
   Replace `print(...)` with `log_event("chunk_validation", **summary_dict)` or return `summary_dict` to the caller and have the CLI log it.

2. **Schema version normalization in one function**

```python
def ensure_chunk_schema(rec: dict, *, default_version="v1") -> dict:
    rec.setdefault("schema_version", default_version)
    # … any validation …
    return rec
```

Call this both where you formerly had `ensure_chunk_schema` **and** in `process_pass_a`, instead of duplicating loops.

3. **Deduplicate `dedupe_preserve_order`**
   Keep it once (e.g., in `io.py` or `utils.py`) and import it where needed.

4. **Remove obvious duplicates**

   * The double `import uuid` in `chunking.py`.
   * Any repeated load of indices or manifests.

---

# Test additions (quick but high-value)

* **Config invariants:** invalid token ranges and shard indices throw at `finalize()` regardless of entry point (env/file/CLI).
* **JSONL streaming:** huge file reads stay O(1) memory; corrupt lines are skipped but don’t abort.
* **Hash guard:** bad `DOCSTOKG_HASH_ALG` falls back to default and logs a warning.
* **No import side-effects:** importing `chunking.py` does not create directories; running the stage does.
* **Provider contract:** invalid serializer spec raises `TypeError` early.
* **Coalesce optimization:** unchanged records aren’t re-tokenized (mock tokenizer to assert call counts).
* **Embedding cache:** repeated distinct configs don’t grow memory unbounded; `flush_llm_cache()` clears state.
* **Optional `tqdm`:** module works with and without `tqdm` installed.

---

# Suggested rollout

**Day 0 (P0):** JSONL streaming, hash guard, `finalize()` recompute + invariants (Chunker/Embedding), remove duplicate loads, drop duplicate import.
**Day 1 (P1):** Coalesce re-tokenization optimization; optional `tqdm`; LLM cache LRU + flush.
**Day 2 (P2):** Split `core.py` → `io.py`, `config.py`, `env.py`, `logging.py`; add typed dataclasses and overlay.
**Day 3 (P3):** Structured logging for validators; dedupe helpers; centralize schema-version logic; plugin contract checks.
**Day 4:** Tests for the above + CI gates.

---

If you want, I can turn any subset of the above into concrete, file-ready snippets aligned to your current function/class names (e.g., exact `ChunkerCfg` fields, your `log_event` signature, current serializer base type).
