Absolutely — here’s a single, **prioritized**, “do-this-next” consolidation of the four OntologyDownload reviews, with a narrative overview *and* drop-in code snippets for each change. I’ve kept names generic but close to what you already use (e.g., `StreamingDownloader`, `ResolvedConfig`, `extract_zip_safe`, etc.). If any symbol names differ in your tree, treat these as small edits.

---

# TL;DR (prioritized)

**P0 — correctness & safety (fix first)**

1. **Range resume hardening:** avoid corrupting `.part` when a server ignores `Range`.
2. **Env overrides honored everywhere:** `ResolvedConfig.from_defaults()` must apply `_apply_env_overrides`.
3. **Extraction limits are runtime-tunable:** default to `None` and fill from active config at call time.
4. **Bound DNS cache:** TTL *and* LRU cap to prevent unbounded growth.

**P1 — robustness & DX**
5) **Managed logger isolation:** set `propagate=False` (or param) to stop double-logging.
6) **Plugins load exactly once, thread-safe:** consolidate sentinels and add locking/`lru_cache`.
7) **Stop mirroring plugin state in validators:** import the single loader accessor (no duplicate flags).
8) **Retry helper doesn’t swallow interrupts:** catch `Exception`, not `BaseException`.
9) **Python version check is programmatic:** raise typed error, don’t `SystemExit`.
10) **Rate-limit parsing is strict or loudly warns.**

**P2 — performance & scalability**
11) **Checksum fetch is streaming + capped.**
12) **DNS cache cap (dup of #4):** make it explicit and enforced.
13) **Avoid double-copy after download:** write to final destination (or hardlink) instead of `copy2`.
14) **Replace recursive retry on SHA mismatch with bounded loop.**
15) **Reuse a cached `psutil.Process` handle in `log_memory_usage()`.**

Each change below has: **Problem → What we’ll change → Why it helps → Code**.

---

## P0 — correctness & safety

### 1) Range resume hardening (no silent corruption)

**Problem**
When resuming, we open `.part` in append mode if `resume_position > 0`. If the origin ignores `Range` and returns `200 OK` instead of `206 Partial Content`, we append the *entire* object to the partial, corrupting it (double length).

**Change (what)**

* If we requested a `Range` and get **non-206** (e.g., `200`), **truncate and restart from byte 0** (open `wb`, not `ab`).
* Optionally, validate alignment by checking the first bytes of the response against the tail of the partial when `206`.

**Why this helps**
Guarantees correctness under servers that don’t honor `Range` (or proxies that strip it) — no more silent corruption.

**Code (drop-in pattern inside your streaming downloader)**

```python
# streaming_downloader.py (or wherever download_stream/StreamingDownloader lives)
def download_stream(session, url, dest_part: Path, *, resume_position: int = 0, timeout: float = 30.0):
    headers = {}
    want_range = resume_position > 0
    if want_range:
        headers["Range"] = f"bytes={resume_position}-"

    resp = session.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    status = resp.status_code
    accepts_ranges = resp.headers.get("Accept-Ranges", "").lower()
    range_honored = (status == 206)

    # If we asked for Range but server gave 200 (full body), start over.
    open_mode = "ab" if (want_range and range_honored) else "wb"

    if want_range and not range_honored:
        # We expected to append but cannot; clear partial (truncate)
        try:
            dest_part.unlink(missing_ok=True)
        except Exception:
            pass

    with dest_part.open(open_mode) as f:
        # Optional: alignment check when 206
        if want_range and range_honored:
            # If available, compare Content-Range start with resume_position
            cr = resp.headers.get("Content-Range", "")
            # Content-Range: bytes <start>-<end>/<total>
            if cr.startswith("bytes "):
                start_str = cr.split()[1].split("-")[0]
                try:
                    start_val = int(start_str)
                    if start_val != resume_position:
                        # Server started at a different offset; safest is restart
                        f.close()
                        dest_part.unlink(missing_ok=True)
                        return download_stream(session, url, dest_part, resume_position=0, timeout=timeout)
                except Exception:
                    # If unparsable, continue (we still have 206)
                    pass

        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if not chunk:
                continue
            f.write(chunk)

    return dest_part
```

---

### 2) Ensure env overrides apply in `from_defaults()`

**Problem**
`ResolvedConfig.from_defaults()` is widely used, but it **doesn’t** call `_apply_env_overrides`, so `ONTOFETCH_*` settings are silently ignored unless callers use another entry point.

**Change (what)**
Call `_apply_env_overrides` from `from_defaults()` (or provide a unified constructor like `ResolvedConfig.from_env_and_defaults()` and migrate callers).

**Why this helps**
Operators get consistent behavior whether they configure via CLI, env, or defaults.

**Code**

```python
# config.py
@dataclass
class ResolvedConfig:
    # ... fields ...
    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        cfg = cls(
            # the current defaults...
        )
        cfg._apply_env_overrides(os.environ)  # NEW
        return cfg

    def _apply_env_overrides(self, env: Mapping[str, str]) -> None:
        # existing logic; ensure it covers timeouts, retries, max sizes, etc.
        # e.g.:
        v = env.get("ONTOFETCH_TIMEOUT_S")
        if v:
            self.http_timeout_s = float(v)
        # etc...
```

---

### 3) Extraction limits are runtime-tunable (no import-time binding)

**Problem**
`_DEFAULT_MAX_UNCOMPRESSED_BYTES` is computed at import, and functions like `extract_zip_safe(..., max_uncompressed_bytes=_DEFAULT_MAX_UNCOMPRESSED_BYTES)` **bind the default at definition time**. Calls that rely on the default never see runtime overrides.

**Change (what)**
Make the parameter default `None`, and inside the function, pull from the active config (or fallback default) at **call time**.

**Why this helps**
Ops can tighten/loosen limits per run or via env without code changes.

**Code**

```python
# archive.py
def extract_zip_safe(zip_path: Path, dest_dir: Path, *, max_uncompressed_bytes: int | None = None) -> None:
    if max_uncompressed_bytes is None:
        max_uncompressed_bytes = get_active_config().max_uncompressed_bytes  # or a resolver/context accessor
    # proceed with bounded expansion check...

def extract_tar_safe(tar_path: Path, dest_dir: Path, *, max_uncompressed_bytes: int | None = None) -> None:
    if max_uncompressed_bytes is None:
        max_uncompressed_bytes = get_active_config().max_uncompressed_bytes
    # proceed...
```

> If there isn’t an “active config” accessor, pass the limit explicitly down the call-chain (preferred).

---

### 4) Bound the DNS cache (TTL + LRU cap)

**Problem**
`_cached_getaddrinfo` implements TTL, but never evicts; many distinct hosts can grow `_DNS_CACHE` until process exit.

**Change (what)**
Store `({expires_at, value})` and prune expired entries on access. Enforce a **size cap** via simple LRU.

**Why this helps**
Keeps memory bounded while preserving the latency win from caching.

**Code**

```python
# dns_cache.py
import socket, time, threading
from collections import OrderedDict

_DNS_LOCK = threading.Lock()
_DNS_TTL_S = 300.0
_DNS_MAX_ENTRIES = 4096
_DNS_CACHE: "OrderedDict[str, tuple[float, object]]" = OrderedDict()

def _prune(now: float) -> None:
    # drop expired
    keys = [k for k, (exp, _) in _DNS_CACHE.items() if exp <= now]
    for k in keys:
        _DNS_CACHE.pop(k, None)
    # enforce LRU cap
    while len(_DNS_CACHE) > _DNS_MAX_ENTRIES:
        _DNS_CACHE.popitem(last=False)

def cached_getaddrinfo(host: str, port: int, family=0, type=0, proto=0, flags=0):
    key = f"{host}:{port}:{family}:{type}:{proto}:{flags}"
    now = time.time()
    with _DNS_LOCK:
        ent = _DNS_CACHE.get(key)
        if ent and ent[0] > now:
            # refresh LRU position
            _DNS_CACHE.move_to_end(key)
            return ent[1]
    # miss or expired
    res = socket.getaddrinfo(host, port, family, type, proto, flags)
    with _DNS_LOCK:
        _DNS_CACHE[key] = (now + _DNS_TTL_S, res)
        _DNS_CACHE.move_to_end(key)
        _prune(now)
    return res
```

---

## P1 — robustness & DX

### 5) Keep structured logs self-contained (no duplicate root logs)

**Problem**
`setup_logging` installs JSON handlers but leaves `logger.propagate = True`, so the same records bubble to the root logger and get printed twice (JSON + plaintext).

**Change (what)**
Default `propagate=False`, with a flag to opt-in if callers *want* propagation.

**Why this helps**
Eliminates double-logging and mixed formats.

**Code**

```python
# logging_setup.py
def setup_logging(level="INFO", *, propagate: bool = False) -> logging.Logger:
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    logger.setLevel(level)
    logger.handlers.clear()
    # add your JSON/structured handler(s)...
    logger.addHandler(json_handler)
    logger.propagate = propagate  # NEW: default False
    return logger
```

---

### 6) Plugins load exactly once, thread-safe (no races)

**Problem**
`plugins.load_resolver_plugins()` / `load_validator_plugins()` gate with module-level booleans but no lock; concurrent calls can race and load twice.

**Change (what)**
Use a single accessor protected by a lock or `functools.lru_cache(maxsize=1)`.

**Why this helps**
Idempotent, thread-safe initialization; no partial registries.

**Code**

```python
# plugins.py
import threading
from functools import lru_cache

_PLUGINS_LOCK = threading.Lock()
_RESOLVER_REGISTRY = {}
_VALIDATOR_REGISTRY = {}

def _discover() -> tuple[dict, dict]:
    # discover resolver/validator entry points once
    # return (resolvers, validators)
    ...

@lru_cache(maxsize=1)
def ensure_plugins_loaded() -> tuple[dict, dict]:
    # lru_cache already uses an internal lock; extra lock optional
    with _PLUGINS_LOCK:
        if not _RESOLVER_REGISTRY and not _VALIDATOR_REGISTRY:
            resolvers, validators = _discover()
            _RESOLVER_REGISTRY.update(resolvers)
            _VALIDATOR_REGISTRY.update(validators)
    return _RESOLVER_REGISTRY, _VALIDATOR_REGISTRY

def get_resolver_registry() -> dict: return ensure_plugins_loaded()[0]
def get_validator_registry() -> dict: return ensure_plugins_loaded()[1]
```

---

### 7) Stop mirroring plugin state in validators

**Problem**
`validation.py` redefines its own `_RESOLVER_PLUGINS_LOADED`, etc., then tries to mutate `plugins`’ copies — easy to desynchronize.

**Change (what)**
In `validation.py`, import the **accessors** (`get_validator_registry`) and use them; delete any local sentinels.

**Why this helps**
Single source of truth for plugin state.

**Code**

```python
# validation.py
from .plugins import get_validator_registry

def run_validators(...):
    registry = get_validator_registry()
    # use registry directly; no local flags
```

---

### 8) Retry helper: don’t swallow interrupts

**Problem**
`retry_with_backoff` catches `BaseException`, which includes `KeyboardInterrupt` and `SystemExit`.

**Change (what)**
Catch `Exception` only; if you must do a broad catch, explicitly re-raise interrupts.

**Why this helps**
Respect process signals and predictable shutdown.

**Code**

```python
# retry.py
def retry_with_backoff(fn, *, attempts=3, backoff=lambda i: 2**i):
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            last_exc = e
            if i == attempts:
                raise
            time.sleep(backoff(i))
    raise last_exc
```

---

### 9) Python version check: raise a typed exception

**Problem**
`ensure_python_version()` prints + `SystemExit`. That’s brittle for library users.

**Change (what)**
Raise an `OntologyDownloadError` (or subtype) so callers can catch it; CLI can still convert to exit.

**Why this helps**
Better DX and testability.

**Code**

```python
# errors.py
class OntologyDownloadError(Exception):
    """Base error for OntologyDownload subsystem."""

class UnsupportedPythonError(OntologyDownloadError):
    pass

# compat.py
import sys
from .errors import UnsupportedPythonError

def ensure_python_version(min_major=3, min_minor=10) -> None:
    if (sys.version_info.major, sys.version_info.minor) < (min_major, min_minor):
        raise UnsupportedPythonError(
            f"Python >= {min_major}.{min_minor} required; found {sys.version.split()[0]}"
        )

# CLI
try:
    ensure_python_version()
except UnsupportedPythonError as e:
    print(str(e), file=sys.stderr)
    sys.exit(2)
```

---

### 10) Rate-limit parse: strict (or warn loudly)

**Problem**
`parse_service_rate_limit("garbage")` returns `None`, silently disabling service overrides.

**Change (what)**
Either: (a) **raise `ValueError`**, or (b) **log a warning** and fall back to global limit.

**Why this helps**
Misconfigurations surface quickly; operators learn immediately.

**Code**

```python
# rate_limit.py
import re, logging
log = logging.getLogger("DocsToKG.OntologyDownload")

_LIMIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*s(?:ec)?\s*$", re.I)

def parse_service_rate_limit(s: str) -> float:
    m = _LIMIT_RE.match(s or "")
    if not m:
        raise ValueError(f"Bad rate limit spec {s!r}; expected like '5/s'")
    return float(m.group(1))

# or, lenient:
def parse_service_rate_limit_or_warn(s: str, default: float) -> float:
    try:
        return parse_service_rate_limit(s)
    except ValueError:
        log.warning("Invalid per-service rate limit %r; using default %s/s", s, default)
        return default
```

---

## P2 — performance & scalability

### 11) Stream checksum fetch with a hard cap

**Problem**
`_fetch_checksum_from_url` downloads entire response into memory (`response.text`) and *then* scans. A bad endpoint can return huge payloads.

**Change (what)**
Stream with `iter_content`, enforce a **byte ceiling**, and scan incrementally with a rolling window (to catch digests across chunk boundaries).

**Why this helps**
Prevents memory blowups; keeps latency bounded.

**Code**

```python
# checksums.py
import re, io

DIGEST_RE = re.compile(r"(?i)\b([a-f0-9]{64})\b")  # example for sha256
MAX_BYTES = 2 * 1024 * 1024  # 2 MiB cap (tune as needed)

def fetch_checksum_streaming(session, url: str, *, max_bytes: int = MAX_BYTES, timeout: float = 10.0) -> str | None:
    resp = session.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    total = 0
    tail = b""
    for chunk in resp.iter_content(8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise OntologyDownloadError(f"checksum response too large (> {max_bytes} bytes)")
        buf = tail + chunk
        # scan current buffer
        m = DIGEST_RE.search(buf.decode("utf-8", errors="ignore"))
        if m:
            return m.group(1)
        # keep last 63 chars (digest length - 1) to bridge boundaries
        tail = buf[-128:]  # safe window
    return None
```

---

### 12) (Reinforce) DNS cache cap

Already covered in **P0 #4**. If your code has multiple cache sites, centralize to a single module.

---

### 13) Avoid double-copy after download

**Problem**
After `pooch.retrieve`, we `copy2` to final dest, doubling IO and peak disk usage.

**Change (what)**
Either: (a) configure `pooch.retrieve` to write directly to the **final destination** (`path`+`fname`), or (b) create a **hardlink** from cache into the final location (same filesystem), or (c) `os.replace` a temp into place.

**Why this helps**
Cuts IO/time in half for large ontologies; reduces storage pressure.

**Code**

```python
# io_download.py
import os, shutil

def retrieve_to_dest(url: str, dest: Path, *, downloader) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Option A: instruct pooch to place file directly
    local = pooch.retrieve(url=url, path=str(dest.parent), fname=dest.name, downloader=downloader)
    if Path(local) != dest:
        # If pooch insists on cache path, try hardlink; fallback to move
        try:
            os.link(local, dest)
        except OSError:
            shutil.move(local, dest)
    return dest
```

---

### 14) Replace recursive retry on checksum mismatch with a bounded loop

**Problem**
On SHA mismatch, `download_stream` recursively calls itself. If the file changes repeatedly, you risk deep recursion and wasteful re-downloads.

**Change (what)**
Use an iterative loop with a counter and clear logging.

**Why this helps**
Predictable retry behavior; easier to reason about.

**Code**

```python
# io_download.py
def download_and_verify(session, url, dest, expected_sha256: str, *, max_attempts: int = 3):
    for attempt in range(1, max_attempts + 1):
        tmp = dest.with_suffix(".part")
        download_stream(session, url, tmp, resume_position=0)
        actual = sha256_file(tmp)
        if actual == expected_sha256:
            os.replace(tmp, dest)
            return dest
        log.warning("sha_mismatch", attempt=attempt, expected=expected_sha256, actual=actual, url=url)
        tmp.unlink(missing_ok=True)
    raise OntologyDownloadError(f"checksum mismatch after {max_attempts} attempts: {url}")
```

---

### 15) Reuse a cached `psutil.Process` in `log_memory_usage()`

**Problem**
Creating `psutil.Process()` each call is overhead when logging at DEBUG for many batches.

**Change (what)**
Create **one** process handle at import and reuse; optionally guard with a try/except if psutil is optional.

**Why this helps**
Small but consistent perf win with lots of logging.

**Code**

```python
# telemetry.py
try:
    import psutil
    _PROC = psutil.Process()  # cached
except Exception:
    psutil = None
    _PROC = None

def log_memory_usage(log):
    if not psutil or not _PROC:
        return
    mem = _PROC.memory_info().rss
    log.info("mem_usage", rss_bytes=mem)
```

---

# Rollout plan (low-thrash order)

* **Day 0 (P0):** Range resume hardening; env overrides in `from_defaults`; extraction limits on call; DNS cache cap.
* **Day 1 (P1):** Logger isolation; plugin once-loader; remove mirrored flags; retry helper; python version error; rate-limit parsing.
* **Day 2 (P2):** Streaming checksum; no double-copy; iterative checksum retries; cached `psutil.Process`.
* **Day 3:** Tests & CI gates (see below).

---

# Minimal test checklist (hits the risky parts)

* **Resume hardening:** simulate server answering `200` to a Range request → verify we restart and the final file hash matches. Simulate proper `206` with correct `Content-Range`.
* **Env overrides path:** set `ONTOFETCH_*` and construct via `ResolvedConfig.from_defaults()` → overrides present.
* **Extraction limit default:** call `extract_*_safe()` with no param after changing active config → limit honored; with explicit param → param wins.
* **DNS cache cap:** resolve > 5k unique hosts → cache doesn’t exceed cap; expired entries removed on access.
* **Logger propagation:** with a root handler installed, you see only one JSON record per event.
* **Plugins concurrency:** call `ensure_plugins_loaded()` from multiple threads → no duplicates.
* **Retry helper:** sends `KeyboardInterrupt`/`SystemExit` through; retries other exceptions.
* **Version check:** raises `UnsupportedPythonError`; CLI catches and exits with code 2.
* **Rate-limit parse:** invalid strings raise or warn; valid `5/s` yields `5.0`.
* **Checksum streaming:** large response refuses > cap; small checksum file returns digest.
* **No double-copy:** final file appears directly in dest directory; no second copy.
* **Iterative retries:** repeated mismatches stop at attempt N and error.

---

If you’d like, I can turn any subset of the above into a small, surgical PR against your current branch (e.g., a “P0-hotfix” patch with items 1–4).
