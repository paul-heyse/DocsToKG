Absolutely—here are **surgical, copy-pasteable changes** for the first two items you flagged. I’ve written them as a series of “where → what to change → exact code” examples so it’s unambiguous.

> Scope
>
> 1. **Make `download_execution` use the atomic writer + size enforcement** (P1 parity).
> 2. **Finish httpx + hishel wiring** so **all GETs** go through a single shared **httpx.Client with hishel CacheTransport**, and emit the cache-aware attempt tokens (`cache-hit`, `http-304`).

---

# 1) Wire the **atomic writer** into `download_execution` (and enforce `Content-Length`)

### 1.1 Add imports & function args

**File:** `src/DocsToKG/ContentDownload/download_execution.py`

**Before**

```python
import os, time
# ... no atomic writer import
def stream_candidate_payload(plan: DownloadPlan, *,
    session=None,
    timeout_s: Optional[float] = None,
    chunk_size: int = 1 << 20,
    expected_len: Optional[int] = None,
    telemetry: Optional[AttemptSink] = None,
    run_id: Optional[str] = None,
) -> DownloadStreamResult:
    ...
```

**After**

```python
import os, time
from DocsToKG.ContentDownload.io_utils import atomic_write_stream, SizeMismatchError
# or: from DocsToKG.ContentDownload.streaming import stream_to_part as atomic_writer (rename arg names accordingly)

def stream_candidate_payload(plan: DownloadPlan, *,
    session,                                 # httpx client wrapper (required, not optional)
    timeout_s: Optional[float] = None,
    chunk_size: int = 1 << 20,
    verify_content_length: bool = True,      # NEW: enforce length when header present
    telemetry: Optional[AttemptSink] = None,
    run_id: Optional[str] = None,
) -> DownloadStreamResult:
    ...
```

> Rationale: we no longer accept `expected_len` as input; we calculate it from the response headers and feed it into the atomic writer internally.

---

### 1.2 Replace the manual write loop with `atomic_write_stream(...)`

**Before (condensed)**

```python
resp = session.get(url, stream=True, allow_redirects=True, timeout=timeout_s)
tmp_dir = os.getcwd()
tmp_path = os.path.join(tmp_dir, ".part-download.tmp")
bytes_written = 0
with open(tmp_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        f.write(chunk)
        bytes_written += len(chunk)

_emit(..., status="http-200", http_status=resp.status_code,
      content_type=resp.headers.get("Content-Type"),
      bytes_written=bytes_written,
      content_length_hdr=int(resp.headers.get("Content-Length", "0")) or None)
return DownloadStreamResult(path_tmp=tmp_path, bytes_written=bytes_written,
                            http_status=resp.status_code, content_type=...)
```

**After (atomic + size enforcement)**

```python
resp = session.get(url, stream=True, timeout=timeout_s)  # hishel-enabled httpx client
content_len_hdr = resp.headers.get("Content-Length")
expected_len = int(content_len_hdr) if (content_len_hdr and content_len_hdr.isdigit()) else None

tmp_dir = os.getcwd()                                     # or a configured staging dir
os.makedirs(tmp_dir, exist_ok=True)
tmp_path = os.path.join(tmp_dir, ".part-download.tmp")

try:
    bytes_written = atomic_write_stream(
        dest_path=tmp_path,
        byte_iter=resp.iter_bytes(),                      # httpx iterator
        expected_len=(expected_len if verify_content_length else None),
        chunk_size=chunk_size
    )

    _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
          verb="GET", status="http-200", http_status=resp.status_code,
          content_type=resp.headers.get("Content-Type"), bytes_written=bytes_written,
          content_length_hdr=expected_len)
    return DownloadStreamResult(
        path_tmp=tmp_path,
        bytes_written=bytes_written,
        http_status=resp.status_code,
        content_type=resp.headers.get("Content-Type"),
    )

except SizeMismatchError:
    # Temp file has already been removed by atomic_write_stream on mismatch
    _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
          verb="GET", status="size-mismatch", http_status=resp.status_code,
          content_type=resp.headers.get("Content-Type"), reason="size-mismatch",
          content_length_hdr=expected_len)
    # Bubble an error outcome up to finalize or convert here:
    raise DownloadError("size-mismatch")
```

> Notes
> • `atomic_write_stream` must `fsync()` the file and use `os.replace()` to ensure atomic commit when you later rename/move into the final location.
> • If you prefer: call your existing `streaming.stream_to_part()` with equivalent semantics.

---

### 1.3 Handle **304** short-circuit (no write) and **HEAD** fallback edge cases

Immediately after GET:

```python
from_cache = bool(resp.extensions.get("from_cache"))
revalidated = bool(resp.extensions.get("revalidated"))

# Emit handshake
_emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
      verb="GET", status="http-get", http_status=resp.status_code,
      content_type=resp.headers.get("Content-Type"), elapsed_ms=elapsed_ms)

if from_cache and not revalidated:
    _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
          verb="GET", status="cache-hit", http_status=resp.status_code,
          content_type=resp.headers.get("Content-Type"), reason="ok")
    # You *can* return a DownloadStreamResult pointing at existing file if you cache bodies on disk;
    # In our design, cache-hit means no disk write here; finalization path should interpret accordingly.

if revalidated and resp.status_code == 304:
    _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
          verb="GET", status="http-304", http_status=304,
          content_type=resp.headers.get("Content-Type"), reason="not-modified")
    return DownloadStreamResult(path_tmp="", bytes_written=0, http_status=304,
                                content_type=resp.headers.get("Content-Type"))
```

> If you sometimes get `405` on HEAD, wrap the HEAD call in `try/except` and proceed without HEAD (policy-gate can be GET-only).

---

### 1.4 Pass the new flags from the pipeline

**File:** `src/DocsToKG/ContentDownload/pipeline.py`

**Before**

```python
stream = stream_candidate_payload(
    adj_plan, session=client, timeout_s=None,
    chunk_size=self._dlp.chunk_size_bytes,
    telemetry=self._telemetry, run_id=self._run_id
)
```

**After**

```python
stream = stream_candidate_payload(
    adj_plan,
    session=client,
    timeout_s=None,
    chunk_size=self._dlp.chunk_size_bytes,
    verify_content_length=self._dlp.verify_content_length,   # NEW
    telemetry=self._telemetry,
    run_id=self._run_id,
)
```

---

### 1.5 Finalization unchanged, but make sure **rename is atomic** and **verify** is optional

In `finalize_candidate_download(...)`, keep:

* **Atomic move** (`os.replace`) into final path.
* Optional **post-move verification** (`if verify_content_length and expected_len`) is already covered by the atomic writer; keep PDF tail check if you have it.

---

### 1.6 Minimal tests to add/adjust

* **Size mismatch**: fake GET declares `Content-Length: 500000` but yields 100k → attempt `size-mismatch`, no final file, `DownloadError("size-mismatch")`.
* **Happy path**: `Content-Length` correct → attempt `http-200` with `bytes_written==expected_len`, final file exists (inode updated after rename).
* **304 path**: return `DownloadStreamResult(path_tmp="", http_status=304)`; finalization returns skip outcome; no file created.

---

# 2) Finish **httpx + hishel** wiring and deprecate direct `requests` usage

Goal: **exactly one** shared **httpx.Client** with **hishel CacheTransport** for all GETs, wrapped by a per-resolver **RateRetryClient** (rate limit + retry + telemetry). No module should call `requests.get()` / `requests.Session()`.

### 2.1 Build hishel transport + shared httpx client (one place)

**File:** `src/DocsToKG/ContentDownload/httpx/hishel_build.py`

```python
import httpx, hishel

def build_hishel_transport(cfg_hishel) -> hishel.CacheTransport:
    storage = hishel.FileStorage(base_path=cfg_hishel.base_path,
                                 ttl=cfg_hishel.ttl_seconds,
                                 check_ttl_every=cfg_hishel.check_ttl_every_seconds)
    controller = hishel.Controller(
        force_cache=cfg_hishel.force_cache,
        allow_heuristics=cfg_hishel.allow_heuristics,
        allow_stale=cfg_hishel.allow_stale,
        always_revalidate=cfg_hishel.always_revalidate,
        cache_private=cfg_hishel.cache_private,
        cacheable_methods=cfg_hishel.cacheable_methods,
    )
    return hishel.CacheTransport(transport=httpx.HTTPTransport(),
                                 controller=controller, storage=storage)

def build_httpx_client_with_hishel(cfg_http, transport) -> httpx.Client:
    headers = {"User-Agent": cfg_http.user_agent}
    if cfg_http.mailto and "mailto:" not in headers["User-Agent"]:
        headers["User-Agent"] += f" (+mailto:{cfg_http.mailto})"
    timeouts = httpx.Timeout(connect=cfg_http.timeout_connect_s,
                             read=cfg_http.timeout_read_s,
                             write=cfg_http.timeout_read_s,
                             pool=cfg_http.timeout_read_s)
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    return httpx.Client(transport=transport, headers=headers, timeout=timeouts,
                        limits=limits, verify=bool(cfg_http.verify_tls),
                        proxies=cfg_http.proxies or None)
```

---

### 2.2 Wrap the shared client once per resolver for **rate + retry + telemetry**

**File:** `src/DocsToKG/ContentDownload/httpx/client.py`

* Ensure **TokenBucket** is thread-safe (mutex).
* Ensure **retry** honors `Retry-After`.
* **Refund** a token on pure **cache-hit** (`from_cache && !revalidated`).

```python
class RateRetryClient:
    def __init__(self, resolver_name, client: httpx.Client, *, retry_policy, rate_policy, telemetry, run_id):
        self.name = resolver_name; self.c = client; self.t = telemetry; self.run_id = run_id
        self.bucket = TokenBucket(rate_policy.capacity, rate_policy.refill_per_sec, rate_policy.burst)
        # retry params cached...

    def head(self, url: str, **kw) -> httpx.Response:
        return self._request("HEAD", url, **kw)

    def get(self, url: str, **kw) -> httpx.Response:
        return self._request("GET", url, **kw)

    def _request(self, method: str, url: str, **kw) -> httpx.Response:
        sleep_s = self.bucket.consume(1.0)
        if sleep_s > 0:
            time.sleep(sleep_s)
            self._emit_retry(url, reason="backoff", sleep_ms=int(sleep_s * 1000), attempt=0)

        last_exc = None
        for i in range(self.max_attempts):
            try:
                resp = self.c.request(method, url, **kw)
                from_cache = bool(resp.extensions.get("from_cache"))
                revalidated = bool(resp.extensions.get("revalidated"))
                if from_cache and not revalidated:
                    self.bucket.refund(1.0)
                if resp.status_code in self.statuses and not from_cache:
                    if i < self.max_attempts - 1:
                        delay = self._compute_delay(i, resp)
                        self._emit_retry(url, reason="retry-after" if resp.headers.get("Retry-After") else "backoff",
                                         sleep_ms=int(delay * 1000), attempt=i+1, http_status=resp.status_code)
                        time.sleep(delay); continue
                return resp
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as e:
                last_exc = e
                if i < self.max_attempts - 1:
                    delay = self._compute_delay(i, None)
                    self._emit_retry(url, reason="conn-error", sleep_ms=int(delay * 1000), attempt=i+1)
                    time.sleep(delay); continue
                raise
        return resp  # type: ignore
```

---

### 2.3 Bootstrap: **build once**, pass a **client map** into the pipeline

**File:** `src/DocsToKG/ContentDownload/bootstrap.py`

```python
from DocsToKG.ContentDownload.httpx.hishel_build import build_hishel_transport, build_httpx_client_with_hishel
from DocsToKG.ContentDownload.httpx.client import RateRetryClient

def run_from_config(cfg, *, artifacts=None, ...):
    telemetry = build_run_telemetry(cfg)
    hishel_transport = build_hishel_transport(cfg.hishel)
    shared_client = build_httpx_client_with_hishel(cfg.http, hishel_transport)

    resolvers = build_resolvers(cfg.resolvers.order, cfg)

    clients = {}
    for name in cfg.resolvers.order:
        rcfg = getattr(cfg.resolvers, name, None)
        if not rcfg or not rcfg.enabled: continue
        clients[name] = RateRetryClient(
            resolver_name=name, client=shared_client,
            retry_policy=rcfg.retry, rate_policy=rcfg.rate_limit,
            telemetry=telemetry, run_id=cfg.run_id
        )

    pipeline = ResolverPipeline(
        resolvers=resolvers, session=clients,
        telemetry=telemetry, run_id=cfg.run_id,
        robots=cfg.robots, dlp=cfg.download,
    )
    # process artifacts...
```

---

### 2.4 Replace all **`requests`** call sites with the wrapper (`RateRetryClient`)

**Find them**:

```bash
grep -R "requests\.get(" -n src/DocsToKG/ContentDownload
grep -R "requests\.Session" -n src/DocsToKG/ContentDownload
grep -R "\.get(" -n src/DocsToKG/ContentDownload | grep -v httpx | grep -v RateRetryClient
```

**Change each call** in **execution/resolvers/legacy networking** to accept a `session` that **must** be your *per-resolver wrapper*, not a plain `requests.Session`:

* Head:

  ```python
  head = session.head(url, timeout=timeout_s)
  ```

* Get:

  ```python
  resp = session.get(url, timeout=timeout_s)
  ```

> If you have modules that still create their own `Session()`, delete those constructions and thread the `session` from the pipeline.

---

### 2.5 Emit cache-aware attempt tokens in `download_execution`

We already showed this above; ensure these two lines are present after GET:

```python
if from_cache and not revalidated:
    _emit(..., status="cache-hit", reason="ok")

if revalidated and resp.status_code == 304:
    _emit(..., status="http-304", reason="not-modified")
```

> Also ensure your `AttemptStatus` union (if using Literals) includes `"cache-hit"` and `"http-304"`, and `ReasonCode` includes `"not-modified"`.

---

### 2.6 Mark **legacy modules** as deprecated + route through the new path

* If you have a `networking.py` or `legacy_http.py`, change its methods to **delegate** to the new wrapper (or delete the file once all imports are migrated).
* Keep a small **compat shim** for two releases to avoid breaking internal tools; print a one-time warning on import.

---

### 2.7 Minimal tests to add/adjust

* **hishel flow** (MockTransport):

  * Run 1 → 200 with `ETag`, attempts: `http-get` + `http-200`.
  * Run 2 → **cache-hit**, attempts include `cache-hit`.
  * Run 3 (optional) → **304**, attempts include `http-304`.
* **search/replace guard**: a CI step that fails if `requests.get` or `requests.Session` is found under `src/DocsToKG/ContentDownload`.

```bash
# simple CI guard
! grep -R "requests\.get(" -n src/DocsToKG/ContentDownload
! grep -R "requests\.Session" -n src/DocsToKG/ContentDownload
```

---

## Quick “diff guide” summary

**TL;DR changes you’ll actually commit:**

1. **download_execution.py**

   * Add `verify_content_length` arg; remove manual write; call `atomic_write_stream`; catch `SizeMismatchError` and emit `size-mismatch`.
   * After GET: check `resp.extensions` and emit `cache-hit` / `http-304`.

2. **pipeline.py**

   * Pass `verify_content_length=self._dlp.verify_content_length` into `stream_candidate_payload`.

3. **httpx/hishel_build.py**

   * Add `build_hishel_transport` + `build_httpx_client_with_hishel`.

4. **httpx/client.py**

   * Ensure `TokenBucket` is lock-protected; wrap shared client; refund on `cache-hit`.

5. **bootstrap.py**

   * Build **one** shared httpx client; build per-resolver `RateRetryClient`s; pass client map into pipeline.

6. **Repo-wide:**

   * Remove/redirect any direct `requests` usages; CI guard to prevent reintroduction.

7. **Tests:**

   * Size-mismatch, happy path with CL, hishel 200→cache-hit→304.

---

I
