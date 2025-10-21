Absolutely — here’s an **iron-clad PR checklist** with a **step-by-step swap-out guide**, plus extra concrete examples and guardrails so there’s no ambiguity.

---

# PR: Atomic Writer + httpx+hishel Unification

> Goal:
>
> 1. Replace the ad-hoc temp-file write loop with the **atomic writer** + **Content-Length enforcement**.
> 2. Route **all HTTP GETs** through **one shared `httpx.Client` with hishel CacheTransport**, wrapped by a per-resolver **RateRetryClient** (rate-limit + retry + telemetry).
>    Status target: **production-ready**, test-backed, CI-guarded.

---

## Why this change (short)

* **Correctness**: atomic write (`fsync` + `os.replace`) guarantees no partial finals; CL verification catches truncation.
* **Performance & kindness**: a single, shared `httpx` pool + hishel yields connection reuse, cache hits, and cheap 304 revalidations.
* **Observability**: cache-aware tokens (`cache-hit`, `http-304`) make behavior obvious in CSV & dashboards.
* **Uniformity**: a single HTTP path keeps rate-limits, retries, and telemetry consistent across resolvers.

---

## PR Checklist (copy into the PR description)

### 0) Pre-flight

* [ ] Confirm Pydantic v2 config is the single source for `http`, `hishel`, and `download` settings (no legacy `DownloadConfig` code paths used for these).
* [ ] Identify any direct `requests` usages:

  ```bash
  grep -R "requests\.get(" -n src/DocsToKG/ContentDownload || true
  grep -R "requests\.Session" -n src/DocsToKG/ContentDownload || true
  ```

  Record locations; all will be removed/redirected.

---

### 1) Atomic writer swap-in (download_execution)

**Files:** `src/DocsToKG/ContentDownload/download_execution.py` (and/or the canonical execution module)

* [ ] Add imports:

  ```python
  from DocsToKG.ContentDownload.io_utils import atomic_write_stream, SizeMismatchError
  ```

* [ ] Update function signature for streaming:

  ```python
  def stream_candidate_payload(plan: DownloadPlan, *,
      session,                              # httpx wrapper (required)
      timeout_s: Optional[float] = None,
      chunk_size: int = 1 << 20,
      verify_content_length: bool = True,   # NEW
      telemetry: Optional[AttemptSink] = None,
      run_id: Optional[str] = None,
  ) -> DownloadStreamResult:
  ```

* [ ] **Replace** the manual write loop with `atomic_write_stream(...)`:

  ```python
  resp = session.get(plan.url, stream=True, timeout=timeout_s)
  cl = resp.headers.get("Content-Length")
  expected_len = int(cl) if (cl and cl.isdigit()) else None

  try:
      bytes_written = atomic_write_stream(
          dest_path=tmp_path,
          byte_iter=resp.iter_bytes(),              # httpx iterator
          expected_len=(expected_len if verify_content_length else None),
          chunk_size=chunk_size
      )
      _emit(... status="http-200", bytes_written=bytes_written, content_length_hdr=expected_len)
      return DownloadStreamResult(path_tmp=tmp_path, bytes_written=bytes_written, http_status=resp.status_code, content_type=resp.headers.get("Content-Type"))
  except SizeMismatchError:
      _emit(... status="size-mismatch", reason="size-mismatch", content_length_hdr=expected_len)
      raise DownloadError("size-mismatch")
  ```

* [ ] **Emit cache-aware tokens** after GET handshake:

  ```python
  from_cache = bool(resp.extensions.get("from_cache"))
  revalidated = bool(resp.extensions.get("revalidated"))
  _emit(... status="http-get", http_status=resp.status_code, ...)

  if from_cache and not revalidated:
      _emit(... status="cache-hit", reason="ok")
  elif revalidated and resp.status_code == 304:
      _emit(... status="http-304", reason="not-modified")
      return DownloadStreamResult(path_tmp="", bytes_written=0, http_status=304, content_type=resp.headers.get("Content-Type"))
  ```

* [ ] **Pipeline** passes `verify_content_length` from config:

  ```python
  stream = stream_candidate_payload(
      plan,
      session=client,
      timeout_s=None,
      chunk_size=self._dlp.chunk_size_bytes,
      verify_content_length=self._dlp.verify_content_length,
      telemetry=self._telemetry,
      run_id=self._run_id,
  )
  ```

---

### 2) Unify network stack (httpx + hishel everywhere)

**Files:**

* `src/DocsToKG/ContentDownload/httpx/hishel_build.py` (NEW/UPDATED builder)

* `src/DocsToKG/ContentDownload/httpx/client.py` (RateRetryClient)

* `src/DocsToKG/ContentDownload/bootstrap.py` (wire shared client + per-resolver wrappers)

* *Any legacy networking modules must be removed or delegate to the wrapper.*

* [ ] Create the hishel transport and shared `httpx.Client`:

  ```python
  transport = hishel.CacheTransport(
      transport=httpx.HTTPTransport(),
      controller=hishel.Controller(... from cfg.hishel ...),
      storage=hishel.FileStorage(base_path=cfg.hishel.base_path, ...),
  )
  shared_client = httpx.Client(
      transport=transport,
      headers={"User-Agent": ua}, timeout=httpx.Timeout(...), limits=httpx.Limits(...),
      verify=bool(cfg.http.verify_tls), proxies=cfg.http.proxies or None,
  )
  ```

* [ ] Wrap shared client **per resolver**:

  ```python
  clients[name] = RateRetryClient(
      resolver_name=name,
      client=shared_client,
      retry_policy=rcfg.retry,
      rate_policy=rcfg.rate_limit,
      telemetry=telemetry,
      run_id=cfg.run_id,
  )
  ```

* [ ] Ensure **TokenBucket** in `RateRetryClient` is **thread-safe** (mutex) and **refund** on pure cache hit:

  ```python
  if from_cache and not revalidated:
      self.bucket.refund(1.0)
  ```

* [ ] **Replace all `requests` call sites** to use the provided `session`:

  * `requests.get(...)` → `session.get(url, timeout=...)`
  * `requests.Session()` creation → **delete**; session is injected from pipeline

* [ ] Keep **HEAD** using the same wrapper:

  ```python
  head = session.head(url, timeout=...)  # wrapper → httpx → hishel path
  ```

---

### 3) CI guardrails

* [ ] Add a CI step that **fails** if `requests` sneaks back in:

  ```bash
  # .github/workflows/lint.yml (as a step)
  - name: Forbid direct requests usage
    run: |
      ! grep -R "requests\.get(" -n src/DocsToKG/ContentDownload || (echo "Found direct requests.get" && exit 1)
      ! grep -R "requests\.Session" -n src/DocsToKG/ContentDownload || (echo "Found requests.Session" && exit 1)
  ```

* [ ] (Optional) Add a CI step to pin the **atomic writer** usage (defense in depth):

  ```bash
  # fail if code writes a manual loop in execution module
  ! grep -R "open(.*,\"wb\")" -n src/DocsToKG/ContentDownload/download_execution.py || true
  ```

---

### 4) Tests (must-haves)

* [ ] **Happy path + CL OK**: GET 200 with `Content-Length`, atomic writer runs, final bytes match CL; attempt `http-200` includes `bytes_written` & `content_length_hdr`.
* [ ] **CL mismatch**: declare `Content-Length: 500000`, stream 100k → attempt `size-mismatch`; no final file; raises `DownloadError("size-mismatch")`.
* [ ] **Cache hit**: hishel `from_cache==True`, `revalidated==False` → attempt `cache-hit`; no token consumption (refund tested via bucket state if observable).
* [ ] **304 path**: revalidated 304 → attempt `http-304` + outcome skip; no file write.
* [ ] **Search guard**: verify CI grep rules pass.
* [ ] **No direct `requests`**: repository grep returns empty.

---

### 5) Docs & CHANGELOG

* [ ] Update `ARCHITECTURE_hishel.md` to reflect that **all GETs** are now through httpx+hishel; mention `cache-hit` and `http-304`.
* [ ] Update `ARCHITECTURE.md` “Execution” to state that **atomic_write_stream** is the canonical write path with CL enforcement.
* [ ] CHANGELOG: “Swapped manual temp write for atomic writer + CL verify; unified httpx+hishel path; new attempt tokens `cache-hit`, `http-304`; CI guards against requests.”

---

## The swap-out explained (super explicit)

### A) Manual write loop → Atomic writer

**Before**

```python
resp = session.get(url, stream=True, ...)
tmp_path = os.path.join(tmp_dir, ".part.tmp")
bytes_written = 0
with open(tmp_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if not chunk: continue
        f.write(chunk); bytes_written += len(chunk)
# no fsync; later move; no CL check
```

**After**

```python
resp = session.get(url, stream=True, ...)
cl = resp.headers.get("Content-Length")
expected_len = int(cl) if (cl and cl.isdigit()) else None

bytes_written = atomic_write_stream(
    dest_path=tmp_path,
    byte_iter=resp.iter_bytes(),                # httpx iterator
    expected_len=(expected_len if verify_content_length else None),
    chunk_size=chunk_size
)
# inside: write → fsync → verify size (if expected) → os.replace(tmp, dest) is done in finalize
```

**Why better**

* **Durability**: `fsync` flushes; `os.replace` is atomic → never half-written finals.
* **Integrity**: size check catches truncation immediately.
* **Reusability**: a single function handles all peculiarities (chunking, exceptions, cleanup).

### B) `requests.*` calls → `RateRetryClient` (httpx + hishel)

**Before**

```python
import requests
s = requests.Session()
resp = s.get(url, timeout=(10,60))
```

**After**

```python
# obtain per-resolver client from pipeline: session
resp = session.get(url, timeout=timeout_s)
# under the hood:
#  - uses shared httpx.Client (connection reuse)
#  - hishel cache (304 & cache-hit behavior)
#  - rate limit + retry (+ telemetry on retries)
```

**Why better**

* **One HTTP path**: policy (rate/retry), caching, telemetry, and timeouts are uniform.
* **Cache wins**: `cache-hit` → zero network + refund token; `http-304` avoids bodies.
* **Cleaner code**: no per-module sessions or duplicate plumbing.

---

## Extra examples (edge cases you’ll likely hit)

### 1) HEAD 405 (or HEAD unreliable)

```python
try:
    head = session.head(url, timeout=timeout_s)
    _emit(... status="http-head", http_status=head.status_code, content_type=head.headers.get("Content-Type"))
except httpx.HTTPError:
    # Skip HEAD; proceed to GET — still safe because GET is cached/retried and type policies can check after first chunk.
    pass
```

### 2) Missing `Content-Length` (chunked transfer)

* `expected_len = None` → atomic writer **skips** size check.
* Integrity still preserved by atomic move and optional PDF tail check.

### 3) Redirects

* httpx follows redirects by default; hishel keys cache on the final URL.
* Emit `http-get` on the response you receive; you can add an attribute `redirects=len(resp.history)` if you want to track it.

### 4) 206 Partial Content (should not happen)

* Treat as error: emit `download-error` + reason `unexpected-ct` or `unexpected-status`; **do not** keep file.

### 5) S3 or cross-device moves (if/when you switch)

* Keep temp file under the same mount as the final path to preserve atomicity.
* If not possible → fallback: write to CAS staging on the same device, then handle S3 upload in a separate step; catalog records `s3://` URI after a successful put.

---

## “Definition of Done” (acceptance)

* All streaming paths in `download_execution` call **atomic_write_stream**; manual open/write loops are gone.
* All HTTP paths (HEAD/GET) in execution & resolvers accept an injected **RateRetryClient**; there is **no direct `requests` usage** under `src/DocsToKG/ContentDownload`.
* Attempt CSV shows **cache-aware tokens** where applicable (`cache-hit`, `http-304`), and **size-mismatch** when CL fails.
* Tests cover **happy path**, **size mismatch**, **cache-hit**, **304**; CI grep guard passes.
* Docs updated; CHANGELOG entry added.

---

## Rollout / rollback

* **Rollout**: ship behind config switches you already have:

  * `download.verify_content_length` (default **true**),
  * `hishel.enabled` (default **true**).
* **Rollback**: if needed, temporarily set `hishel.enabled=false` to bypass caching while keeping httpx route; set `verify_content_length=false` to ignore CL while investigating origin issues. (Atomic write remains; that’s safe to keep.)

---

If you’d like, I can generate a **unified diff** patch for each file in this checklist (ready to apply), and a **one-shot PR template** containing the checklist as GitHub markdown.
