Awesome — below are **ready-to-apply unified diff patches** (so you can `git apply` them) **and** a **one-shot PR template** you can paste straight into GitHub.

> Notes before you apply
>
> * Paths assume your code lives under `src/DocsToKG/ContentDownload/…`.
> * If your atomic writer lives under a different module (e.g., `streaming.py`), adjust the import in patch **0001** accordingly.
> * If you already have `httpx/client.py`, patch **0004** will either create or replace it; same for `httpx/hishel_build.py`.
> * If your `api/types.py` already includes `"cache-hit"` and `"http-304"`, patch **0006** will be a no-op.

---

# 🧩 Unified diff patches

> Apply in order:
>
> ```bash
> git checkout -b chore/atomic-writer-httpx-hishel
> git apply 0001-download-execution-atomic-writer.patch
> git apply 0002-pipeline-verify-content-length.patch
> git apply 0003-hishel-builders.patch
> git apply 0004-rate-retry-client.patch
> git apply 0005-bootstrap-wire-shared-client.patch
> git apply 0006-extend-attempt-tokens.patch
> git apply 0007-ci-guard-requests.patch
> ```

---

## 0001-download-execution-atomic-writer.patch

```diff
*** a/src/DocsToKG/ContentDownload/download_execution.py
--- b/src/DocsToKG/ContentDownload/download_execution.py
@@
-from __future__ import annotations
-import os
-import time
-from typing import Optional
-
-from DocsToKG.ContentDownload.api.types import (
-    DownloadPlan,
-    DownloadStreamResult,
-)
-from DocsToKG.ContentDownload.telemetry import AttemptSink
-from DocsToKG.ContentDownload.api.exceptions import SkipDownload, DownloadError
+from __future__ import annotations
+import os
+import time
+from typing import Optional
+
+from DocsToKG.ContentDownload.api.types import DownloadPlan, DownloadStreamResult
+from DocsToKG.ContentDownload.telemetry import AttemptSink
+from DocsToKG.ContentDownload.api.exceptions import SkipDownload, DownloadError
+from DocsToKG.ContentDownload.io_utils import atomic_write_stream, SizeMismatchError
@@
-def stream_candidate_payload(
-    plan: DownloadPlan,
-    *,
-    session=None,  # your HTTP client, injected by pipeline
-    timeout_s: Optional[float] = None,
-    chunk_size: int = 1 << 20,
-    expected_len: Optional[int] = None,
-    telemetry: Optional[AttemptSink] = None,
-    run_id: Optional[str] = None,
-) -> DownloadStreamResult:
-    """
-    Stream the response body into a tmp file. Emit HEAD/GET attempts here.
-    Return DownloadStreamResult; do not rename to final path here.
-    """
-    url = plan.url
-
-    # HEAD (optional)
-    t0 = time.monotonic_ns()
-    head = session.head(url, allow_redirects=True, timeout=timeout_s)
-    elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
-    _emit(
-        telemetry,
-        run_id=run_id,
-        resolver=plan.resolver_name,
-        url=url,
-        verb="HEAD",
-        status="http-head",
-        http_status=head.status_code,
-        content_type=head.headers.get("Content-Type"),
-        elapsed_ms=elapsed_ms,
-    )
-
-    # GET
-    t0 = time.monotonic_ns()
-    resp = session.get(url, stream=True, allow_redirects=True, timeout=timeout_s)
-    elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
-    _emit(
-        telemetry,
-        run_id=run_id,
-        resolver=plan.resolver_name,
-        url=url,
-        verb="GET",
-        status="http-get",
-        http_status=resp.status_code,
-        content_type=resp.headers.get("Content-Type"),
-        elapsed_ms=elapsed_ms,
-    )
-
-    # Write to tmp in same directory as final target (final target path determined later)
-    tmp_dir = os.getcwd()  # replace with configured location
-    os.makedirs(tmp_dir, exist_ok=True)
-    tmp_path = os.path.join(tmp_dir, ".part-download.tmp")
-
-    bytes_written = 0
-    with open(tmp_path, "wb") as f:
-        for piece in resp.content.iter_chunked(chunk_size):
-            if piece:
-                f.write(piece)
-                bytes_written += len(piece)
-
-    # Optional: log a 200 event with size info
-    _emit(
-        telemetry,
-        run_id=run_id,
-        resolver=plan.resolver_name,
-        url=url,
-        verb="GET",
-        status="http-200",
-        http_status=resp.status_code,
-        content_type=resp.headers.get("Content-Type"),
-        bytes_written=bytes_written,
-        content_length_hdr=int(resp.headers.get("Content-Length", "0")) or None,
-    )
-
-    return DownloadStreamResult(
-        path_tmp=tmp_path,
-        bytes_written=bytes_written,
-        http_status=resp.status_code,
-        content_type=resp.headers.get("Content-Type"),
-    )
+def stream_candidate_payload(
+    plan: DownloadPlan,
+    *,
+    session,                                  # httpx-based RateRetryClient (required)
+    timeout_s: Optional[float] = None,
+    chunk_size: int = 1 << 20,
+    verify_content_length: bool = True,       # NEW: enforce CL when header present
+    telemetry: Optional[AttemptSink] = None,
+    run_id: Optional[str] = None,
+) -> DownloadStreamResult:
+    """
+    Stream the response body into a tmp file with atomic writer + CL enforcement.
+    Uses hishel metadata to emit cache-aware tokens.
+    """
+    url = plan.url
+
+    # Optional HEAD probe (policy/type checks)
+    try:
+        t0 = time.monotonic_ns()
+        head = session.head(url, timeout=timeout_s)
+        elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
+        _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
+              verb="HEAD", status="http-head", http_status=head.status_code,
+              content_type=head.headers.get("Content-Type"), elapsed_ms=elapsed_ms)
+    except Exception:
+        # Some servers 405 on HEAD; proceed without failing
+        pass
+
+    # GET via httpx + hishel
+    t0 = time.monotonic_ns()
+    resp = session.get(url, timeout=timeout_s)
+    elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
+
+    from_cache = bool(getattr(resp, "extensions", {}).get("from_cache"))
+    revalidated = bool(getattr(resp, "extensions", {}).get("revalidated"))
+
+    _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
+          verb="GET", status="http-get", http_status=resp.status_code,
+          content_type=resp.headers.get("Content-Type"), elapsed_ms=elapsed_ms)
+
+    # Cache-aware short-circuits
+    if from_cache and not revalidated:
+        _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
+              verb="GET", status="cache-hit", http_status=resp.status_code,
+              content_type=resp.headers.get("Content-Type"), reason="ok")
+    if revalidated and resp.status_code == 304:
+        _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
+              verb="GET", status="http-304", http_status=304,
+              content_type=resp.headers.get("Content-Type"), reason="not-modified")
+        return DownloadStreamResult(path_tmp="", bytes_written=0, http_status=304,
+                                    content_type=resp.headers.get("Content-Type"))
+
+    # Atomic write + CL enforcement
+    tmp_dir = os.getcwd()                                # or configured staging dir
+    os.makedirs(tmp_dir, exist_ok=True)
+    tmp_path = os.path.join(tmp_dir, ".part-download.tmp")
+    cl = resp.headers.get("Content-Length")
+    expected_len = int(cl) if (cl and cl.isdigit()) else None
+
+    try:
+        bytes_written = atomic_write_stream(
+            dest_path=tmp_path,
+            byte_iter=resp.iter_bytes(),
+            expected_len=(expected_len if verify_content_length else None),
+            chunk_size=chunk_size
+        )
+        _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
+              verb="GET", status="http-200", http_status=resp.status_code,
+              content_type=resp.headers.get("Content-Type"),
+              bytes_written=bytes_written, content_length_hdr=expected_len)
+        return DownloadStreamResult(path_tmp=tmp_path, bytes_written=bytes_written,
+                                    http_status=resp.status_code,
+                                    content_type=resp.headers.get("Content-Type"))
+    except SizeMismatchError:
+        _emit(telemetry, run_id=run_id, resolver=plan.resolver_name, url=url,
+              verb="GET", status="size-mismatch", http_status=resp.status_code,
+              content_type=resp.headers.get("Content-Type"), reason="size-mismatch",
+              content_length_hdr=expected_len)
+        raise DownloadError("size-mismatch")
```

---

## 0002-pipeline-verify-content-length.patch

```diff
*** a/src/DocsToKG/ContentDownload/pipeline.py
--- b/src/DocsToKG/ContentDownload/pipeline.py
@@
-            stream = stream_candidate_payload(
-                adj_plan,
-                session=client,
-                timeout_s=None,  # supply from config
-                chunk_size=self._dlp.chunk_size_bytes,
-                telemetry=self._telemetry,
-                run_id=self._run_id,
-            )
+            stream = stream_candidate_payload(
+                adj_plan,
+                session=client,
+                timeout_s=None,  # supply from config
+                chunk_size=self._dlp.chunk_size_bytes,
+                verify_content_length=self._dlp.verify_content_length,  # NEW
+                telemetry=self._telemetry,
+                run_id=self._run_id,
+            )
```

---

## 0003-hishel-builders.patch (NEW FILE)

```diff
*** /dev/null
--- b/src/DocsToKG/ContentDownload/httpx/hishel_build.py
@@
+from __future__ import annotations
+import httpx
+import hishel
+
+def build_hishel_transport(cfg_hishel) -> hishel.CacheTransport:
+    if cfg_hishel.backend == "file":
+        storage = hishel.FileStorage(
+            base_path=cfg_hishel.base_path,
+            ttl=cfg_hishel.ttl_seconds,
+            check_ttl_every=cfg_hishel.check_ttl_every_seconds,
+        )
+    elif cfg_hishel.backend == "sqlite":
+        import sqlite3
+        storage = hishel.SQLiteStorage(
+            connection=sqlite3.connect(cfg_hishel.sqlite_path),
+            ttl=cfg_hishel.ttl_seconds,
+        )
+    elif cfg_hishel.backend == "redis":
+        import redis
+        storage = hishel.RedisStorage(client=redis.Redis.from_url(cfg_hishel.redis_url),
+                                      ttl=cfg_hishel.ttl_seconds)
+    elif cfg_hishel.backend == "s3":
+        import boto3
+        storage = hishel.S3Storage(bucket_name=cfg_hishel.s3_bucket,
+                                   client=boto3.client("s3"),
+                                   ttl=cfg_hishel.ttl_seconds,
+                                   check_ttl_every=cfg_hishel.check_ttl_every_seconds)
+    else:
+        raise ValueError(f"Unknown hishel backend: {cfg_hishel.backend}")
+
+    controller = hishel.Controller(
+        force_cache=cfg_hishel.force_cache,
+        allow_heuristics=cfg_hishel.allow_heuristics,
+        allow_stale=cfg_hishel.allow_stale,
+        always_revalidate=cfg_hishel.always_revalidate,
+        cache_private=cfg_hishel.cache_private,
+        cacheable_methods=cfg_hishel.cacheable_methods,
+    )
+    return hishel.CacheTransport(transport=httpx.HTTPTransport(),
+                                 controller=controller, storage=storage)
+
+def build_httpx_client_with_hishel(cfg_http, transport) -> httpx.Client:
+    headers = {"User-Agent": cfg_http.user_agent}
+    if cfg_http.mailto and "mailto:" not in headers["User-Agent"]:
+        headers["User-Agent"] += f" (+mailto:{cfg_http.mailto})"
+    timeouts = httpx.Timeout(connect=cfg_http.timeout_connect_s,
+                             read=cfg_http.timeout_read_s,
+                             write=cfg_http.timeout_read_s,
+                             pool=cfg_http.timeout_read_s)
+    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
+    return httpx.Client(transport=transport, headers=headers, timeout=timeouts,
+                        limits=limits, verify=bool(cfg_http.verify_tls),
+                        proxies=cfg_http.proxies or None)
```

---

## 0004-rate-retry-client.patch (NEW FILE)

```diff
*** /dev/null
--- b/src/DocsToKG/ContentDownload/httpx/client.py
@@
+from __future__ import annotations
+import time
+import random
+import threading
+from typing import Optional
+import httpx
+from DocsToKG.ContentDownload.telemetry.build import RunTelemetry
+
+class TokenBucket:
+    def __init__(self, capacity: int, refill_per_sec: float, burst: int):
+        self.capacity = max(1, capacity)
+        self.refill = max(0.01, refill_per_sec)
+        self.burst = max(0, burst)
+        self.tokens = float(self.capacity + self.burst)
+        self.t0 = time.monotonic()
+        self._lock = threading.Lock()
+    def _tick(self):
+        now = time.monotonic()
+        dt = now - self.t0
+        self.t0 = now
+        self.tokens = min(self.capacity + self.burst, self.tokens + dt * self.refill)
+    def consume(self, amount=1.0) -> float:
+        with self._lock:
+            self._tick()
+            if self.tokens >= amount:
+                self.tokens -= amount
+                return 0.0
+            need = amount - self.tokens
+            return need / self.refill
+    def refund(self, amount=1.0):
+        with self._lock:
+            self._tick()
+            self.tokens = min(self.capacity + self.burst, self.tokens + amount)
+
+class RateRetryClient:
+    """
+    Per-resolver wrapper around a shared httpx.Client that enforces
+    token-bucket rate limiting, retries with Retry-After/backoff, and
+    emits retry attempts. hishel cache metadata is honored (refund token on cache-hit).
+    """
+    def __init__(self, resolver_name: str, client: httpx.Client, *,
+                 retry_policy, rate_policy, telemetry: Optional[RunTelemetry], run_id: Optional[str]):
+        self.name = resolver_name
+        self.c = client
+        self.t = telemetry
+        self.run_id = run_id
+        self.bucket = TokenBucket(rate_policy.capacity, rate_policy.refill_per_sec, rate_policy.burst)
+        self.statuses = set(int(x) for x in (retry_policy.retry_statuses or []))
+        self.base = max(0, int(retry_policy.base_delay_ms))
+        self.maxd = max(self.base, int(retry_policy.max_delay_ms))
+        self.jitter = max(0, int(retry_policy.jitter_ms))
+        self.max_attempts = max(1, int(retry_policy.max_attempts))
+
+    def head(self, url: str, **kw) -> httpx.Response:
+        return self._request("HEAD", url, **kw)
+    def get(self, url: str, **kw) -> httpx.Response:
+        return self._request("GET", url, **kw)
+
+    def _request(self, method: str, url: str, **kw) -> httpx.Response:
+        sleep_s = self.bucket.consume(1.0)
+        if sleep_s > 0:
+            time.sleep(sleep_s)
+            self._emit_retry(url, reason="backoff", sleep_ms=int(sleep_s * 1000), attempt=0)
+        last_exc = None
+        for i in range(self.max_attempts):
+            try:
+                resp = self.c.request(method, url, **kw)
+                from_cache = bool(resp.extensions.get("from_cache"))
+                revalidated = bool(resp.extensions.get("revalidated"))
+                if from_cache and not revalidated:
+                    self.bucket.refund(1.0)
+                if resp.status_code in self.statuses and not from_cache:
+                    if i < self.max_attempts - 1:
+                        delay = self._compute_delay(i, resp)
+                        self._emit_retry(url, reason="retry-after" if resp.headers.get("Retry-After") else "backoff",
+                                         sleep_ms=int(delay * 1000), attempt=i+1, http_status=resp.status_code)
+                        time.sleep(delay)
+                        continue
+                return resp
+            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError):
+                if i < self.max_attempts - 1:
+                    delay = self._compute_delay(i, None)
+                    self._emit_retry(url, reason="conn-error", sleep_ms=int(delay * 1000), attempt=i+1)
+                    time.sleep(delay)
+                    continue
+                raise
+        return resp  # type: ignore
+
+    def _compute_delay(self, attempt_idx: int, resp: Optional[httpx.Response]) -> float:
+        ra = 0.0
+        if resp is not None:
+            h = resp.headers.get("Retry-After")
+            if h:
+                try: ra = float(h)
+                except ValueError: ra = 0.0
+        backoff = min(self.maxd, self.base * (2 ** attempt_idx) if self.base > 0 else 0)
+        if self.jitter: backoff += random.randint(0, self.jitter)
+        return max(ra, backoff / 1000.0)
+
+    def _emit_retry(self, url: str, *, reason: str, sleep_ms: int, attempt: int, http_status: int | None = None):
+        if not self.t:
+            return
+        self.t.log_attempt(run_id=self.run_id, resolver=self.name, url=url,
+                           verb="GET", status="retry", http_status=http_status,
+                           reason=reason, elapsed_ms=sleep_ms, extra={"attempt": attempt})
```

---

## 0005-bootstrap-wire-shared-client.patch

```diff
*** a/src/DocsToKG/ContentDownload/bootstrap.py
--- b/src/DocsToKG/ContentDownload/bootstrap.py
@@
-from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
-from DocsToKG.ContentDownload.telemetry.build import build_run_telemetry
-from DocsToKG.ContentDownload.http.session import build_session
-from DocsToKG.ContentDownload.http.client import HttpClient
-from DocsToKG.ContentDownload.resolvers import build_resolvers
-from DocsToKG.ContentDownload.pipeline import ResolverPipeline
+from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
+from DocsToKG.ContentDownload.telemetry.build import build_run_telemetry
+from DocsToKG.ContentDownload.httpx.hishel_build import build_hishel_transport, build_httpx_client_with_hishel
+from DocsToKG.ContentDownload.httpx.client import RateRetryClient
+from DocsToKG.ContentDownload.resolvers import build_resolvers
+from DocsToKG.ContentDownload.pipeline import ResolverPipeline
@@
 def run_from_config(
     cfg: ContentDownloadConfig,
     *,
     artifacts=None,
     record_html_paths: bool = True,
 ) -> None:
-    telemetry = build_run_telemetry(cfg)
-    session = build_session(cfg.http)
-
-    # Instantiate resolvers
-    resolvers = build_resolvers(cfg.resolvers.order, cfg)
-
-    # Build a per-resolver HTTP client with policies
-    clients = {}
-    for name in cfg.resolvers.order:
-        rcfg = getattr(cfg.resolvers, name, None)
-        if not rcfg or not rcfg.enabled:
-            continue
-        clients[name] = HttpClient(
-            resolver_name=name,
-            session=session,
-            retry_policy=rcfg.retry,
-            rate_policy=rcfg.rate_limit,
-            timeouts=cfg.http,  # has timeout_connect_s, timeout_read_s
-            telemetry=telemetry,
-            run_id=cfg.run_id,
-        )
+    telemetry = build_run_telemetry(cfg)
+    # Shared hishel-enabled client
+    hishel_transport = build_hishel_transport(cfg.hishel)
+    shared_client = build_httpx_client_with_hishel(cfg.http, hishel_transport)
+
+    # Resolvers
+    resolvers = build_resolvers(cfg.resolvers.order, cfg)
+
+    # Per-resolver wrappers (rate+retry+telemetry)
+    clients = {}
+    for name in cfg.resolvers.order:
+        rcfg = getattr(cfg.resolvers, name, None)
+        if not rcfg or not rcfg.enabled:
+            continue
+        clients[name] = RateRetryClient(
+            resolver_name=name,
+            client=shared_client,
+            retry_policy=rcfg.retry,
+            rate_policy=rcfg.rate_limit,
+            telemetry=telemetry,
+            run_id=cfg.run_id,
+        )
@@
     pipeline = ResolverPipeline(
         resolvers=resolvers,
-        session=clients,            # pass client map; pipeline will pick per plan.resolver_name
+        session=clients,            # per-resolver httpx wrapper map
         telemetry=telemetry,
         run_id=cfg.run_id,
         robots=cfg.robots,          # if your pipeline/prepare_* uses robots
         dlp=cfg.download,           # download policy knobs (atomic, chunk, verify)
     )
```

---

## 0006-extend-attempt-tokens.patch (add tokens if not present)

```diff
*** a/src/DocsToKG/ContentDownload/api/types.py
--- b/src/DocsToKG/ContentDownload/api/types.py
@@
-from typing import Literal
+from typing import Literal
@@
-AttemptStatus = Literal[
-    "http-head",
-    "http-get",
-    "http-200",
-    "http-304",
-    "robots-fetch",
-    "robots-disallowed",
-    "retry",
-    "size-mismatch",
-    "content-policy-skip",
-    "download-error",
-]
+AttemptStatus = Literal[
+    "http-head",
+    "http-get",
+    "http-200",
+    "http-304",
+    "cache-hit",
+    "robots-fetch",
+    "robots-disallowed",
+    "retry",
+    "size-mismatch",
+    "content-policy-skip",
+    "download-error",
+]
@@
-ReasonCode = Literal[
-    "ok",
-    "not-modified",
-    "retry-after",
-    "backoff",
-    "robots",
-    "policy-type",
-    "policy-size",
-    "timeout",
-    "conn-error",
-    "tls-error",
-    "too-large",
-    "unexpected-ct",
-    "size-mismatch",
-]
+ReasonCode = Literal[
+    "ok",
+    "not-modified",
+    "retry-after",
+    "backoff",
+    "robots",
+    "policy-type",
+    "policy-size",
+    "timeout",
+    "conn-error",
+    "tls-error",
+    "too-large",
+    "unexpected-ct",
+    "size-mismatch",
+]
```

---

## 0007-ci-guard-requests.patch (NEW CI workflow)

```diff
*** /dev/null
--- b/.github/workflows/guard-requests.yml
@@
+name: Guard against direct requests usage
+
+on:
+  pull_request:
+    paths:
+      - "src/DocsToKG/ContentDownload/**"
+      - ".github/workflows/guard-requests.yml"
+
+jobs:
+  guard:
+    runs-on: ubuntu-latest
+    steps:
+      - name: Checkout
+        uses: actions/checkout@v4
+      - name: Forbid direct requests usage
+        shell: bash
+        run: |
+          set -euo pipefail
+          ! grep -R "requests\.get(" -n src/DocsToKG/ContentDownload || (echo "Found direct requests.get usage" && exit 1)
+          ! grep -R "requests\.Session" -n src/DocsToKG/ContentDownload || (echo "Found requests.Session usage" && exit 1)
```

---

# 🧾 One-shot PR template (paste into GitHub)

```markdown
# PR: Atomic Writer + httpx+hishel Unification

## Summary
This PR:
1) Replaces the ad-hoc write loop in `download_execution` with the **atomic writer** (`fsync` + `os.replace`) and **Content-Length enforcement**.
2) Unifies all HTTP calls behind one shared **`httpx.Client` with hishel CacheTransport**, wrapped per resolver by **RateRetryClient** (rate-limit + retry + telemetry).

**New/updated attempt tokens:** `cache-hit` (pure cache serve) and `http-304` (revalidated, not modified).

---

## Why
- **Correctness:** atomic commit prevents partial finals; CL enforcement detects truncation deterministically.
- **Performance:** connection reuse and cache hits reduce latency; 304 revalidation avoids body transfer.
- **Consistency:** single HTTP path means uniform rate limits, retries, and telemetry.
- **Observability:** cache-aware tokens light up our dashboard panels (cache hit ratio, revalidation ratio).

---

## Changes
- `download_execution.py`: use `atomic_write_stream`, add `verify_content_length`, emit `cache-hit`/`http-304`.
- `pipeline.py`: pass `verify_content_length` from `DownloadPolicy`.
- `httpx/hishel_build.py`: new builders for hishel transport + shared `httpx.Client`.
- `httpx/client.py`: new `RateRetryClient` with thread-safe token bucket; refund on pure cache hit.
- `bootstrap.py`: wire shared client + per-resolver wrappers into `ResolverPipeline`.
- `api/types.py`: extend `AttemptStatus` to include `cache-hit` (if not already present).
- CI: `.github/workflows/guard-requests.yml` forbids re-introduction of `requests`.

---

## Checklist
### Pre-flight
- [ ] No remaining direct `requests` usages (`grep` guard included).
- [ ] Pydantic v2 config remains the single source for `http`, `hishel`, `download`.

### Atomic writer
- [ ] Manual write loop replaced with `atomic_write_stream`.
- [ ] CL mismatch raises `DownloadError("size-mismatch")` and emits `size-mismatch` attempt.
- [ ] Happy path emits `http-200` with `bytes_written` and `content_length_hdr`.

### httpx + hishel
- [ ] Shared `httpx.Client` with hishel transport built once.
- [ ] Per-resolver `RateRetryClient` wrappers used everywhere (`head`/`get`).
- [ ] Cache-aware tokens emitted: `cache-hit`, `http-304`.
- [ ] Token bucket protects concurrency; refunds on pure cache hits.

### Tests
- [ ] Happy path + correct CL.
- [ ] Size mismatch.
- [ ] Cache hit (no network).
- [ ] 304 revalidation (skip).
- [ ] CI guards pass.

### Docs & Changelog
- [ ] `ARCHITECTURE_hishel.md` updated to reflect unified path & new tokens.
- [ ] `ARCHITECTURE.md` (execution) updated to state atomic writer is canonical.
- [ ] Changelog entry added.

---

## Testing Plan
- **Unit:** simulate GET 200 with `Content-Length` → assert bytes & attempts; simulate short stream → assert `size-mismatch`.
- **Cache:** Mock hishel transport → run1: 200 (ETag) → run2: pure cache `cache-hit` → run3: revalidate 304.
- **Integration:** run a small artifact list; confirm attempts CSV shows `http-get/http-200` and (on rerun) `cache-hit/http-304`.
- **CI:** verify grep guards catch any `requests` reintroduction.

---

## Rollout / Rollback
- **Rollout:** defaults unchanged. If a provider has broken CL headers, temporarily run with `download.verify_content_length=false`.
- **Rollback:** set `hishel.enabled=false` to bypass caching while keeping httpx; atomic writer remains safe to keep.

---

## Risks & Mitigations
- **CL header inaccuracies** → handled via `verify_content_length` toggle & telemetry; dashboards will highlight spikes.
- **Collector down** (OTel) → best-effort; CSV/JSONL remain authoritative.
- **Unknown edge domains** (HEAD 405) → we fall back to GET; no functional impact.

---

## Screenshots / Evidence
- Attempts CSV snippet demonstrating `http-200`, `cache-hit`, `http-304`.
- Grafana panels for **Cache Hit Ratio** and **Revalidation Ratio** trending as expected on reruns.

```

---

If you want me to tailor the hunks to your *exact* current function signatures (to ensure `git apply` is 100% clean), paste the current `stream_candidate_payload` content and the block in `pipeline.py` where it’s called, and I’ll regenerate precise diffs on top.
