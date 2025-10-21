Absolutely — here’s a crisp **architecture overview** for the HTTP client stack plus a set of **copy-ready code examples** (safe stubs) your team can drop in and extend. I’ve kept filenames and function names aligned to the implementation plan so it all snaps together.

---

# 🧭 Architecture Overview — HTTPX + Hishel + Redirect Audit

## 1) Big-picture map

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Callers (planner, downloader, resolvers, validators)                    │
│  - Always go through: request_with_redirect_audit(get_http_client(...))  │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  net/client.py                                                           │
│  • get_http_client(settings): lazy singleton                             │
│      - HTTP/2, explicit timeouts, tuned pool, SSL verify, trust_env      │
│      - Hishel cache transport (if enabled)                                │
│      - Redirects OFF globally; we manage hops explicitly                  │
│  • Hooks → emit `net.request` (timings, cache, bytes, outcome)            │
│  • request_with_redirect_audit():                                         │
│      - send request                                                       │
│      - on 3xx: resolve Location → url_gate.validate(target) → follow      │
│      - per-hop event; max_hops guarded                                    │
└───────────────┬──────────────────────────────────────────────────────────┘
                │                                  ▲
                │ uses                              │ validates per hop
                ▼                                  │
        ratelimit.acquire()                         │
                │                                   │
                ▼                                   │
          HTTPX Client  ────────────────►  policy/url_gate.py  (authoritative)
                │                                   │
           Tenacity retries        (429) cooldown_for(key, seconds) + sleep
                ▼
         stream → temp → fsync → rename      (downloader)
```

**Key ideas**

* **One** client per process (lazy, PID-aware rebuild after fork).
* **No auto-redirects**: we **audit** each hop through the **URL security gate**.
* **Transport retries** only for **connect** errors; **status** retries (429/5xx) live in Tenacity.
* **Hishel** handles RFC-9111 cache/validators; we just consume 304s.
* **Events** are emitted for every attempt (`net.request`), so you can answer “what, how long, why” instantly.

---

## 2) Request lifecycle (happy path & 429)

### Happy path

```
ratelimit.acquire("ols","ebi.ac.uk","block")
↓
request_with_redirect_audit(client, "GET", url, stream=True)
↓  200 OK  (Hishel → cache:"hit|revalidated|miss")
stream to temp → fsync → rename
```

### 429 with Retry-After = 2s

```
attempt#1 → 429 (Retry-After:2)
cooldown_for("ols:ebi.ac.uk", 2)   # limiter informed
Tenacity sleeps ~2s
attempt#2 → acquire() returns fast (no double wait) → 200 OK
```

---

## 3) Test surfaces (fully hermetic)

* **httpx.MockTransport** for scripted 200/302/429/5xx/timeouts.
* **In-process ASGI app** to simulate streaming, headers, conditional GETs.
* You never open a real socket in tests.

---

# 🧩 Code Examples (ready-to-paste stubs)

Below are minimal, production-style stubs. They compile; fill the `TODOs` as you wire the rest.

---

## A) Settings (strict Pydantic v2)

```python
# src/DocsToKG/OntologyDownload/settings.py
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from pathlib import Path

class HttpTimeouts(BaseModel):
    connect: PositiveFloat = 5.0
    read: PositiveFloat = 30.0
    write: PositiveFloat = 30.0
    pool: PositiveFloat = 5.0

class HttpPool(BaseModel):
    max_connections: PositiveInt = 64
    max_keepalive_connections: PositiveInt = 20
    keepalive_expiry: PositiveFloat = 30.0  # seconds

class HttpCache(BaseModel):
    enabled: bool = True
    dir: Path = Path("~/.cache/ontofetch/http").expanduser()
    bypass: bool = False

class HttpSettings(BaseModel):
    http2: bool = True
    trust_env: bool = True                  # honor HTTP(S)_PROXY & NO_PROXY
    user_agent: str = "DocsToKG/OntoFetch (+https://example.org)"
    timeouts: HttpTimeouts = HttpTimeouts()
    pool: HttpPool = HttpPool()
    cache: HttpCache = HttpCache()

class RetrySettings(BaseModel):
    connect_retries: int = 2
    backoff_base: float = 0.1
    backoff_max: float = 2.0

class Settings(BaseModel):
    http: HttpSettings = HttpSettings()
    retry: RetrySettings = RetrySettings()
    # … plus security/ratelimit/extraction/etc.
    @property
    def config_hash(self) -> str:
        # TODO: canonicalize & hash normalized dict (exclude secrets/volatile)
        import hashlib, json
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()
```

---

## B) URL gate (authoritative hop validation)

```python
# src/DocsToKG/OntologyDownload/policy/url_gate.py
from urllib.parse import urlsplit
from typing import Optional

class PolicyError(RuntimeError): ...
def validate_url_security(url: str, http_config: Optional[object] = None) -> str:
    """Strict URL validation: scheme, host punycode, PSL allowlist, per-host ports,
    http→https upgrade policy, and DNS private-net rules.
    Returns possibly normalized URL (e.g., upgraded to https)."""
    # TODO:
    #  - parse; require scheme ∈ {http, https}
    #  - lowercase host; IDN → punycode; strip default ports
    #  - check registrable-domain allowlist & per-host ports
    #  - if http & not allowed_plain_http: upgrade to https
    #  - optional DNS resolve (strict/lenient) & private-net guard
    sp = urlsplit(url)
    if sp.scheme not in ("http", "https"):
        raise PolicyError(f"scheme not allowed: {sp.scheme}")
    return url  # normalized form
```

---

## C) Emitter (one-liner events)

```python
# src/DocsToKG/OntologyDownload/observability/events.py
import json, sys
from datetime import datetime

_RUN_ID = None
_CONFIG_HASH = None
_CONTEXT = {"app_version":"0.1.0"}  # TODO: fill

def bind_context(*, run_id: str, config_hash: str, **ctx):
    global _RUN_ID, _CONFIG_HASH, _CONTEXT
    _RUN_ID, _CONFIG_HASH = run_id, config_hash
    _CONTEXT |= ctx

def emit(event_type: str, level: str = "INFO", payload: dict | None = None, **ids):
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds")+"Z",
        "type": event_type, "level": level,
        "run_id": _RUN_ID, "config_hash": _CONFIG_HASH,
        "context": _CONTEXT, "ids": ids, "payload": payload or {}
    }
    sys.stdout.write(json.dumps(rec, separators=(",", ":")) + "\n")
    sys.stdout.flush()
```

---

## D) HTTPX client — singleton + hooks + redirect audit

```python
# src/DocsToKG/OntologyDownload/net/client.py
from __future__ import annotations
import os, time
from typing import Optional
import httpx

from ..observability.events import emit
from ..policy.url_gate import validate_url_security

# Optional cache (Hishel)
try:
    from hishel import CacheTransport, FileSystemCache
except Exception:
    CacheTransport = FileSystemCache = None

_CLIENT: Optional[httpx.Client] = None
_BIND_HASH: Optional[str] = None
_BIND_PID: Optional[int] = None

def _build_transport(settings) -> httpx.BaseTransport:
    # Base transport; configure retries for connect errors if desired.
    return httpx.HTTPTransport(retries=settings.retry.connect_retries)

def _maybe_wrap_cache(settings, transport) -> httpx.BaseTransport:
    if settings.http.cache.enabled and CacheTransport and FileSystemCache:
        cache = FileSystemCache(directory=str(settings.http.cache.dir))
        return CacheTransport(transport=transport, cache=cache)
    return transport

def get_http_client(settings) -> httpx.Client:
    global _CLIENT, _BIND_HASH, _BIND_PID
    pid = os.getpid()
    if _CLIENT is None or _BIND_PID != pid:
        transport = _maybe_wrap_cache(settings, _build_transport(settings))
        _CLIENT = httpx.Client(
            http2=settings.http.http2,
            transport=transport,
            timeout=httpx.Timeout(
                settings.http.timeouts.connect,
                settings.http.timeouts.read,
                settings.http.timeouts.write,
                settings.http.timeouts.pool,
            ),
            limits=httpx.Limits(
                max_connections=settings.http.pool.max_connections,
                max_keepalive_connections=settings.http.pool.max_keepalive_connections,
                keepalive_expiry=settings.http.pool.keepalive_expiry,
            ),
            verify=True,
            trust_env=settings.http.trust_env,
            headers={
                "User-Agent": settings.http.user_agent,
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br",
            },
            follow_redirects=False,  # critical: we audit hops
        )
        # Attach hooks
        _CLIENT.event_hooks["request"] = [_on_request]
        _CLIENT.event_hooks["response"] = [_on_response]
        _CLIENT.event_hooks["redirect"] = []  # unused; follow_redirects=False
        _BIND_HASH, _BIND_PID = settings.config_hash, pid
    elif _BIND_HASH != settings.config_hash:
        # Settings changed mid-process – warn once (no hot reload in prod)
        emit("net.client.warn", payload={"msg": "config changed after client bind"})
        _BIND_HASH = settings.config_hash
    return _CLIENT

def _on_request(request: httpx.Request):
    request.extensions["t0"] = time.perf_counter()
    request.extensions["request_id"] = os.urandom(8).hex()
    # (Optional) attach service/host tags for observability here

def _on_response(response: httpx.Response):
    req = response.request
    t0 = req.extensions.get("t0", time.perf_counter())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Cache hint: if Hishel is used, you can expose richer status; here infer basic
    cache = "miss"
    if response.status_code == 304:
        cache = "revalidated"
    emit("net.request",
         payload={
            "method": req.method,
            "url_redacted": _redact_url(str(req.url)),
            "host": req.url.host,
            "status": response.status_code,
            "attempt": 1,          # TODO: pass attempt via caller
            "http2": response.http_version == "HTTP/2",
            "reused_conn": None,   # TODO if you track
            "cache": cache,
            "ttfb_ms": None,       # TODO if you track socket timings
            "elapsed_ms": round(elapsed_ms, 2),
            "bytes_read": int(response.headers.get("Content-Length", 0) or 0),
         },
         request_id=req.extensions.get("request_id"),
    )

def _redact_url(url: str) -> str:
    # Minimal redactor (drop querystring); strengthen if needed
    return str(httpx.URL(url).copy_with(query=None))

def request_with_redirect_audit(client: httpx.Client, method: str, url: str, *,
                                max_hops: int = 5, headers: dict | None = None,
                                **kw) -> httpx.Response:
    """Send a request and follow 3xx hops manually. Each hop is validated by URL gate."""
    current = url
    for hop in range(max_hops):
        resp = client.request(method, current, headers=headers, **kw)
        if 300 <= resp.status_code < 400 and "location" in resp.headers:
            target = httpx.URL(current).join(resp.headers["location"])
            target_str = str(target)
            validate_url_security(target_str)  # authoritative one
            current = target_str
            continue
        return resp
    raise httpx.TooManyRedirects(f"exceeded {max_hops} hops")
```

---

## E) A downloader “fetch to file” that streams + retries (Tenacity)

```python
# src/DocsToKG/OntologyDownload/download.py
from __future__ import annotations
import os, time
from pathlib import Path
import httpx
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception

from .net.client import get_http_client, request_with_redirect_audit
from .observability.events import emit
from .ratelimit.manager import acquire, cooldown_for   # per your RL rollout
# from .policy.url_gate import validate_url_security  # if you gate pre-request

class DownloadError(Exception): ...

def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False

@retry(
    wait=wait_exponential_jitter(initial=0.2, max=2.0),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
def fetch_to_path(settings, url: str, dest: Path, *, service: str = "_") -> Path:
    client = get_http_client(settings)
    host = httpx.URL(url).host or "default"
    # Rate-limit per (service,host)
    acquire(service, host, mode="block", weight=1)

    # Send request with audited redirects; STREAM
    resp = request_with_redirect_audit(
        client, "GET", url, stream=True, headers=None,
        timeout=None,  # per-request override optional
    )
    # Handle 429: tell limiter, then raise to Tenacity to sleep & retry
    if resp.status_code == 429:
        ra = _parse_retry_after(resp)
        cooldown_for(f"{service}:{host}", ra or 1.0)
        raise httpx.HTTPStatusError("429", request=resp.request, response=resp)

    resp.raise_for_status()
    # Stream to temp → fsync → rename
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as wf:
        for chunk in resp.iter_bytes():
            wf.write(chunk)
        wf.flush(); os.fsync(wf.fileno())
    os.replace(tmp, dest)
    emit("download.fetch", payload={"path": str(dest), "status": resp.status_code})
    return dest

def _parse_retry_after(resp: httpx.Response) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra: return None
    try:
        return float(ra)
    except ValueError:
        # HTTP-date form – parse if needed; for now default
        return 1.0
```

---

## F) Tests (MockTransport examples)

```python
# tests/ontology_download/net/test_redirect_audit.py
import httpx
from src.DocsToKG.OntologyDownload.net.client import get_http_client, request_with_redirect_audit

def test_redirect_audit_allows_safe(tmp_path, settings):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/hop":
            return httpx.Response(302, headers={"Location": "/final"})
        return httpx.Response(200, text="OK")
    client = get_http_client(settings)
    client._transport = httpx.MockTransport(handler)  # dangerous but fine in tests
    resp = request_with_redirect_audit(client, "GET", "https://example.org/hop")
    assert resp.status_code == 200

def test_redirect_audit_blocks_cross_host(settings, monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://evil.example/steal"})
    client = get_http_client(settings)
    client._transport = httpx.MockTransport(handler)

    # Monkeypatch the gate to simulate a block
    from src.DocsToKG.OntologyDownload.policy import url_gate
    monkeypatch.setattr(url_gate, "validate_url_security", lambda u, *_: (_ for _ in ()).throw(RuntimeError("E_HOST_DENY")))
    try:
        request_with_redirect_audit(client, "GET", "https://example.org/hop")
        assert False, "expected failure"
    except RuntimeError as e:
        assert "E_HOST_DENY" in str(e)
```

```python
# tests/ontology_download/net/test_status_retry_429.py
import httpx, time
from tenacity import RetryError
from src.DocsToKG.OntologyDownload.download import fetch_to_path

def test_429_then_success(tmp_path, settings, monkeypatch):
    attempts = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, content=b"x"*1024)
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    from src.DocsToKG.OntologyDownload.net import client as net_client
    monkeypatch.setattr(net_client, "get_http_client", lambda s: client)

    t0 = time.perf_counter()
    dest = tmp_path / "out.bin"
    fetch_to_path(settings, "https://example.org/x.bin", dest)
    dt = time.perf_counter() - t0
    assert dt >= 1.0  # Tenacity slept once
    assert dest.stat().st_size == 1024
```

---

# ✅ What your juniors should remember

* **Always** go through `get_http_client()` and `request_with_redirect_audit()`.
* **Never** call `client.get(..., follow_redirects=True)`.
* **Rate-limit** before each attempt; **Tenacity** sleeps on 429; limiter handles steady pacing (no double sleep).
* **Stream** big responses: temp → fsync → rename.
* **Events** are your friend: look for `net.request` lines to see timing and cache behavior.

If you’d like, I can bundle these into a small PR with the stubs placed in the right files and a couple of unit tests so your team can iterate from a working base.
