Below is a practical, end‑to‑end reference to **HTTPX** (sync & async HTTP client for Python) tailored for refactoring an AI agent that currently maintains custom HTTP logic (timeouts, connection pooling, retries, streaming, proxies, etc.). Examples are written to be safe in production code and easy to slot into an agent’s networking layer.

> **What is HTTPX?**
> A fully‑featured HTTP client for Python that offers both synchronous and asynchronous APIs, HTTP/1.1 by default plus optional HTTP/2, strong timeout semantics, connection pooling, cookies, auth, proxy support, and pluggable transports (including in‑process WSGI/ASGI) — with an API broadly familiar to `requests`. ([PyPI][1])

---

## 1) Versions, install, and supported Python

* **Latest stable (as of Oct 20, 2025):** `0.28.1` on PyPI. (There are 1.0 pre‑releases in 2025, but the latest stable is 0.28.1.) ([PyPI][1])
* **Python:** stable 0.28.1 lists Python **3.8+** on PyPI; some docs pages mention 3.9+, which aligns with forward‑looking docs — check your runtime and pin versions if needed. ([PyPI][1])
* **Install:**

  ```bash
  pip install httpx             # core (HTTP/1.1)
  pip install httpx[http2]      # add HTTP/2 via 'h2'
  pip install httpx[socks]      # SOCKS proxy support
  pip install httpx[zstd]       # zstandard decoding (0.27.1+)
  ```

  ([PyPI][1])

**Important deprecations in 0.28.0+:**

* The **string path** form of `verify="path/to/certs.pem"` is deprecated; prefer an `ssl.SSLContext` (details below).
* The **`cert=` parameter** is deprecated; use `SSLContext.load_cert_chain`.
* Older `proxies=` shortcut and the `app=` shortcut were removed/deprecated in favor of explicit transports and mounts. ([GitHub][2])

---

## 2) Core mental model

* **Top‑level helpers** (`httpx.get/post/...`) are great for quick calls, but **do not reuse connections**. Use a **Client** (sync) or **AsyncClient** (async) for agents or any nontrivial workload to get pooling, cookie persistence, and shared config. ([HTTPX][3])
* Clients are **safe to share** across threads/tasks and should be long‑lived in your app/agent. ([HTTPX][4])

---

## 3) The Client lifecycle (sync & async)

**Synchronous**

```python
import httpx

with httpx.Client(
    base_url="https://api.example.com",
    timeout=httpx.Timeout(5.0),          # see §4
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    follow_redirects=False,
) as client:
    r = client.get("/v1/resource")
    r.raise_for_status()
    data = r.json()
```

**Asynchronous**

```python
import httpx
import asyncio

async def fetch():
    async with httpx.AsyncClient(base_url="https://api.example.com") as client:
        r = await client.get("/v1/resource")
        r.raise_for_status()
        return r.json()

asyncio.run(fetch())
```

Why the `with`/`async with`? It ensures pools are closed cleanly and connections returned. Clients also support explicit `.close()` / `await .aclose()`. ([HTTPX][3])

**`base_url`** lets you call with relative paths — great for agents targeting a specific service. ([HTTPX][3])

---

## 4) Timeouts (be explicit)

HTTPX has **strict 5‑second timeouts by default** — unlike `requests`, which defaults to no timeout. That default is split into **connect/read/write/pool** phases and is configurable. For agents, set timeouts intentionally per your SLAs. ([HTTPX][5])

```python
timeout = httpx.Timeout(  # seconds
    connect=5.0,
    read=10.0,
    write=10.0,
    pool=5.0,
)
client = httpx.Client(timeout=timeout)
```

Timeout exceptions: `ConnectTimeout`, `ReadTimeout`, `WriteTimeout`, `PoolTimeout` (all derive from `TimeoutException`). ([HTTPX][6])

---

## 5) Redirects (opt‑in)

HTTPX **does not follow redirects by default** (you must set `follow_redirects=True` per request or on the client). `response.history` captures any followed redirects; `response.next_request` gives you the next step if you’re handling manually. ([HTTPX][7])

---

## 6) Connection pooling & limits

Pool sizing (on Clients) is controlled with `httpx.Limits`:

* `max_connections`: default **100**
* `max_keepalive_connections`: default **20**
* `keepalive_expiry`: default **5.0** seconds

Tune these if your agent fans out to many hosts or needs more concurrency. ([HTTPX][8])

```python
limits = httpx.Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=10)
client = httpx.Client(limits=limits)
```

---

## 7) HTTP/2 (opt‑in) and HTTP/3 (status)

Enable **HTTP/2** by installing extras and passing `http2=True` on the client. You can check which version was used with `response.http_version`. HTTP/2 is particularly useful for highly concurrent async agents due to multiplexing. ([HTTPX][9])

> HTTP/3 is not advertised as supported by HTTPX; the docs reference HTTP/1.1 & HTTP/2. If you require HTTP/3, you’ll currently need alternative stacks (e.g., `curl_cffi`, or bespoke clients). For most agent workloads, HTTP/2 gives ample concurrency benefits. ([HTTPX][10])

---

## 8) Proxies (HTTP & SOCKS), mounts, and routing

* Simple: pass a single proxy URL via `proxy=...` on the client/top‑level calls to route all traffic.
* Advanced: use **mounted transports** to route per scheme/host/port (e.g., different proxies for HTTP vs HTTPS).
* **SOCKS** is supported as an optional extra (`httpx[socks]`).
* **Environment variables** `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/NO_PROXY` are honored unless `trust_env=False`. ([HTTPX][11])

```python
# One proxy for everything
client = httpx.Client(proxy="http://localhost:8030")

# Different proxies per scheme using mounts (recommended in 0.28+)
proxy_mounts = {
    "http://":  httpx.HTTPTransport(proxy="http://localhost:8030"),
    "https://": httpx.HTTPTransport(proxy="http://localhost:8031"),
}
client = httpx.Client(mounts=proxy_mounts)
```

> **Note (0.28.0):** the old `proxies=` shortcut was removed; use `proxy=` or `mounts=` + transports. ([GitHub][2])

---

## 9) SSL/TLS (verification & client certs)

* Verification is **on by default**, using **certifi**; invalid certs raise an error. Set `verify=False` to disable (avoid in production). ([HTTPX][12])
* Prefer an **`ssl.SSLContext`** to customize verification or client certs:

  ```python
  import ssl, certifi, httpx

  ctx = ssl.create_default_context(cafile=certifi.where())
  # Optional client certs:
  # ctx.load_cert_chain(certfile="client.pem", keyfile="client.key", password=None)

  client = httpx.Client(verify=ctx)
  ```

  This is also the migration path for 0.28 deprecations (string `verify=...` & `cert=`). ([HTTPX][12])

---

## 10) Authentication

Built‑ins: **Basic**, **Digest**, and **NetRC**; you can also implement custom multi‑request auth flows by subclassing `httpx.Auth`. For token flows that must read request/response bodies, set `requires_request_body` / `requires_response_body`. ([HTTPX][13])

```python
auth = httpx.BasicAuth("user", "pass")         # or httpx.DigestAuth, httpx.NetRCAuth
client = httpx.Client(auth=auth)
```

---

## 11) Cookies

Use the client to persist cookies across requests. You can pass a dict or an `httpx.Cookies` jar; you can set domain/path‑scoped cookies programmatically. ([HTTPX][7])

```python
cookies = httpx.Cookies()
cookies.set("token", "abc123", domain="api.example.com")
client  = httpx.Client(cookies=cookies)
```

---

## 12) Streaming (downloads & uploads) and memory safety

HTTPX encourages explicit streaming with a context manager instead of `stream=True`. Use `.iter_bytes()/.iter_text()/.iter_lines()` for sync and `.aiter_*()` for async. You can also **conditionally** call `response.read()` inside the stream if the body is small. ([HTTPX][7])

```python
# Streaming download to file
with httpx.stream("GET", url) as r:
    r.raise_for_status()
    with open("file.bin", "wb") as f:
        for chunk in r.iter_bytes():
            f.write(chunk)
```

Uploads are streaming by default when you pass file objects or a generator (use async generators with `AsyncClient`). ([HTTPX][3])

---

## 13) Events, logging, and observability

* **Event hooks** run for every request/response and can be used for logging/metrics or enforcing policies (e.g., always call `raise_for_status()`):

  ```python
  def log_request(req):  print(f">>> {req.method} {req.url}")
  def log_response(resp): print(f"<<< {resp.status_code} {resp.request.url}")

  client = httpx.Client(event_hooks={"request": [log_request], "response": [log_response]})
  ```

  Event hooks are invoked **after a request is prepared (before send)** and **after a response is received (before return)**, and may modify the objects if you need to. For async clients, hooks must be `async def`. ([HTTPX][14])
* You can also enable **debug logging** for `httpx` and `httpcore` with Python’s `logging` to trace low‑level network operations. ([HTTPX][15])

---

## 14) Retrying: what’s built‑in and what isn’t

* HTTPX **now exposes basic retrying at the transport level**: pass `retries=<int>` when you build a transport (sync or async) and inject it into a Client. **This only retries connection errors & connect timeouts** (e.g., `ConnectError`, `ConnectTimeout`). Use it for smoothing transient network hiccups. ([HTTPX][16])

  ```python
  transport = httpx.HTTPTransport(retries=2)            # or AsyncHTTPTransport for async
  client = httpx.Client(transport=transport)
  ```

* For **richer retry policies** (backoffs, retry‑after, status‑code–based retries, idempotency rules), layer a general-purpose library such as **tenacity** or use a third‑party **httpx‑retries** plugin. (The HTTPX docs explicitly suggest tenacity for more complex retry behavior.) ([HTTPX][16])

---

## 15) Transports & mounts: customization superpowers

* **Transports** let you control the low‑level I/O implementation:

  * `HTTPTransport` / `AsyncHTTPTransport` (real network I/O) — supports `retries`, `local_address`, `uds` (Unix sockets), and proxy configuration.
  * `WSGITransport` / `ASGITransport` call **in‑process web apps** directly — ideal for integration tests or simulating dependencies without real network calls.
  * `MockTransport` returns canned responses for deterministic tests. ([HTTPX][16])
* **Mounts** let you route by scheme, host, or port (e.g., special transport for `*.internal`, disabling HTTP/2 for a problematic domain, or custom proxy round‑robin). ([HTTPX][16])

---

## 16) Exceptions (you should catch these in an agent)

Hierarchy highlights:

* `HTTPError` (base)

  * `RequestError` → network/transport issues (`ConnectError`, `ReadError`, `WriteError`, `TimeoutException` and its variants, `ProxyError`, etc.)
  * `HTTPStatusError` → raised only when you call `response.raise_for_status()` and the status is 4xx/5xx
* Others: `InvalidURL`, `CookieConflict`, `TooManyRedirects`, `StreamError` family… ([HTTPX][6])

Example:

```python
try:
    r = client.get(url, timeout=10)
    r.raise_for_status()
except httpx.TimeoutException:
    ...
except httpx.RequestError as exc:
    # DNS/connection/TLS/protocol/etc.
    ...
except httpx.HTTPStatusError as exc:
    # 4xx/5xx after .raise_for_status()
    ...
```

---

## 17) Encoding, compression, and content helpers

* `response.text` uses charset detection when not explicitly declared; you can override with `response.encoding`.
* HTTPX automatically decodes `gzip`/`deflate`, and supports **brotli** and **zstd** if extras are installed. ([HTTPX][7])

---

## 18) Requests compatibility: what’s different

HTTPX keeps a similar feel to `requests` with a few deliberate differences that your refactor should account for:

* **Redirects** are **off** by default (enable per request or client).
* **Timeouts** default to strict values (set to `None` for requests‑like behavior).
* **Streaming** uses `with client.stream(...):`, not `stream=True`.
* **Proxy mapping** uses **mounts** (`"http://", "https://", "all://"`), not `{"http": ..., "https": ...}`.
* **URL** fields are `httpx.URL` objects; cast with `str()` if you need plain strings.
* **Some request methods** (`GET/HEAD/DELETE/OPTIONS`) do not accept body args (`content/data/json/files`); use `.request(...)` if you must send a body with these methods.
* Prefer `response.is_success` over `is_ok`, and use `.next_request` name for manual redirect chains. ([HTTPX][17])

---

## 19) Environment variables (ops‑friendly)

HTTPX reads common env vars unless `trust_env=False`: `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, and `SSL_CERT_FILE`/`SSL_CERT_DIR` for trust stores. This is convenient for containerized agents — but remember to **disable** it when you need deterministic behavior. ([HTTPX][18])

---

## 20) Testing, mocking & local apps

* **MockTransport** for deterministic unit tests without network I/O.
* **WSGITransport/ASGITransport** to call Flask/FastAPI/etc. **in‑process** — perfect for contract tests in an agent’s CI. ([HTTPX][16])

---

## 21) Caching (third‑party)

HTTPX itself doesn’t include HTTP caching. Use **Hishel** or `httpx‑caching` if your agent needs RFC‑compliant caching. ([HTTPX][17])

---

## 22) Recipes you can paste into an agent

### A) Robust “default client” for an agent (sync)

```python
import ssl, certifi, httpx

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

DEFAULT_TIMEOUT = httpx.Timeout(connect=3, read=15, write=15, pool=3)
DEFAULT_LIMITS  = httpx.Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=10)

TRANSPORT = httpx.HTTPTransport(
    retries=2,             # only connect errors/timeouts
    # local_address="0.0.0.0",          # if you must bind
    # uds="/var/run/docker.sock",       # Unix Domain Socket
)

CLIENT = httpx.Client(
    verify=SSL_CTX,
    timeout=DEFAULT_TIMEOUT,
    limits=DEFAULT_LIMITS,
    transport=TRANSPORT,
    follow_redirects=False,
    base_url="https://api.example.com",
    # mounts={"http://": httpx.HTTPTransport(proxy="http://proxy:8080")},
)
```

This pattern centralizes SSL, timeouts, limits, and basic connection retries at the transport layer. For richer retry behavior (e.g., `Retry‑After`, status‑based backoff), compose with tenacity or `httpx-retries`. ([HTTPX][16])

### B) Async batch fan‑out with HTTP/2

```python
import asyncio, httpx

async def fetch(client, url):
    r = await client.get(url)
    r.raise_for_status()
    return r.json()

async def batch(urls):
    async with httpx.AsyncClient(http2=True, timeout=10) as client:
        return await asyncio.gather(*(fetch(client, u) for u in urls))

# asyncio.run(batch([...]))
```

HTTP/2 multiplexing helps on high‑concurrency workloads typical for agents. ([HTTPX][9])

### C) Stream to disk with progress (safe memory)

```python
import httpx, tempfile
from tqdm import tqdm

url = "https://example.com/large.bin"
with tempfile.NamedTemporaryFile() as f:
    with httpx.stream("GET", url) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        with tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_bytes():
                f.write(chunk)
                bar.update(r.num_bytes_downloaded - bar.n)
```

([HTTPX][3])

### D) Event hooks to enforce policies globally

```python
def add_agent_headers(request: httpx.Request) -> None:
    request.headers.setdefault("User-Agent", "my-agent/1.0")

def fail_on_4xx_5xx(response: httpx.Response) -> None:
    response.raise_for_status()

client = httpx.Client(event_hooks={"request": [add_agent_headers],
                                   "response": [fail_on_4xx_5xx]})
```

([HTTPX][14])

### E) Call a FastAPI app in‑process (no network)

```python
from myapp import app  # FastAPI/Starlette app
import httpx, anyio

async def test_health():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.get("/health")
        assert r.status_code == 200
```

([HTTPX][16])

---

## 23) Mapping your custom code to HTTPX

| If your agent currently…          | Refactor to HTTPX by…                                                                                                                   |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Manages its own connection pool   | Use a single long‑lived `Client`/`AsyncClient` with `Limits`. ([HTTPX][8])                                                              |
| Implements timeouts per phase     | Use `httpx.Timeout(connect=…, read=…, write=…, pool=…)`. ([HTTPX][5])                                                                   |
| Follows redirects implicitly      | Decide explicitly: keep `follow_redirects=False` for clarity, or enable per call/client. ([HTTPX][7])                                   |
| Implements home‑grown retries     | Use `HTTPTransport(retries=N)` for connect retries; for status‑aware/backoff strategies, add tenacity or `httpx-retries`. ([HTTPX][16]) |
| Hand‑rolls proxy routing          | Use `mounts={...}` with `HTTPTransport(proxy=...)` by scheme/host; honor env with `trust_env=True`. ([HTTPX][11])                       |
| Embeds cert bundles in weird ways | Build an `SSLContext` and pass it via `verify=ctx` (recommended in 0.28+). ([HTTPX][12])                                                |
| Reimplements cookie jars          | Use `Cookies` on the client to persist across calls (and across redirects). ([HTTPX][7])                                                |
| Writes ad‑hoc log hooks           | Use event hooks and/or Python `logging` for `httpx` & `httpcore`. ([HTTPX][14])                                                         |
| Tests with fake servers           | Use `MockTransport`, `WSGITransport`, or `ASGITransport` to keep tests hermetic. ([HTTPX][16])                                          |
| Implements on‑disk caching        | Swap to **Hishel**/`httpx-caching` for standardized HTTP caching. ([HTTPX][19])                                                         |

---

## 24) “Gotchas” & best practices for agents

1. **Don’t create clients in a hot loop.** Reuse a single client to benefit from pooling and keepalives. ([HTTPX][20])
2. **Be deliberate about timeouts** (defaults are strict; good!). Align them with external API SLOs; don’t set to `None` unless you understand the risk. ([HTTPX][5])
3. **Redirects**: Enabling them globally can hide unnecessary calls; keep them off unless you need them. ([HTTPX][17])
4. **SSL settings** belong on the client at construction time; if you need different SSL policies, use multiple clients. ([HTTPX][17])
5. **Proxies**: prefer mounts for complex topologies; `NO_PROXY`/wildcards can exclude internal hosts. ([HTTPX][18])
6. **Streaming**: when downloading/uploading large payloads, use the streaming APIs to avoid memory spikes and always consume/close responses. ([HTTPX][7])
7. **Retries**: transport `retries` only handles connect issues; layer broader retry policies where appropriate. ([HTTPX][16])
8. **HTTP/2**: opt‑in and measure; it helps when you have many concurrent requests to the same host. ([HTTPX][9])

---

## 25) Minimal, safe scaffolds you can copy

**Sync façade for your agent**

```python
class HttpClient:
    def __init__(self, base_url=None):
        self._client = httpx.Client(
            base_url=base_url or "",
            timeout=httpx.Timeout(3, read=15, write=15, pool=3),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            transport=httpx.HTTPTransport(retries=2),
            follow_redirects=False,
            trust_env=True,
        )

    def request(self, method, url, **kw):
        r = self._client.request(method, url, **kw)
        r.raise_for_status()
        return r

    def close(self):
        self._client.close()
```

**Async façade**

```python
class AsyncHttpClient:
    def __init__(self, base_url=None):
        self._client = httpx.AsyncClient(
            base_url=base_url or "",
            timeout=httpx.Timeout(3, read=15, write=15, pool=3),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            transport=httpx.AsyncHTTPTransport(retries=2),
            http2=True,   # opt-in
            follow_redirects=False,
            trust_env=True,
        )

    async def request(self, method, url, **kw):
        r = await self._client.request(method, url, **kw)
        r.raise_for_status()
        return r

    async def aclose(self):
        await self._client.aclose()
```

---

### Quick checklist for your refactor

* [ ] Replace ad‑hoc pools with one `Client`/`AsyncClient`; set `Limits`. ([HTTPX][8])
* [ ] Make **timeouts explicit**; codify defaults per service. ([HTTPX][5])
* [ ] Decide **redirects** strategy; default off. ([HTTPX][7])
* [ ] Use **transport `retries`** for connect issues; add tenacity / `httpx-retries` for richer logic. ([HTTPX][16])
* [ ] Move SSL and proxy logic to client creation; prefer `SSLContext` and `mounts`. ([HTTPX][12])
* [ ] Swap internal caching to **Hishel** (if needed). ([HTTPX][19])
* [ ] Add event hooks or logging for observability. ([HTTPX][14])
* [ ] If concurrency is heavy, consider `http2=True` and measure. ([HTTPX][9])

---

## Sources and further reading

* Official docs: quickstart, advanced usage (timeouts, proxies, limits, transports, SSL, auth, event hooks), async support, logging, requests‑compatibility. ([HTTPX][7])
* API reference (function signatures; client is thread‑safe). ([HTTPX][4])
* Exception hierarchy. ([HTTPX][6])
* PyPI page (latest stable, install options). ([PyPI][1])
* Release notes 0.28.x (SSL config deprecations; proxies/app shortcut removals). ([GitHub][2])
* Third‑party ecosystem (Hishel caching, httpx‑retries, pytest‑httpx, RESPX, etc.). ([HTTPX][19])

If you’d like, I can adapt this into a drop‑in module for your agent (sync or async) with your preferred defaults and a thin retry policy.

[1]: https://pypi.org/project/httpx/ "httpx · PyPI"
[2]: https://github.com/encode/httpx/releases "Releases · encode/httpx · GitHub"
[3]: https://www.python-httpx.org/advanced/clients/ "Clients - HTTPX"
[4]: https://www.python-httpx.org/api/ "Developer Interface - HTTPX"
[5]: https://www.python-httpx.org/advanced/timeouts/ "Timeouts - HTTPX"
[6]: https://www.python-httpx.org/exceptions/ "Exceptions - HTTPX"
[7]: https://www.python-httpx.org/quickstart/ "QuickStart - HTTPX"
[8]: https://www.python-httpx.org/advanced/resource-limits/ "Resource Limits - HTTPX"
[9]: https://www.python-httpx.org/http2/ "HTTP/2 Support - HTTPX"
[10]: https://www.python-httpx.org/?utm_source=chatgpt.com "HTTPX"
[11]: https://www.python-httpx.org/advanced/proxies/ "Proxies - HTTPX"
[12]: https://www.python-httpx.org/advanced/ssl/ "SSL - HTTPX"
[13]: https://www.python-httpx.org/advanced/authentication/ "Authentication - HTTPX"
[14]: https://www.python-httpx.org/advanced/event-hooks/ "Event Hooks - HTTPX"
[15]: https://www.python-httpx.org/logging/ "Logging - HTTPX"
[16]: https://www.python-httpx.org/advanced/transports/ "Transports - HTTPX"
[17]: https://www.python-httpx.org/compatibility/ "Requests Compatibility - HTTPX"
[18]: https://www.python-httpx.org/environment_variables/ "Environment Variables - HTTPX"
[19]: https://www.python-httpx.org/third_party_packages/ "Third Party Packages - HTTPX"
[20]: https://www.python-httpx.org/async/ "Async Support - HTTPX"
