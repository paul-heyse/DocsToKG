Below is a practical, end‑to‑end **refactorer’s guide to `aiohttp`** that you can hand to an AI programming agent replacing custom HTTP or web‑server code. It focuses on “what to use, when, and why,” with safe defaults, gotchas, and drop‑in recipes.

---

## 0) What `aiohttp` is (and isn’t)

* **What:** `aiohttp` is an asyncio‑native **HTTP client and server** toolkit. It offers a high‑performance client (`ClientSession`) with connection pooling, streaming, cookies, redirects, proxies, WebSocket client; and a production‑ready web server (`aiohttp.web`) with routing, middlewares, signals, cleanup contexts, static files, and WebSocket server. Current docs show **v3.13.1**. ([AIOHTTP Documentation][1])
* **Notable support:** Python 3.9+; **Python 3.14 and free‑threaded (no GIL) builds** are supported as of 3.13.x. Python 3.8 support was dropped in the 3.11.11 line. ([AIOHTTP Documentation][2])
* **What it is not:** It is **not an ASGI framework** and does not natively implement the ASGI interface (there are community adapters). It **does not provide built‑in automatic request retries** (use middleware or a small helper lib). ([GitHub][3])

---

## 1) The mental model

### Client side

* Create **one long‑lived `ClientSession`** and reuse it. Each session owns a **connection pool** with keep‑alive and cookie storage. Prefer `async with ClientSession():` or call `await session.close()`. ([AIOHTTP Documentation][4])
* Requests are **async context managers**: `async with session.get(url) as resp:` ensures the response is closed/released back to the pool. Stream big bodies instead of reading them entirely into RAM. ([AIOHTTP Documentation][4])
* **Timeouts** are explicit and granular via `ClientTimeout` (`total`, `connect`, `sock_connect`, `sock_read`, `ceil_threshold`). Default total is 300s; default `sock_connect` is 30s to allow IPv6/IPv4 fallback. ([AIOHTTP Documentation][4])
* Configure transport via a **Connector** (`TCPConnector`): connection limits, DNS caching/TTL, custom resolver (via `aiodns`), Unix sockets, named pipes (Windows), SSL context, etc. ([AIOHTTP Documentation][5])

### Server side

* Build an app with `web.Application()`, add routes/middlewares, and run with `web.run_app(app)` or programmatically via `AppRunner`/`TCPSite`. Manage long‑lived resources with **`cleanup_ctx`** (startup/teardown as one unit); use **signals** (`on_startup`, `on_shutdown`, `on_cleanup`); and follow **graceful shutdown** steps to close sockets/WebSockets cleanly. ([AIOHTTP Documentation][6])

---

## 2) “Client” API—safe, idiomatic usage

### Minimal hardened client (drop‑in)

```python
import asyncio
from aiohttp import ClientSession, ClientTimeout, TCPConnector

DEFAULT_TIMEOUT = ClientTimeout(total=60)  # tighten from 300s
# Conservative pool limits; tune per workload.
CONNECTOR = TCPConnector(limit=100, limit_per_host=20, ttl_dns_cache=300)

async def fetch_json(url: str, session: ClientSession):
    async with session.get(url, raise_for_status=True) as resp:
        return await resp.json()

async def main(urls):
    async with ClientSession(timeout=DEFAULT_TIMEOUT, connector=CONNECTOR) as session:
        return await asyncio.gather(*(fetch_json(u, session) for u in urls))
```

**Why this shape works:** one session (pool reuse), explicit timeouts, per‑request `raise_for_status`, and streaming by default when you iterate the content. ([AIOHTTP Documentation][4])

### Key features and how to replace custom code

| Need                  | `aiohttp` feature                                       | Notes                                                                                                           |
| --------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Connection pooling    | `ClientSession` + `TCPConnector`                        | `limit`, `limit_per_host`, `use_dns_cache`, `ttl_dns_cache` to control pool & DNS. ([AIOHTTP Documentation][5]) |
| Retry/backoff         | **Client middleware** or small lib (`aiohttp-retry`)    | `aiohttp` itself doesn’t auto‑retry; cookbook shows retry middleware patterns. ([AIOHTTP Documentation][7])     |
| Timeouts              | `ClientTimeout`                                         | Set at session or per request. Default total=300s; sock_connect=30s. ([AIOHTTP Documentation][4])               |
| JSON send/receive     | `json=` request kwarg; `await resp.json()`              | Can plug custom serializer; strict/relaxed content-type handling options. ([AIOHTTP Documentation][4])          |
| Large downloads       | `resp.content.iter_chunked()`                           | Stream to file, don’t buffer whole body. ([AIOHTTP Documentation][4])                                           |
| File/multipart upload | `FormData()` or stream an async generator               | Streams automatically when given file‑like or async generator. ([AIOHTTP Documentation][4])                     |
| Cookies               | `CookieJar` / `DummyCookieJar`                          | Strict by default; can allow IP‑cookies with `unsafe=True`. ([AIOHTTP Documentation][5])                        |
| Redirects             | Enabled by default; inspect `resp.history`              | Turn off with `allow_redirects=False`. ([AIOHTTP Documentation][5])                                             |
| Proxies & env         | Proxy support via params / `trust_env=True`             | See advanced client/proxy docs. ([AIOHTTP Documentation][5])                                                    |
| SSL/TLS               | `ssl=False` to disable verify, or pass `ssl.SSLContext` | Supports fingerprint pinning and `certifi` examples. ([AIOHTTP Documentation][5])                               |
| WebSockets (client)   | `session.ws_connect()`                                  | **Single task** must handle both read and write loop. ([AIOHTTP Documentation][4])                              |
| Observability         | **TraceConfig** lifecycle hooks                         | Trace DNS, connection queueing, redirects, headers sent, chunks, etc. ([AIOHTTP Documentation][8])              |

> **Connector performance knobs you likely want**:
> `TCPConnector(limit=..., limit_per_host=..., ttl_dns_cache=..., resolver=AsyncResolver(...))` and, when needed, a custom `ssl` context. ([AIOHTTP Documentation][5])

---

## 3) “Server” API—patterns that replace home‑grown servers

### Minimal service with lifecycle & health

```python
import asyncio
from aiohttp import web

routes = web.RouteTableDef()

@routes.get("/healthz")
async def health(_):
    return web.json_response({"ok": True})

async def background_task(app: web.Application):
    # Start
    task = asyncio.create_task(do_work())
    # Hand control back to runner
    yield
    # Teardown
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

async def do_work():
    while True:
        await asyncio.sleep(1)

def make_app() -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    app.cleanup_ctx.append(background_task)   # startup + teardown paired
    return app

web.run_app(make_app())  # or use AppRunner/TCPSite programmatically
```

This demonstrates **`cleanup_ctx`** for paired startup/cleanup and a simple route table; in production, add middlewares (logging/auth/errors), signal handlers, and graceful shutdown hooks to close WebSockets/streams. ([AIOHTTP Documentation][6])

### What you get out of the box

* **Routing:** decorators/RouteTableDef, variable paths, class‑based views, reverse URL building. ([AIOHTTP Documentation][9])
* **Middlewares & signals:** request/response middleware chains; `on_startup`, `on_shutdown`, `on_cleanup`. ([AIOHTTP Documentation][9])
* **Lifecycle & deployment:** `run_app()` (blocking) or **`AppRunner`** + `TCPSite` (fully async or multiple sites). **Graceful shutdown** orders stop‑listen, closes idle keep‑alive, runs shutdown handlers, waits for handlers, then cancels stragglers; example shown for closing WebSockets with a code. ([AIOHTTP Documentation][6])
* **Background tasks:** run tasks alongside the server using `cleanup_ctx` (recommended) or `on_startup` with explicit cancellation on cleanup. ([AIOHTTP Documentation][6])

---

## 4) Client middleware (retries, tokens, SSRF guard)

`aiohttp` now has **client middlewares**. Typical patterns include a **bounded retry loop**, **token refresh**, or **SSRF protection**. Cookbook examples include concise snippets and “do-not-shoot-yourself” notes (e.g., avoid infinite recursion by passing `middlewares=()` for internal calls). If your custom code injected headers, retried on certain status codes, or refreshed tokens, port that logic into a small, testable middleware. ([AIOHTTP Documentation][7])

If you prefer a library, **`aiohttp-retry`** wraps the client for configurable retry strategies. ([PyPI][10])

---

## 5) Tracing (hooks for metrics/logging)

Instrument your client with `TraceConfig` to observe **DNS cache hits/misses, connection queueing/creation/reuse, redirects, headers sent, chunks sent/received, request end/exception**. These hooks are ideal for adding metrics and structured logs without monkey‑patching. ([AIOHTTP Documentation][8])

---

## 6) Version notes relevant to refactors (2024–2025)

* **3.11.11 (Dec 2024):** **dropped Python 3.8**; multiple perf fixes; cookie/static handling improvements. ([AIOHTTP Documentation][2])
* **3.12.x (2025):** `DigestAuthMiddleware` improvements; redirect/body handling fixes; **trailer parsing** added to the Python HTTP parser (ties into request‑smuggling hardening). Also note changes like `ssl_shutdown_timeout` deprecations and defaults. ([AIOHTTP Documentation][2])
* **3.13.1 (Oct 2025):** **Python 3.14 supported**; **free‑threaded** CPython builds supported; **Zstandard** compression support. (Good news for long‑term compatibility.) ([AIOHTTP Documentation][2])

> Security footnote: If you compile/run without C extensions (pure‑Python HTTP parser), older releases had request‑smuggling weaknesses. Keep `aiohttp` updated and prefer wheels with the C `llhttp` parser enabled. ([GitHub][11])

---

## 7) What `aiohttp` intentionally does **not** cover (and what to use instead)

* **ASGI:** `aiohttp.web` is its own server stack and **does not natively speak ASGI**. You can still interop with ASGI apps via adapters (e.g., `aiohttp-asgi` / client‑side ASGI connector), but if ASGI is a hard requirement consider an ASGI framework (FastAPI/Starlette/Quart) + ASGI server (Uvicorn/Hypercorn). ([GitHub][3])
* **HTTP/2:** `aiohttp` does **not** advertise HTTP/2 support natively in 3.x; issues discussing it remain open/closed without an implementation. If client HTTP/2 is a must, look at `httpx` (client) and ASGI/HTTP/2 servers. ([GitHub][12])
* **Automatic retries/rate limiting:** roll with client middleware or add `aiohttp-retry`/`tenacity`. ([PyPI][10])

---

## 8) Refactor checklist (map your custom code → aiohttp)

| **If your old code…**                | **Replace with…**                                                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| Opened raw sockets / DIY pooling     | `ClientSession` + `TCPConnector` with `limit`, `limit_per_host`, `ttl_dns_cache` tuned. ([AIOHTTP Documentation][5])               |
| Reimplemented URL joining, base URLs | `ClientSession(base_url=...)` (be mindful of directory semantics introduced/clarified in 3.12). ([AIOHTTP Documentation][13])      |
| Did ad‑hoc JSON serialization        | `json=` for requests; `await resp.json()` for responses; plug `json_serialize` if needed. ([AIOHTTP Documentation][4])             |
| Buffered large downloads             | `async for chunk in resp.content.iter_chunked(n): ...` to stream to disk. ([AIOHTTP Documentation][4])                             |
| Built retry/backoff loops            | Client middleware (cookbook), or `aiohttp-retry`. ([AIOHTTP Documentation][7])                                                     |
| Managed cookies manually             | Use session cookie jar; strict by default; opt‑out with `CookieJar(unsafe=True)` for IPs in tests. ([AIOHTTP Documentation][5])    |
| Hard‑coded DNS/timeout logic         | `ClientTimeout` and `AsyncResolver(nameservers=[...])`. ([AIOHTTP Documentation][5])                                               |
| Custom WebSocket loops               | `ws = await session.ws_connect(...)`; **single task** does both read and write coordination. ([AIOHTTP Documentation][4])          |
| DIY server lifecycle                 | `web.Application`, `cleanup_ctx`, signals, and `AppRunner`/`TCPSite`; follow graceful shutdown steps. ([AIOHTTP Documentation][6]) |

---

## 9) Common pitfalls (and how to avoid them)

* **Creating a `ClientSession` per request** → **Don’t.** Reuse a session; it’s your pool + cookie jar. ([AIOHTTP Documentation][4])
* **Forgetting to close responses** → Always use `async with session.get(...) as resp:`; otherwise the connection can’t be reused. Stream big bodies. ([AIOHTTP Documentation][4])
* **Relying on implicit timeouts** → Set `ClientTimeout` that fits your SLOs; default 300s is often too lax for services. ([AIOHTTP Documentation][4])
* **Infinite retry loops in middleware** → Keep loops bounded (cookbook shows safe patterns). ([AIOHTTP Documentation][7])
* **WebSocket concurrency** → Keep **one reader/writer task** pattern as documented for client websockets. ([AIOHTTP Documentation][4])
* **Server shutdown leaks** → Track open websockets/streams and close them in `on_shutdown`. Use the documented graceful shutdown sequence. ([AIOHTTP Documentation][6])
* **Compiling without C extensions** → You’ll get the slower Python parser and were historically more exposed to parsing leniencies; prefer wheels/C parser and keep versions current. ([GitHub][11])

---

## 10) Recipes you’ll likely need

### a) Streaming download with backpressure

```python
async def download_to(path: str, url: str, session: ClientSession, chunk=1<<15):
    async with session.get(url, raise_for_status=True) as resp:
        with open(path, "wb") as f:
            async for piece in resp.content.iter_chunked(chunk):
                f.write(piece)
```

Streams and releases connection reliably. ([AIOHTTP Documentation][4])

### b) Retry‑with‑jitter middleware (bounded)

```python
import asyncio, random
from typing import Callable
from aiohttp import ClientResponse, ClientRequest

ClientHandlerType = Callable[[ClientRequest], "asyncio.Future[ClientResponse]"]

async def retry_middleware(req: ClientRequest, handler: ClientHandlerType) -> ClientResponse:
    last = None
    for attempt in range(3):
        resp = await handler(req)
        if resp.ok:
            return resp
        last = resp
        await asyncio.sleep(0.2 * (2 ** attempt) + random.random() * 0.1)
    return last
# usage: ClientSession(middlewares=(retry_middleware,))
```

Pattern adapted from the official cookbook’s bounded loop guidance. ([AIOHTTP Documentation][7])

### c) Programmatic server start/stop (AppRunner)

```python
async def serve(app: web.Application, host="0.0.0.0", port=8080):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner  # later: await runner.cleanup()
```

Use this when embedding an `aiohttp` app in a larger async program. ([AIOHTTP Documentation][6])

---

## 11) Testing

Use **`pytest-aiohttp`** to spin up a test server/client with fixtures; run pytest in asyncio mode. This keeps event‑loop handling, cleanup, and lifecycle correct by default. ([AIOHTTP Documentation][14])

---

## 12) Quick comparison (choose the right tool)

* **Need ASGI interop, OpenAPI generation, or HTTP/2 client?** Prefer **ASGI frameworks (FastAPI/Starlette)** and **HTTPX** for the client. If you stay with `aiohttp`, note community adapters exist but ASGI is not native. ([GitHub][3])
* **Need raw control over sockets/pools without reinventing HTTP?** `aiohttp` strikes the balance with `TCPConnector` knobs and trace hooks. ([AIOHTTP Documentation][5])

---

## 13) TL;DR migration plan for an AI agent

1. **Replace custom HTTP client** with a single **`ClientSession`** + **`TCPConnector(limit, limit_per_host, ttl_dns_cache)`** and a **`ClientTimeout(total=... , sock_read=...)`** that matches your SLO. ([AIOHTTP Documentation][5])
2. **Port retries / auth / header injection** to **client middlewares** (or add `aiohttp-retry`). Keep loops bounded; avoid recursion in middleware by using `middlewares=()` for internal calls. ([AIOHTTP Documentation][7])
3. **Stream big bodies** with `iter_chunked`; always use `async with ... as resp` to release connections. ([AIOHTTP Documentation][4])
4. **Instrument** with `TraceConfig` to confirm DNS, pooling, redirects, and timings behave as expected. ([AIOHTTP Documentation][8])
5. **If you host a server**, move resource init/teardown into **`cleanup_ctx`**; add middlewares; support **graceful shutdown** and **WebSocket** closure. Consider `AppRunner` for multi‑site or embedded deployments. ([AIOHTTP Documentation][6])
6. **Stay current** (3.13.x) for Python 3.14/free‑threaded support and parser/security hardening. Use wheels with C `llhttp` parser when possible. ([AIOHTTP Documentation][2])

---

### Sources (selected)

* Official docs home (shows current version): **aiohttp 3.13.1**. ([AIOHTTP Documentation][1])
* Client quickstart (sessions, streaming, JSON, ws, timeouts). ([AIOHTTP Documentation][4])
* Advanced client (connectors, DNS, SSL, middleware overview). ([AIOHTTP Documentation][5])
* Client middleware cookbook (retry, token refresh, SSRF, best practices). ([AIOHTTP Documentation][7])
* Tracing reference (TraceConfig signals). ([AIOHTTP Documentation][8])
* Server docs (index of features). ([AIOHTTP Documentation][9])
* Server advanced (cleanup contexts, background tasks, graceful shutdown, websocket shutdown). ([AIOHTTP Documentation][6])
* Changelog (Py 3.14 + free‑threaded support; Python 3.8 dropped; security‑adjacent parser notes). ([AIOHTTP Documentation][2])
* ASGI compatibility discussion and adapters. ([GitHub][3])
* Third‑party retry helper. ([PyPI][10])

If you’d like, I can tailor a **mechanical refactor plan** to your codebase (e.g., transform rules for your custom client class, or server lifecycle wrappers) and generate tests that assert behavior parity under timeouts, retries, redirects, cookies, and streaming.

[1]: https://docs.aiohttp.org/?utm_source=chatgpt.com "Welcome to AIOHTTP — aiohttp 3.13.1 documentation"
[2]: https://docs.aiohttp.org/en/stable/changes.html?utm_source=chatgpt.com "Changelog — aiohttp 3.13.1 documentation"
[3]: https://github.com/aio-libs/aiohttp/issues/8233?utm_source=chatgpt.com "ASGI support · Issue #8233 · aio-libs/aiohttp"
[4]: https://docs.aiohttp.org/en/stable/client_quickstart.html "Client Quickstart — aiohttp 3.13.1 documentation"
[5]: https://docs.aiohttp.org/en/stable/client_advanced.html "Advanced Client Usage — aiohttp 3.13.1 documentation"
[6]: https://docs.aiohttp.org/en/stable/web_advanced.html "Web Server Advanced — aiohttp 3.13.1 documentation"
[7]: https://docs.aiohttp.org/en/stable/client_middleware_cookbook.html "Client Middleware Cookbook — aiohttp 3.13.1 documentation"
[8]: https://docs.aiohttp.org/en/stable/tracing_reference.html "Tracing Reference — aiohttp 3.13.1 documentation"
[9]: https://docs.aiohttp.org/en/stable/web.html "Server — aiohttp 3.13.1 documentation"
[10]: https://pypi.org/project/aiohttp-retry/?utm_source=chatgpt.com "aiohttp-retry"
[11]: https://github.com/aio-libs/aiohttp/security/advisories/GHSA-8qpw-xqxj-h4r2?utm_source=chatgpt.com "HTTP parser (the python one, not llhttp) still overly lenient ..."
[12]: https://github.com/aio-libs/aiohttp/issues/5631?utm_source=chatgpt.com "HTTP2 support · Issue #5631 · aio-libs/aiohttp"
[13]: https://docs.aiohttp.org/en/stable/client_reference.html "Client Reference — aiohttp 3.13.1 documentation"
[14]: https://docs.aiohttp.org/en/stable/testing.html?utm_source=chatgpt.com "Testing — aiohttp 3.13.0 documentation"
