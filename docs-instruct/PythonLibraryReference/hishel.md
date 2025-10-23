Below is a compact but thorough “engineering‑refactor” guide to **hishel**—an RFC‑9111–compliant HTTP caching library for Python that integrates with **HTTPX** (and **HTTP Core**), with optional (beta) support for **Requests**. It’s written for an AI programming agent replacing ad‑hoc caching with hishel.

---

## 1) What hishel is (and isn’t)

**hishel** (“remember” in Armenian) is a **client‑side HTTP cache** you attach to HTTPX (or HTTP Core) so responses can be reused according to **RFC 9111**—including `Cache-Control`, `ETag`, `Last-Modified`, `Vary`, freshness, and validation. You can use it via **drop‑in clients**, **transport wrappers**, or **HTTP Core pools**, and you can pick a storage backend (filesystem, in‑memory, Redis, SQLite, S3). As of **Oct 18, 2025**, the latest stable on PyPI is **0.1.5** (a 1.0.0.dev0 pre‑release exists). ([PyPI][1])

---

## 2) Install & versions

```bash
pip install hishel              # core
pip install hishel[httpx]       # HTTPX helpers (optional)
pip install hishel[redis]       # Redis storage
pip install hishel[sqlite]      # Async SQLite (beta extras)
pip install hishel[s3]          # S3 storage
pip install hishel[yaml]        # YAML serializer
```

* Requires Python 3.9+. Extras advertised on PyPI include `httpx`, `requests`, `redis`, `sqlite`, `s3`, `yaml`. Current stable docs are versioned at `/0.1.5/`. ([PyPI][1])
* HTTPX explicitly lists **hishel** as the caching option in its “Third‑party packages.” ([HTTPX][2])

---

## 3) Mental model (RFC‑9111 in practice)

* **Freshness**: hishel will serve fresh cached responses without contacting the origin; stale responses require **revalidation** (conditional requests using `ETag`/`Last-Modified`). ([RFC Editor][3])
* **Validation outcomes**: 304 → refresh metadata and use cache; 2xx/5xx → replace or invalidate. hishel exposes an **explicit state machine** API in beta (`IdleClient → FromCache | NeedRevalidation | CacheMiss → …`). ([Hishel][4])
* **Request directives** like `only-if-cached`, `min-fresh`, `max-stale`, `max-age`, `no-cache`, `no-store` are honored; `only-if-cached` can return **504** without contacting the origin. ([Hishel][5])

---

## 4) Integration patterns (choose one)

> **Rule of thumb:** prefer **transports** when you already have a configured HTTPX client (more flexible); use **cache clients** for the fastest drop‑in swap; use **HTTP Core** integration when you’re working below HTTPX. ([Hishel][6])

### A) Drop‑in **HTTPX** cache clients (sync/async)

```python
import hishel

# Sync
with hishel.CacheClient() as client:
    r1 = client.get("https://api.example.com/data")
    r2 = client.get("https://api.example.com/data")  # cached

# Async
import asyncio
async def main():
    async with hishel.AsyncCacheClient() as client:
        await client.get("https://api.example.com/data")
        await client.get("https://api.example.com/data")  # cached
asyncio.run(main())
```

These mirror `httpx.Client`/`AsyncClient` (same kwargs), with extra cache config. ([Hishel][6])

### B) **HTTPX transports** (recommended for flexibility)

```python
import hishel, httpx

cache_transport = hishel.CacheTransport(transport=httpx.HTTPTransport())
with httpx.Client(transport=cache_transport) as client:
    client.get("https://example.com")
    client.get("https://example.com")  # from cache
```

You can chain with your own transport(s) and provide storage independently. See HTTPX “transports” background. ([Hishel][6])

### C) **HTTP Core** connection pools (low‑level)

```python
import hishel, httpcore

with httpcore.ConnectionPool() as pool:
    with hishel.CacheConnectionPool(pool=pool) as cached:
        cached.get("https://example.com")
        cached.get("https://example.com")  # cached
```

Useful when you sit under HTTPX or embed caching into a custom client. ([Hishel][6])

> **Avoid monkeypatching**: `hishel.install_cache` exists but is for experiments only. Prefer transports/clients. ([Hishel][6])

---

## 5) Storage backends (choose per deployment)

*All storages support an optional **TTL** (purge policy for the storage). This **is not** RFC freshness; it’s how long entries are kept at rest. Freshness/validation still follows RFC‑9111 via the controller.* ([Hishel][7])

1. **Filesystem** (default: `~/.cache/hishel`)

```python
storage = hishel.FileStorage(base_path="/var/cache/hishel",
                             ttl=3600,           # purge after 1h (at-rest)
                             check_ttl_every=600 # sweep every 10m
)
client  = hishel.CacheClient(storage=storage)
```

Directory layout uses hashed keys; you can customize **cache keys** (below) to make file names meaningful. ([Hishel][7])

2. **In‑memory** (fastest, volatile; LFU eviction)

```python
storage = hishel.InMemoryStorage(capacity=64)  # LFU when capacity exceeded
client  = hishel.CacheClient(storage=storage)
```

Good for ephemeral caching or tests; configure capacity to cap RAM use. ([Hishel][7])

3. **Redis**

```python
import redis, hishel
storage = hishel.RedisStorage(  # pip install hishel[redis]
    client=redis.Redis(host="redis.internal", port=6379),
    ttl=3600
)
client = hishel.CacheClient(storage=storage)
```

Use when multiple processes/hosts must share a cache. ([Hishel][7])

4. **SQLite**

```python
import sqlite3, hishel
storage = hishel.SQLiteStorage(               # pip install hishel[sqlite] for async beta
    connection=sqlite3.connect("cache.db", timeout=5),
    ttl=3600
)
client = hishel.CacheClient(storage=storage)
```

Lightweight persistent cache file; use when you need speed and single‑file deploys. ([Hishel][7])

5. **AWS S3**

```python
import boto3, hishel
s3 = boto3.client("s3")
storage = hishel.S3Storage(bucket_name="my-cache-bucket",
                           client=s3, ttl=3600, check_ttl_every=600)
client  = hishel.CacheClient(storage=storage)
```

For large/centralized caches across instances. ([Hishel][7])

> **Lifecycle & closing**: since **0.1.5**, storages expose `close()` in their API (useful for app shutdown). ([PyPI][1])

---

## 6) Serialization

Storages serialize cached pairs (request, response, metadata). Pick **JSON** (default), **YAML**, or **pickle** (be careful with untrusted data). Example structure exposes response, request, and `metadata` (`cache_key`, `number_of_uses`, `created_at`). ([Hishel][8])

```python
serializer = hishel.JSONSerializer()  # or YAMLSerializer / PickleSerializer
storage    = hishel.FileStorage(serializer=serializer)
```

---

## 7) Cache behavior controls (Controller)

`hishel.Controller` configures how strictly RFC‑9111 is applied:

```python
import hishel
controller = hishel.Controller(
    force_cache=False,                         # force caching even if headers forbid
    cacheable_methods=["GET"],                 # add "POST" if you purposely cache POST
    cacheable_status_codes=[200, 301, 308],
    allow_heuristics=False,                    # enable to infer freshness (see below)
    cache_private=True,                        # set False for shared caches
    allow_stale=False,                         # allow serving stale on connectivity errors
    always_revalidate=False,                   # always send conditional validation
    key_generator=None                         # customize cache key (see below)
)
client = hishel.CacheClient(controller=controller)
```

Notes & extensions:

* **Heuristics**: enable `allow_heuristics=True`. By default hishel is conservative; you can opt‑in to the broader RFC “heuristically cacheable” statuses using the library constant `HEURISTICALLY_CACHEABLE_STATUS_CODES`. ([Hishel][9])
* **Shared cache**: set `cache_private=False` to avoid caching `Cache-Control: private` responses. ([Hishel][9])
* **Stale on network errors**: set `allow_stale=True`. ([Hishel][9])
* **Always revalidate**: set `always_revalidate=True` (useful with rate‑limited APIs like GitHub to keep content fresh but cheap). ([Hishel][9])
* **Custom cache keys**: include method/host/body, etc., using a function `(httpcore.Request, body_bytes) -> str`. Good for POST/GraphQL. ([Hishel][9])

```python
import httpcore
from hishel._utils import generate_key

def key_with_method_host(request: httpcore.Request, body: bytes) -> str:
    base = generate_key(request, body)
    return f"{request.method.decode()}|{request.url.host.decode()}|{base}"

controller = hishel.Controller(key_generator=key_with_method_host)
```

---

## 8) Request/response knobs you can flip per call

### A) RFC request headers (work everywhere)

* `Cache-Control: only-if-cached` → use cache or **504**; do **not** hit the network.
* `Cache-Control: max-age=SECONDS`, `max-stale=SECONDS`, `min-fresh=SECONDS`, `no-cache`, `no-store`.
  Example:

```python
client.get("https://example.com", headers=[
    ("Cache-Control", "only-if-cached")
])
```

([Hishel][5])

### B) HTTPX **extensions** (hishel’s syntactic sugar)

* **Request**:

  * `extensions={"force_cache": True}` → cache even when headers would forbid.
  * `extensions={"cache_disabled": True}` → ignore cache and don’t store (equivalent to adding the right RFC headers).
* **Response extensions**:

  * `response.extensions["from_cache"]` → `True` if served from cache.
  * `response.extensions["revalidated"]` → `True` if a conditional request just happened.
  * If `from_cache` is `True`, `response.extensions["cache_metadata"]` includes `cache_key`, `created_at`, `number_of_uses`. ([Hishel][10])

### C) Beta “metadata” API (typed; HTTPX & Requests)

Prefer HTTPX `extensions` for type safety; same flags can be sent as **headers** for Requests:

* **Request metadata**:
  `hishel_ttl`, `hishel_refresh_ttl_on_access`, `hishel_spec_ignore` (ignore RFC rules—use with caution), `hishel_body_key` (include body in key for POST/GraphQL).
  Header equivalents: `X-Hishel-Ttl`, `X-Hishel-Refresh-Ttl-On-Access`, `X-Hishel-Spec-Ignore`, `X-Hishel-Body-Key`.
* **Response metadata** (beta naming): `hishel_from_cache`, `hishel_revalidated`, `hishel_stored`, etc. ([Hishel][11])

---

## 9) Observability & debugging

* **Logging:** enable `hishel.controller` at `DEBUG` to see decisions like “considered cacheable”, “needs revalidation”, “using stale,” etc. ([Hishel][12])

```python
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("hishel.controller").setLevel(logging.DEBUG)
```

* **Programmatic status:** look at `response.extensions["from_cache"]`, `["revalidated"]`, and (when cached) `["cache_metadata"]`. ([Hishel][10])

---

## 10) Requests compatibility (beta)

If your refactor must support **Requests**, mount the **CacheAdapter**:

```python
import requests
from hishel.beta.requests import CacheAdapter   # beta path

session = requests.Session()
session.mount("https://", CacheAdapter())
session.mount("http://",  CacheAdapter())

r = session.get("https://api.example.com/data")
# Next call will be cached; you can inspect headers like:
bool_from_cache = (r.headers.get("X-Hishel-From-Cache") == "True")
```

You can configure storage and RFC options via adapter kwargs; per‑request metadata uses `X-Hishel-*` headers. (Docs list full adapter options & header names.) ([Hishel][13])

> **Naming note:** The GitHub README shows non‑beta import paths for Requests; the versioned docs for **0.1.5** place Requests integration under **`hishel.beta.requests`**. Check your installed version before choosing import paths. ([GitHub][14])

---

## 11) Recipes for common refactors

### A) Replace custom “in‑memory GET cache”

* Before: dict keyed by URL → value bytes + ad‑hoc TTL.
* After: `InMemoryStorage(capacity=N)` + `CacheClient()`. You get RFC behavior, validation, `Vary`, and LFU eviction. ([Hishel][7])

```python
client = hishel.CacheClient(storage=hishel.InMemoryStorage(capacity=1024))
```

### B) Cache **POST**/GraphQL safely

* Enable POST in controller and **include body in key**.
* Option 1: controller `key_generator`.
* Option 2 (beta): pass `extensions={"hishel_body_key": True}` (HTTPX). ([Hishel][9])

### C) Bypass cache temporarily (debug or hot‑path freshness)

* Add `extensions={"cache_disabled": True}` (HTTPX) or `Cache-Control: no-store, max-age=0`. ([Hishel][6])

### D) Rate‑limited APIs (e.g., GitHub)

* Use standard caching; optionally `always_revalidate=True` to stay fresh with cheap 304s. ([Hishel][15])

### E) Multi‑process / multi‑host cache

* Use **Redis** or **S3** storages, not in‑memory or plain files. ([Hishel][7])

---

## 12) Gotchas & guardrails

* **Storage TTL vs. RFC freshness**: storage TTL is a purge policy; it does **not** decide whether a response is “fresh.” That’s the controller (RFC rules). ([Hishel][7])
* **Sensitive/private responses**: set `cache_private=False` for shared caches. Verify behavior with `Authorization` or user‑specific data. ([Hishel][9])
* **`only-if-cached` can 504**: this is expected per spec—hishel won’t touch the network. ([Hishel][5])
* **Monkeypatching**: `install_cache` exists but isn’t recommended beyond experiments. Prefer transports or clients. ([Hishel][6])
* **Closing**: if you manage storages explicitly, call `close()` on shutdown (added in 0.1.5). ([PyPI][1])

---

## 13) Minimal “drop‑in” mappings for an AI refactor

| Custom approach you may see                           | hishel replacement                                                              |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- |
| Manual dict cache keyed by URL                        | `CacheClient` + default FileStorage/InMemoryStorage                             |
| Homegrown `If-None-Match` / `If-Modified-Since` logic | Let hishel validate; inspect `response.extensions["revalidated"]`               |
| “Cache POST by hashing body”                          | Controller `key_generator` or beta `hishel_body_key`                            |
| “Serve stale on network error”                        | `Controller(allow_stale=True)`                                                  |
| “Force cache even with no-store”                      | per‑request `extensions={"force_cache": True}` or controller `force_cache=True` |
| “Do not hit network if not cached”                    | `headers=[("Cache-Control", "only-if-cached")]`                                 |

(See the sections above for code.) ([Hishel][9])

---

## 14) HTTPX specifics you may rely on

* **Transports** are where HTTPX makes requests; hishel provides `CacheTransport`/`AsyncCacheTransport`, which you can stack with your own transports. ([Hishel][6])
* **Clients parity**: hishel cache clients accept the same HTTPX client parameters (proxies, auth, HTTP/2, limits, etc.). ([Hishel][6])

---

## 15) Beta: typed integrations & sans‑I/O state machine

* **HTTPX beta API**: strongly‑typed `SyncCacheClient` / `AsyncCacheClient` and `SyncCacheTransport` / `AsyncCacheTransport` in `hishel.beta.httpx`, with typed **metadata** on `extensions` (e.g., `hishel_ttl`) and matching `X-Hishel-*` headers. ([Hishel][16])
* **Requests beta API**: `hishel.beta.requests.CacheAdapter`, response headers expose cache info (e.g., `X-Hishel-From-Cache`, `X-Hishel-Revalidated`). ([Hishel][13])
* **Sans‑I/O state machine**: build your own cache layer around any stack while remaining RFC‑compliant—start at `create_idle_state("client")` and drive transitions. Useful for advanced frameworks or server‑side caching (work in progress). ([Hishel][4])

---

## 16) Example: chaining a custom logging transport + cache (HTTPX)

```python
import httpx, hishel

class LoggingTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport):
        self.transport = transport
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        print("→", request.method, request.url)
        resp = self.transport.handle_request(request)
        print("←", resp.status_code)
        return resp

transport = LoggingTransport(
    hishel.CacheTransport(transport=httpx.HTTPTransport())
)
with httpx.Client(transport=transport) as client:
    client.get("https://example.com")
    client.get("https://example.com")  # logged; comes from cache
```

This mirrors the chaining pattern documented for hishel + HTTPX transports. ([Hishel][6])

---

## 17) Performance notes

* The project’s QuickStart shows large speedups on repeated calls because cached reads avoid network I/O. Realistic gains depend on origin latency, object size, and revalidation frequency. (See docs/examples; HTTP/2 is supported through HTTPX.) ([Hishel][17])

---

## 18) Where to look up specifics quickly

* **Intro / QuickStart / Clients / Transports / HTTP Core / Headers / Extensions / Logging**: hishel docs (v0.1.5). ([Hishel][17])
* **Storages** (File, In‑Memory, Redis, SQLite, S3), TTL sweeps, LFU: hishel storages page. ([Hishel][7])
* **Serializers** & metadata shape: hishel serializers page. ([Hishel][8])
* **Controller options** (heuristics, revalidation, shared/private, custom keys): hishel controllers page. ([Hishel][9])
* **HTTPX transport background**: HTTPX documentation. ([HTTPX][18])
* **RFC 9111** canonical spec. ([RFC Editor][3])
* **Requests & HTTPX typed (beta)** + metadata keys: hishel beta integration & metadata. ([Hishel][13])
* **Versioning / extras / current release**: PyPI. ([PyPI][1])

---

### TL;DR for an automated refactor

1. **Identify** any custom cache layers (URL→bytes dicts, hand‑rolled ETag logic, “don’t hit origin if cached”, “cache POST by hashing body”).
2. **Swap** either to `hishel.CacheClient`/`AsyncCacheClient` or inject `CacheTransport` into existing HTTPX clients.
3. **Pick storage**: InMemory for ephemeral tests; File/SQLite for single‑node persistence; Redis/S3 for shared caches.
4. **Encode policy** in `Controller` (e.g., allow POST, always revalidate, forbid private, allow stale).
5. **Use per‑request controls**: RFC headers or HTTPX `extensions` for one‑offs; beta `hishel_*` metadata is available if you also need Requests.
6. **Instrument** via logger + response extensions; watch `from_cache`/`revalidated` and `cache_metadata.number_of_uses`.

That’s the complete path to replace bespoke caching with a maintainable, RFC‑correct, and portable layer using **hishel**.

[1]: https://pypi.org/project/hishel/ "hishel · PyPI"
[2]: https://www.python-httpx.org/third_party_packages/?utm_source=chatgpt.com "Third Party Packages"
[3]: https://www.rfc-editor.org/rfc/rfc9111.html?utm_source=chatgpt.com "RFC 9111: HTTP Caching"
[4]: https://hishel.com/0.1.5/beta/specification/ "Specification - Hishel"
[5]: https://hishel.com/0.1.5/advanced/http_headers/ "HTTP Headers - Hishel"
[6]: https://hishel.com/0.1.5/userguide/ "User Guide - Hishel"
[7]: https://hishel.com/0.1.5/advanced/storages/ "Storages - Hishel"
[8]: https://hishel.com/0.1.5/advanced/serializers/ "Serializers - Hishel"
[9]: https://hishel.com/0.1.5/advanced/controllers/ "Controllers - Hishel"
[10]: https://hishel.com/0.1.5/advanced/extensions/ "Extensions - Hishel"
[11]: https://hishel.com/0.1.5/beta/metadata/ "Metadata - Hishel"
[12]: https://hishel.com/0.1.5/advanced/logging/ "Logging - Hishel"
[13]: https://hishel.com/0.1.5/beta/integrations/requests/ "Requests - Hishel"
[14]: https://github.com/karpetrosyan/hishel "GitHub - karpetrosyan/hishel: Elegant HTTP Caching for Python"
[15]: https://hishel.com/0.1.5/examples/github/ "GitHub - Hishel"
[16]: https://hishel.com/0.1.5/beta/integrations/httpx/ "Httpx - Hishel"
[17]: https://hishel.com/0.1.5/ "Hishel"
[18]: https://www.python-httpx.org/advanced/transports/?utm_source=chatgpt.com "Transports"
