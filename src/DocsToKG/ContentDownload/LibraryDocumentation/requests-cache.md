Below is a practical, engineer‑oriented reference to **`requests-cache`** geared toward replacing custom HTTP caching logic in an AI agent or service.

---

## What is `requests-cache` (and what problem does it solve)?

`requests-cache` is a persistent HTTP cache that plugs into the `requests` ecosystem. It offers a drop‑in `CachedSession` with TTLs, conditional requests (ETag / Last‑Modified), cache key customization, and multiple storage backends (SQLite, Redis, MongoDB, DynamoDB, filesystem, in‑memory). As of **Oct 20, 2025** the latest stable is **1.2.1** (Python **3.8+**), with pre‑releases for 1.3 available on PyPI. ([PyPI][1])

---

## Install & versions

* **Basic install:**

  ```bash
  pip install requests-cache
  ```

* **With all optional backends/serializers:**

  ```bash
  pip install "requests-cache[all]"
  ```

  (Extras include `redis`, `mongodb`, `dynamodb`, `json`, `yaml`, `bson`, `security`, etc.). ([requests-cache][2])

---

## Core mental model

There are two ways to use it:

1. **Preferred:** create a `CachedSession`, a drop‑in for `requests.Session`.
2. **Alternate:** monkey‑patch `requests` globally with `install_cache()` so `requests.get()`, etc., are transparently cached.

Use sessions for explicit control (especially in multi‑threaded/multiprocess apps). Global patching is convenient but has caveats in bigger or concurrent codebases. ([requests-cache][3])

---

## Quickstart

```python
from datetime import timedelta
from requests_cache import CachedSession

# Local SQLite cache file http_cache.sqlite (default backend is SQLite)
session = CachedSession(
    'http_cache',
    expire_after=timedelta(hours=1),   # default TTL
    cache_control=True,                # respect Cache-Control response headers
    stale_if_error=True,               # serve stale data on errors
)

r1 = session.get('https://httpbin.org/delay/1')  # real request
r2 = session.get('https://httpbin.org/delay/1')  # cache hit; sub-ms
assert r2.from_cache
```

* `CachedSession` behaves like `requests.Session`, with extra cache controls.
* You can also `requests_cache.install_cache(...)` to patch globally. ([requests-cache][3])

---

## Default caching behavior (and how to change it)

* **By default**, only **`GET` and `HEAD`** responses with **status 200** are cached.
  You can expand/limit this via:

  * `allowable_methods=('GET','POST',...)`
  * `allowable_codes=(200, 404, ...)`
  * `filter_fn(response) -> bool` for custom per‑response rules. ([requests-cache][4])

Examples:

```python
session = CachedSession(
    allowable_methods=('GET','POST'),
    allowable_codes=(200, 418),      # cache teapots if you like
)

# Filter out huge responses:
from sys import getsizeof
session = CachedSession(filter_fn=lambda resp: getsizeof(resp.content) < 1_000_000)
```

You can also **allowlist by URL** patterns using `urls_expire_after`, setting `DO_NOT_CACHE` for everything else. ([requests-cache][4])

---

## Expiration, revalidation & freshness: the knobs you’ll actually use

**Where does expiration come from?** Precedence for a given request:

1. **Cache-Control response headers** (if `cache_control=True`)
2. **Cache-Control request headers**
3. Per‑request `expire_after=...`
4. Per‑URL `urls_expire_after={pattern: ...}`
5. Session‑level `expire_after=...` ([requests-cache][5])

**TTL values** can be seconds, `timedelta`, `datetime`, or special constants:

* `NEVER_EXPIRE` (store indefinitely)
* `EXPIRE_IMMEDIATELY` (treat as expired, enables revalidation)
* `DO_NOT_CACHE` (skip cache entirely) ([requests-cache][5])

**Manual refresh & “offline mode”**:

* `refresh=True` (like **F5**) revalidates/refreshes the cached response now.
* `force_refresh=True` (like **Ctrl‑F5**) bypasses validators and always fetches anew.
* `always_revalidate=True` (validate every use when validator present).
* `only_if_cached=True` serves **only** cached data; if missing, returns **504 Not Cached** (RFC‑style behavior). ([requests-cache][5])

**Failure and async-refresh strategies**:

* `stale_if_error=True|timedelta` serves expired content if refresh fails (HTTP error or `RequestException`).
* `stale_while_revalidate=True|timedelta` returns stale immediately and kicks off a **non‑blocking** refresh for next time (library sends a background refresh). ([requests-cache][5])

---

## Cache headers support (if you enable `cache_control=True`)

`requests-cache` implements private cache semantics for the common HTTP caching headers: `Cache-Control`, `ETag`, `Last-Modified`, `Expires`, and `Vary`, plus extensions such as `stale-if-error` and `stale-while-revalidate`. It will automatically send conditional requests (`If-None-Match`, `If-Modified-Since`) when appropriate. **Note:** response header handling is **opt‑in** via `cache_control=True`. ([requests-cache][6])

---

## Request matching & cache keys (how hits are determined)

Matching uses method, URL, query params, request body, and can include select headers. Normalization handles inconsequential differences (e.g., ordering). You can:

* **Ignore volatile params** (e.g., tokens) with `ignored_parameters=[...]`.
* **Match headers** explicitly with `match_headers=True|['Accept-Language', ...]` (also honors server `Vary` when present).
* Provide a **custom key function** via `key_fn` to fully control cache keys.
* If you change matching policy or upgrade to a release that changes matching, you can **recreate keys** (`session.cache.recreate_keys()`) to reindex existing cache data. ([requests-cache][7])

Security note: common credential fields (e.g., `Authorization`, `X-API-KEY`, `access_token`, `api_key`) are **ignored by default** so you don’t persist secrets. ([requests-cache][8])

---

## Storage backends (pick one you can operate in prod)

* **SQLite (default)**: easiest start, file‑based `.sqlite`.
* **Filesystem**: stores as individual files (nice with JSON/YAML serializers).
* **Redis, MongoDB, DynamoDB, GridFS**, or **in‑memory** (`backend='memory'`) are available by name. Some (Redis, MongoDB, DynamoDB) can **auto‑expire** entries via native TTL. ([requests-cache][9])

Backends are chosen with `backend='sqlite'|'redis'|...` or by passing a backend instance. The docs include performance guidance (e.g., SQLite typically isn’t a bottleneck until hundreds of requests/sec). ([requests-cache][9])

**Paths & locations (file‑based backends)**
Use relative/absolute paths or convenience options `use_temp=True` or `use_cache_dir=True` to place files under system temp or cache directories. ([requests-cache][10])

---

## Serializers & security (what’s actually stored)

By default, responses are pickled. You can switch to **JSON**, **YAML**, or **BSON** serializers (handy for human‑readable caches, portability, or Mongo‑friendly documents). Install extras like `requests-cache[json]` or `[yaml]`, and pass `serializer='json'|'yaml'|'bson'` or a custom **SerializerPipeline**. ([requests-cache][11])

If you must use pickle but want tamper‑detection, use **`safe_pickle_serializer(secret_key=...)`** (requires `requests-cache[security]`), which signs entries and rejects modified data on read. ([requests-cache][8])

---

## Observability & cache ops

Every response gets extra attributes:

```python
r = session.get(url)
print(r.from_cache, r.created_at, r.expires, r.is_expired, r.cache_key)
```

You can list URLs, filter responses, find expired entries, and delete by URL, request, or key:

```python
session.cache.urls()
session.cache.filter(expired=True)
session.cache.delete(expired=True)
session.cache.clear()  # nuke
```

These APIs make it easy to add metrics or scheduled maintenance. ([requests-cache][12])

---

## Advanced `requests` features (still work)

* **Event hooks**: You can hook into the response pipeline and branch on `response.from_cache`, e.g., throttle **only** non‑cached requests.
* **Streaming**: `stream=True` works; cached responses are re‑readable, so your iteration code is the same for cache hits and misses. ([requests-cache][13])

---

## Interop with other libraries

`requests-cache` provides a **mixin** so you can combine it with other `requests.Session`‑based libraries:

* Works with **Requests‑HTML**, **Requests‑OAuthlib**, and **Requests‑Ratelimiter** (put caching *before* rate limiting so cache hits don’t count).
* Plays well with **requests‑mock** via disabling/patching patterns or by mounting mock adapters on a `CachedSession`. ([requests-cache][14])

> Async stack? Use **`aiohttp-client-cache`**, a sister project providing similar features for `aiohttp.ClientSession`. ([Aiohttp Client Cache][15])

---

## Production patterns for AI agents

**1) Sensible defaults for read‑heavy agent calls**

```python
from datetime import timedelta
from requests_cache import CachedSession, DO_NOT_CACHE

session = CachedSession(
    'agent_cache',
    expire_after=timedelta(minutes=30),
    cache_control=True,               # use server freshness when available
    stale_if_error=timedelta(minutes=5),
    stale_while_revalidate=timedelta(minutes=2),
    allowable_methods=('GET',),       # keep writes out unless explicitly allowed
    urls_expire_after={
        '*.static.cdn.com': None,     # NEVER_EXPIRE
        '*': DO_NOT_CACHE,            # default blocklist; opt-in endpoints below
    },
)
```

* **Why:** prefer explicit allowlisting; serve stale data briefly if upstream is flaky; revalidate in the background to keep the agent snappy. ([requests-cache][5])

**2) Cache idempotent POSTs (when your API uses POST for reads)**

```python
session = CachedSession(allowable_methods=('GET','POST'))
r = session.post(api, json={'query': 'same payload'})  # now cachable
```

Be careful to ensure idempotency; otherwise, leave POSTs uncached. ([requests-cache][4])

**3) Normalize away volatile params/headers**

```python
session = CachedSession(
    ignored_parameters=['auth-token', 'nonce'],
    match_headers=['Accept-Language'],  # or True to match all
)
```

This increases hit rate while preventing credential leakage into the cache. ([requests-cache][7])

**4) Offline mode with an explicit error if data isn’t cached**

```python
r = session.get(url, only_if_cached=True)  # returns 504 if missing
```

Great for “no network” pathways in agents. ([requests-cache][5])

**5) Choose a backend to match your deployment**

* **SQLite** for single‑host services and cron jobs.
* **Redis** for multi‑instance services; native TTL + shared cache.
* **MongoDB / DynamoDB** if those are your standard ops tools (both support TTL).
  Configure with `backend='redis'|'mongodb'|'dynamodb'` or pass backend instances. ([requests-cache][9])

**6) Production hygiene**

* Periodically `session.cache.delete(expired=True)` or rely on backend TTL.
* Track hit/miss ratios via `response.from_cache` counters.
* If you change matching logic or upgrade to a version that changes keying, run `session.cache.recreate_keys()`. ([requests-cache][5])

---

## Mapping your custom cache features → `requests-cache`

| If your custom code does this…                 | Use this in `requests-cache`                                                                      |                                  |                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------- |
| Global transparent caching                     | `requests_cache.install_cache()` (prefer sessions in large/concurrent apps) ([requests-cache][3]) |                                  |                                                 |
| Per‑URL TTLs                                   | `urls_expire_after={pattern: ttl}` (glob or regex) ([requests-cache][5])                          |                                  |                                                 |
| Fixed TTL for everything                       | `expire_after=...` on the session ([requests-cache][5])                                           |                                  |                                                 |
| Follow server freshness rules                  | `cache_control=True` (honors Cache‑Control, ETag, Expires, Vary, etc.) ([requests-cache][6])      |                                  |                                                 |
| Serve stale on upstream errors                 | `stale_if_error=True                                                                              | timedelta` ([requests-cache][5]) |                                                 |
| Serve stale immediately, refresh in background | `stale_while_revalidate=True                                                                      | timedelta` ([requests-cache][5]) |                                                 |
| Manually refresh now                           | `refresh=True` or `force_refresh=True` per request ([requests-cache][5])                          |                                  |                                                 |
| Offline/Cache‑only mode                        | `only_if_cached=True` (returns 504 if not cached) ([requests-cache][5])                           |                                  |                                                 |
| Restrict caching to certain methods or codes   | `allowable_methods`, `allowable_codes` ([requests-cache][4])                                      |                                  |                                                 |
| Filter by arbitrary logic                      | `filter_fn(response)` ([requests-cache][4])                                                       |                                  |                                                 |
| Custom cache keys                              | `key_fn=...` (or use `ignored_parameters` & `match_headers`) ([requests-cache][7])                |                                  |                                                 |
| Remove secrets from cache                      | Defaults + `ignored_parameters=[...]` ([requests-cache][8])                                       |                                  |                                                 |
| Inspect entries, list URLs, delete             | `session.cache.contains()/filter()/delete()/urls()` ([requests-cache][12])                        |                                  |                                                 |
| Use non‑pickle formats                         | `serializer='json'                                                                                | 'yaml'                           | 'bson'` (install extras) ([requests-cache][11]) |
| Sign entries to detect tampering               | `safe_pickle_serializer(secret_key=...)` (install `[security]`) ([requests-cache][8])             |                                  |                                                 |

---

## Backends: quick recipes

**SQLite (default):**

```python
CachedSession('http_cache', backend='sqlite', use_cache_dir=True)
```

**Redis:**

```python
from requests_cache import CachedSession
session = CachedSession('http_cache', backend='redis', expire_after=3600)
```

**MongoDB / DynamoDB:**

```python
CachedSession('http_cache', backend='mongodb', expire_after=3600)
CachedSession('http_cache', backend='dynamodb', expire_after=3600)
```

TTL auto‑removal is supported by Redis/MongoDB/DynamoDB backends. ([requests-cache][5])

**Filesystem + JSON for human‑readable cache:**

```python
CachedSession('cache_dir', backend='filesystem', serializer='json')
```

Paths & platform‑specific locations are configurable (`use_temp`, `use_cache_dir`). ([requests-cache][10])

---

## Introspection & debugging

```python
r = session.get(url)
print('from_cache:', r.from_cache, 'created_at:', r.created_at, 'expires:', r.expires)
print('expired?', r.is_expired, 'key:', r.cache_key)

# Show cached URLs
print(session.cache.urls())

# Delete only expired entries
session.cache.delete(expired=True)
```

These attributes & methods are built in for visibility and operational hygiene. ([requests-cache][12])

---

## Gotchas & footguns

* **Global patching** (`install_cache`) can be confusing in large codebases, in libraries imported by others, and in multi‑threaded/multiprocess contexts. Prefer explicit sessions. ([requests-cache][3])
* `cache_disabled()` is **not thread‑safe**; treat it as a local convenience around single‑threaded blocks. ([requests-cache][16])
* **Pickle** is not safe against untrusted modifications; if you share caches across trust boundaries, either switch to JSON/YAML/BSON or use `safe_pickle_serializer()` with a secret. ([requests-cache][8])
* Caching **POST** or other non‑idempotent methods can be dangerous; do it only when you’re sure the endpoint is read‑only. (Use `allowable_methods` sparingly.) ([requests-cache][4])
* If you change request‑matching behavior or upgrade to a version with different keying rules, **recreate keys** (`session.cache.recreate_keys()`) to keep using existing data. ([requests-cache][7])

---

## Testability & ecosystem

* Use `backend='memory'` for fast, ephemeral tests; or disable caching in tests with `with requests_cache.disabled(): ...`.
* Works with `requests‑mock` and other tooling using the mixin approach or by temporarily disabling/patching. ([requests-cache][14])

---

## Snippets you’ll likely reuse

**Per‑request TTL override:**

```python
session.get(url, expire_after=30)
```

**Respect server headers everywhere (and refresh on every use if validator exists):**

```python
session = CachedSession(cache_control=True, always_revalidate=True)
```

**Only serve from cache in a given code path (no network calls):**

```python
session.get(url, only_if_cached=True)   # 504 if not in cache
```

**Add a throttle that only affects real requests (not cache hits):**

```python
import time
def throttle(response, *_, **__):
    if not getattr(response, 'from_cache', False):
        time.sleep(0.1)
    return response

session.hooks['response'].append(throttle)
```

([requests-cache][13])

---

## When you need async

If your agent uses `aiohttp`, reach for **`aiohttp-client-cache`**: it mirrors this feature set for `aiohttp.ClientSession`, with similar backends and TTL controls. ([Aiohttp Client Cache][15])

---

### References (selected)

* Official docs: user guide & API (usage, backends, headers, matching, expiration, inspection, serializers). ([requests-cache][3])
* PyPI (version, Python requirement, extras). ([PyPI][1])
* Compatibility patterns (mixins, requests‑mock, ratelimiter). ([requests-cache][14])
* Advanced usage (hooks, streaming). ([requests-cache][13])
* File locations (use temp/cache dirs). ([requests-cache][10])

---

If you share a brief summary of what your custom cache currently does (keying rules, TTL policy, backends, stale behavior), I can draft a drop‑in `CachedSession` configuration (and optional migration script) tailored to your agent’s needs.

[1]: https://pypi.org/project/requests-cache/ "requests-cache · PyPI"
[2]: https://requests-cache.readthedocs.io/en/stable/user_guide/installation.html "Installation - requests-cache 1.2.1 documentation"
[3]: https://requests-cache.readthedocs.io/en/stable/user_guide/general.html "General Usage - requests-cache 1.2.1 documentation"
[4]: https://requests-cache.readthedocs.io/en/stable/user_guide/filtering.html "Cache Filtering - requests-cache 1.2.1 documentation"
[5]: https://requests-cache.readthedocs.io/en/stable/user_guide/expiration.html "Expiration - requests-cache 1.2.1 documentation"
[6]: https://requests-cache.readthedocs.io/en/stable/user_guide/headers.html "Cache Headers - requests-cache 1.2.1 documentation"
[7]: https://requests-cache.readthedocs.io/en/stable/user_guide/matching.html "Request Matching - requests-cache 1.2.1 documentation"
[8]: https://requests-cache.readthedocs.io/en/stable/user_guide/security.html "Security - requests-cache 1.2.1 documentation"
[9]: https://requests-cache.readthedocs.io/en/stable/user_guide/backends.html "Backends - requests-cache 1.2.1 documentation"
[10]: https://requests-cache.readthedocs.io/en/stable/user_guide/files.html "Cache Files - requests-cache 1.2.1 documentation"
[11]: https://requests-cache.readthedocs.io/en/stable/user_guide/serializers.html "Serializers - requests-cache 1.2.1 documentation"
[12]: https://requests-cache.readthedocs.io/en/stable/user_guide/inspection.html "Cache Inspection - requests-cache 1.2.1 documentation"
[13]: https://requests-cache.readthedocs.io/en/stable/user_guide/advanced_requests.html "Advanced Requests - requests-cache 1.2.1 documentation"
[14]: https://requests-cache.readthedocs.io/en/stable/user_guide/compatibility.html "Compatibility with other libraries - requests-cache 1.2.1 documentation"
[15]: https://aiohttp-client-cache.readthedocs.io/?utm_source=chatgpt.com "aiohttp-client-cache 0.12.4 documentation - Read the Docs"
[16]: https://requests-cache.readthedocs.io/en/stable/modules/requests_cache.session.html "Session - requests-cache 1.2.1 documentation"
