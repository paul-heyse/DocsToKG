Absolutely — here’s a **best-in-class, code-free, agent-ready implementation plan** to finish the **HTTPX cut-over & client policy** for `src/DocsToKG/OntologyDownload`. It removes all `requests/SessionPool` usage, stands up a single, well-tuned HTTPX + Hishel client, centralizes redirect safety, and wires tests and docs so regressions can’t hide.

---

# 0) North-star & Definition of Done

**Goal:** Every network call in OntologyDownload goes through **one shared HTTPX client** with explicit **timeouts**, **pool limits**, **HTTP/2**, **SSL**, **cache (Hishel)**, **redirect audit** (no auto-follow), and **structured events** (`net.request`). No `requests` or `SessionPool` remains. Retries are split: **transport connect-retries** only at the client layer; **status-aware backoff** (429/5xx) lives in Tenacity at the call-site.

**Done when:**

* `grep -R "requests\.\|SessionPool"` → **0 matches**.
* All call-sites import/use `get_http_client()` and **stream** bodies.
* Redirects are **off globally**; every hop is validated by the **URL security gate** before follow.
* Events `net.request` emitted for every attempt; `ratelimit.acquire` already in place (from the rate-limiter item).
* Unit/component tests cover 200/206/302/429/5xx/timeouts, with MockTransport & in-process ASGI.
* Docs updated; Settings v2 fields are the **single source of truth** for client behavior.

---

# 1) Files & surfaces (exact paths)

```
src/DocsToKG/OntologyDownload/
  net/
    __init__.py
    client.py            # Client factory, lifecycle, hooks, request helpers
    policy.py            # Timeouts, pool defaults, UA, header policy, range policy
    instrumentation.py   # net.request event builder
  settings.py            # HttpSettings, CacheSettings, RetrySettings (strict Pydantic v2)
  policy/url_gate.py     # (authoritative) validate_url_security(url, http_config)
  download.py / planners # (call-sites) now use get_http_client(); no requests
tests/ontology_download/net/
  test_client_happy.py
  test_redirect_audit.py
  test_timeouts_and_pool.py
  test_status_retries_tenacity.py
  test_caching_hishel.py
  test_streaming_and_memory.py
```

**Removal guard in CI:**

```bash
grep -R "requests\.\|SessionPool" src/DocsToKG/OntologyDownload && { echo "requests still present"; exit 1; } || true
```

---

# 2) Settings (strict Pydantic v2)

Add/confirm in `settings.py` (all contribute to `config_hash`):

**HttpSettings**

* `http2: bool = True`
* `trust_env: bool = True`                     # honor HTTP(S)_PROXY, NO_PROXY
* `user_agent: str = "DocsToKG/OntoFetch (+…)"`

**TimeoutSettings** (per-phase, seconds; all >0)

* `connect: float = 5.0`
* `read: float = 30.0`
* `write: float = 30.0`
* `pool: float = 5.0`                          # acquire-from-pool timeout

**PoolSettings**

* `max_connections: int = 64`
* `max_keepalive_connections: int = 20`
* `keepalive_expiry: float = 30.0`

**CacheSettings** (Hishel)

* `enabled: bool = True`
* `dir: Path = <platformdirs user cache>/ontofetch/http`
* `bypass: bool = False`                       # for forced revalidation tests

**RetrySettings (transport only)**

* `connect_retries: int = 2`
* `backoff_base: float = 0.1`
* `backoff_max: float = 2.0`

*(Status-aware retries stay in Tenacity.)*

---

# 3) Client design (`net/client.py`)

## 3.1 Lazy singleton bound to config_hash & PID

* `_CLIENT: Optional[httpx.Client] = None`
* `_BIND_HASH: Optional[str] = None`
* `_BIND_PID: Optional[int] = None`

```text
def get_http_client(settings) -> httpx.Client:
  if _CLIENT is None or os.getpid()!=_BIND_PID:
      # (Re)build client (PID-aware for fork safety)
  elif settings.config_hash != _BIND_HASH:
      # Warn once: settings changed post-bind; keep current client
  return _CLIENT
```

**Provide** `close_http_client()` and `reset_http_client()` for tests/CLI teardown.

## 3.2 Construction

* **Timeouts**: `httpx.Timeout(connect, read, write, pool)`
* **Limits**: `httpx.Limits(max_connections, max_keepalive_connections, keepalive_expiry)`
* **HTTP/2**: set `http2=True` (leverages concurrency to same host)
* **SSL**: build an `SSLContext` from system trust (or `truststore`/`certifi` if you already use it). Always verify; SNI on.
* **Proxies**: `trust_env=settings.http.trust_env`; allow per-scheme/per-host mounts later if required.
* **Cache**: wrap with **Hishel** if enabled; set cache directory from `CacheSettings.dir`.
* **Redirects**: **disabled globally** (don’t pass `follow_redirects=True`); we manage redirect audit manually.
* **Headers**: configure default headers (User-Agent, Accept, Accept-Encoding) in the client.

## 3.3 Hooks & events

Install **request**/**response**/**error** hooks:

* `on_request`

  * allocate a `request_id` (UUID), attach to request extensions
  * stamp standard headers (UA etc.) if absent
  * record `t0` (monotonic) in request state
  * compute `url_redacted` (using your redactor)
  * add route tags (`service`, `host`) to event **ids**

* `on_response`

  * compute `ttfb_ms` and `elapsed_ms`
  * `bytes_read` from `Content-Length` if known, else from stream accounting if you track
  * `http2`, `reused_conn` (via response extensions if exposed or request info)
  * `cache` = `"hit"|"revalidated"|"miss"|"bypass"` (Hishel exposes cache metadata; infer from status 304 or headers otherwise)
  * emit **`net.request`**

* `on_error`

  * map exceptions to error codes:
    `ConnectTimeout → E_NET_CONNECT`, `ReadTimeout → E_NET_READ`, `ConnectError → E_NET_CONNECT`, `RemoteProtocolError → E_NET_PROTOCOL`, `TLSException → E_TLS`, `TooManyRedirects → E_REDIRECT_LOOP`
  * include `elapsed_ms` and `url_redacted`
  * emit **`net.request`** with `level="ERROR"` + `error_code` then re-raise

---

# 4) Redirect audit (safety) (`net/client.py` + `policy/url_gate.py`)

**Principle:** Auto-redirects off. **Every** hop is explicitly validated.

```text
async def request_with_redirect_audit(client, method, url, *, max_hops=5, headers=None, **kw):
  current = url
  for hop in range(max_hops):
      # 1) send request
      resp = await client.request(method, current, headers=headers, **kw)
      # 2) if 3xx with Location:
      if 300 <= resp.status_code < 400 and "location" in resp.headers:
          target = resp.headers["location"]
          # 3) normalize to absolute against current
          target_abs = urljoin(current, target)
          # 4) validate URL (authoritative URL gate)
          validate_url_security(target_abs, http_config=settings.security)
          # 5) follow (loop)
          current = target_abs
          continue
      return resp
  raise TooManyRedirects
```

* Always **validate** target with the **single** URL gate (punycode, PSL allowlist, per-host ports, http→https upgrades, DNS private-net policy).
* **Never** propagate Authorization across **host** changes automatically.
* Emit **one** `net.request` event **per hop**.

---

# 5) Tenacity & retry split (call-site policy)

**Transport-level** (inside client): retry **connect** errors **only**, small count (e.g., 2), with capped backoff; do **not** retry after bytes are sent.

**Status-aware** (call-site, e.g., downloader):

* 429: parse `Retry-After`; call `ratelimit.cooldown_for(key, seconds)` (from your rate-limiter item); then Tenacity **sleeps** and retries.
* 5xx: retry **idempotent** methods (GET/HEAD) with jitter & bounded attempts.
* 4xx: **no** retry except 408/409/429 when your policy says so.

**Important:** This separation avoids double waits and keeps client logic simple.

---

# 6) Streaming, Range, and memory discipline

* For **downloads**: always `client.stream("GET", url, …)`, then write to **temp in final dir** → `fsync` → **rename** (your secure extraction plan depends on this discipline).
* For **metadata probes**: do **not** buffer bodies; read headers only (or `HEAD` **if the provider is known reliable** for HEAD).
* **Range**: range resume logic lives in downloader; client doesn’t auto-range unless you add a helper (optional).
* Ensure **no accidental eager `.read()`** at client helpers; tests should monitor memory.

---

# 7) Caching semantics (Hishel)

* When `CacheSettings.enabled=True`, wrap client with Hishel transport and a disk cache at `CacheSettings.dir`.
* Revalidation behavior: rely on Hishel to add `If-None-Match`/`If-Modified-Since`; your code just interprets 304 results.
* Provide a `bypass_cache` flag (from settings or per request) to disable caching during certain operations (e.g., forced re-download).

---

# 8) Header policy (predictable & minimal)

* Default headers sent once by the client:

  * `User-Agent: settings.http.user_agent`
  * `Accept: */*` (or narrow for probes if needed)
  * `Accept-Encoding: gzip, deflate, br`
* Do **not** set random headers locally; add per-service auth via pluggable **auth** if required (e.g., httpx-auth), but keep out of downloader core.
* **CookieJar**: disabled by default (stateless); enable only if a resolver requires it.

---

# 9) Tests (deterministic; no real network)

**Fixtures:**

* `http_mock`: HTTPX **MockTransport** with scripted responses (200/206, 302 chain, 429 Retry-After, 5xx, connect/read timeouts).
* `asgi_app`: in-process **ASGI** app to simulate streaming bodies, chunked responses, and conditional GET/HEAD behavior.
* `settings_fixture`: traced settings with a temp cache dir; small timeouts.
* `event_sink`: capture `net.request` events and assert shapes.

**Suites:**

1. **Happy path** (`test_client_happy.py`)

   * 200 with 128 KiB body: emit event with `elapsed_ms>0`, `bytes_read≈131072`, `cache` present when Hishel enabled.

2. **Redirect audit** (`test_redirect_audit.py`)

   * 302→200 same host: two `net.request` events; OK.
   * 302 cross-host to disallowed: `validate_url_security` rejects; error event `E_REDIRECT_DENY` and exception propagated.
   * Loop (6 hops): `TooManyRedirects`, error event `E_REDIRECT_LOOP`.

3. **Timeouts & pool** (`test_timeouts_and_pool.py`)

   * Connect timeout triggers `E_NET_CONNECT`; Read timeout `E_NET_READ`.
   * Pool exhaustion with 65 concurrent GETs blocks or errors after `pool` timeout (assert behavior) — event counts & latencies make sense.

4. **Status retries (call-site)** (`test_status_retries_tenacity.py`)

   * 429 with `Retry-After: 1`: ensure **ratelimit.cooldown** + **sleep** then success; `net.request` events show two attempts; total wall-time ~1s (not ~2s).
   * 5xx: retry GET only; POST not retried.

5. **Caching** (`test_caching_hishel.py`)

   * First GET: cache miss; second GET: `cache:"hit"` or `"revalidated"` with 304.
   * With `bypass=True`: both are misses.

6. **Streaming/memory** (`test_streaming_and_memory.py`)

   * Mock a large body (e.g., 50 MB): assert streaming writes to temp; no huge memory spikes; single `net.request` event captured.

---

# 10) Docs & Runbook

* **README (Networking)**

  * Explain “**one client**” rule; explicit timeouts; HTTP/2; redirects off; streaming discipline.
  * Show how to **add a new service** (per-service base URL, auth, rate-limit override).
  * Describe **caching** (Hishel) and how to bypass for debugging.

* **Settings reference**

  * Enumerate HTTP/Pool/Cache/Retry fields; examples of `.env` and CLI overlay.

* **Troubleshooting**

  * “I still see waits after Retry-After” → check for double wait; ensure limiter doesn’t also sleep.
  * “We leak connections” → verify response streaming closed; use context manager or `.aclose()`.

---

# 11) Performance & budgets

* **Overhead** of hooks + event emission: keep `net.request` handling < 200 µs on MockTransport.
* **Cold start** of client: < 10 ms (without cache dir creation).
* **P95 elapsed** (MockTransport 200/128 KiB): < 5 ms on CI class (informational).

Add a small **pytest-benchmark** microbench (optional) with MockTransport to monitor drift.

---

# 12) Rollout (small, safe PRs)

**PR-HTTP1 — Client skeleton & settings**

* Add `net/client.py`, `net/policy.py`, `net/instrumentation.py`; implement `get_http_client()`; **don’t** switch call-sites yet.
* Add unit tests for `on_request/on_response/on_error` hooks (MockTransport).

**PR-HTTP2 — Redirect audit & URL gate**

* Finish `request_with_redirect_audit()`; ensure it uses the **single** `validate_url_security`.
* Add redirect tests (safe/unsafe/loop).

**PR-HTTP3 — Call-site cut-over**

* Replace `requests/SessionPool` usage in downloader/planners with `get_http_client()` + `request_with_redirect_audit()`.
* Ensure streaming writes and Tenacity policy (429/5xx) remain intact.
* Add CI **grep guard** to fail on any remaining `requests/SessionPool`.

**PR-HTTP4 — Cache & docs**

* Wire Hishel cache; add bypass option; write caching tests.
* Update README & Settings pages.

---

# 13) Acceptance checklist (paste into PR)

* [ ] Single **HTTPX client** (lazy singleton) with explicit timeouts, pool, HTTP/2, SSL; **no** `requests/SessionPool` left.
* [ ] Redirects **off** globally; `request_with_redirect_audit` validates every hop with the **authoritative URL gate**; cross-host safe.
* [ ] Streaming everywhere (no accidental full `.read()`); large bodies don’t spike memory.
* [ ] Transport retries only for **connect** errors; **status** retries (429/5xx) handled by Tenacity with **cooldown_for** interplay (no double wait).
* [ ] `net.request` events emitted for every attempt; shapes validated in tests.
* [ ] Hishel cache integrated; hit/revalidated/miss visible in events; bypass honored.
* [ ] Tests pass: happy path, redirect audit, timeouts/pool, status retries, caching, streaming/memory.
* [ ] Docs updated; settings reference explains knobs; troubleshooting included.

---

## Quick “mental model” for juniors

> **“Every HTTP call goes through one well-behaved client.”**
>
> * The client is **strict**: explicit timeouts, capped pool, **HTTP/2**, safe SSL, **no auto-redirects**.
> * We **stream** everything large.
> * A **cache** (Hishel) saves bandwidth; we can bypass when needed.
> * If we need to follow redirects, we **audit** each hop through our **URL security gate** first.
> * We **measure** everything — each attempt emits a `net.request` event with timing, cache status, and outcome.
> * We **retry** only at the right layer: quick connect glitches at the client; status codes with Tenacity.

This plan lands a predictable, observable, and secure network stack that **everyone** can reason about and **nobody** will need to re-implement ad hoc.
