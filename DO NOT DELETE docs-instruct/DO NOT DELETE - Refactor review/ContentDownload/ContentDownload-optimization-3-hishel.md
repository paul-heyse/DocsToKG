Awesome — here’s a **repo-shaped, junior-dev-friendly implementation plan** to make **Hishel caching** first-class in `src/DocsToKG/ContentDownload`, aligned with RFC-9111 semantics and your HTTPX hub. It includes: config, where to put code, how requests choose cache vs raw, per-host TTLs, stale-while-revalidate, POST body keying, offline mode, instrumentation, and tests.

---

# Goals (what “good” looks like)

* **Single, centralized cache policy**—no per-resolver hacks.
* **RFC-9111-true behavior**—don’t second-guess servers; revalidate with ETag/Last-Modified; respect `Vary`.
* **Opt-in by host/role**—**known** metadata/HTML hosts are cached; PDFs (artifacts) are **never** cached.
* **Per-host TTL** and **SWrV** (stale-while-revalidate) for **metadata only**.
* **Offline** runs use `only-if-cached` (fast 504 on misses).
* **Clean telemetry**—hit rate, revalidation rate, stale served %, savings.

---

# Config surface (new)

Add a small YAML and env/CLI overlays so ops can tune without code changes.

`configs/networking/cache.yaml`

```yaml
version: 1
storage:
  kind: file              # file | memory | (future: redis)
  path: "${RUN_DIR}/cache/http"   # resolved at runtime
  check_ttl_every_s: 600  # garbage collect expired entries every 10 min
controller:
  cacheable_methods: ["GET","HEAD"]
  cacheable_statuses: [200,301,308]     # leave defaults; RFC-compliant
  allow_heuristics: false               # we will not invent TTLs
  default: DO_NOT_CACHE                 # unknown hosts are not cached
  stale_while_revalidate_s: 180         # only used for role=metadata (see policy below)
hosts:
  api.crossref.org:
    ttl_s: 172800            # 2 days
    role:
      metadata:
        ttl_s: 259200        # 3 days (override)
        swrv_s: 180          # 3 min stale-while-revalidate
      landing:
        ttl_s: 86400         # 1 day
  api.openalex.org:
    ttl_s: 172800
    role:
      metadata: { ttl_s: 259200, swrv_s: 180 }
      landing:  { ttl_s: 86400 }
  export.arxiv.org:          # OAI/metadatish endpoints
    ttl_s: 864000            # 10 days
  europepmc.org: { ttl_s: 172800 }
  eutils.ncbi.nlm.nih.gov: { ttl_s: 172800 }
  api.unpaywall.org: { ttl_s: 172800 }
  web.archive.org:
    ttl_s: 172800
    role:
      metadata: { ttl_s: 172800, swrv_s: 120 }
      landing:  { ttl_s: 86400 }
```

Env/CLI overlays (examples):

* `DOCSTOKG_CACHE_YAML=/path/to/cache.yaml`
* `DOCSTOKG_CACHE_HOST__api.crossref.org=ttl_s:259200`
* `DOCSTOKG_CACHE_ROLE__api.openalex.org__metadata=ttl_s:259200,swrv_s:180`
* CLI: `--cache-host api.crossref.org=ttl_s:259200` / `--cache-role api.openalex.org:metadata=ttl_s:259200,swrv_s:180`
* Offline: `--offline` (or `DOCSTOKG_OFFLINE=1`)

> **Policy stance:** default for unknown hosts = **DO_NOT_CACHE**. We explicitly opt-in only the hosts we trust for metadata/HTML.

---

# Module/Code layout (where to put things)

* **`networking.py`** (hub): build cached/raw clients, route every request, apply per-host policy, push `only-if-cached` for offline, and extract cache telemetry from responses.
* **`cache_loader.py`** (new): load YAML/env/CLI → `CacheConfig` dataclasses.
* **`urls.py`** (already planned): provides normalized host used as the policy key.
* **`telemetry/*`**: extend per-request logs with cache flags.

---

# 1) Build the cached & raw clients

In `networking.py` (reusing your shared SSL, limits, rate-limited transport):

* **Cached client** (for `metadata` & `landing` only):

  * Base transport = `HTTPTransport(ssl_context=shared_ctx, retries=connect_retries)`
  * Wrap with your **RateLimitedTransport** (so limits apply **only** on miss/revalidation)
  * Wrap with **Hishel CacheTransport**:

    * `storage = FileStorage(path=cfg.storage.path, check_ttl_every=cfg.storage.check_ttl_every_s)`
    * `controller = CacheController(cacheable_methods=["GET","HEAD"], cacheable_statuses=[200,301,308], allow_heuristics=False)`
    * `transport = CacheTransport(transport=rate_limited_transport, storage=storage, controller=controller)`
  * Build `httpx.Client(transport=transport, limits=..., http2=..., proxies=..., timeout=..., follow_redirects=False)`

* **Raw client** (for `artifact`/PDF):

  * `httpx.Client` with **RateLimitedTransport → HTTPTransport** only. **No CacheTransport**.

> **Quick win already baked-in:** artifacts never touch the cache—always raw streaming.

---

# 2) Central per-host policy

Add a tiny policy resolver in `networking.py`:

```text
resolve_cache_policy(host, role) -> CacheDecision {
  use_cache: bool
  ttl_s: Optional[int]          # if provided, override default TTL
  swrv_s: Optional[int]         # stale-while-revalidate seconds (metadata only)
  body_key: bool                # include POST body in cache key (rare)
}
```

Rules:

* **Unknown host** → `use_cache=False` (route to raw client)
* **Known host + role in {metadata, landing}** → `use_cache=True` with host/role TTL; if not specified, use host TTL; if not specified, **controller defaults** apply.
* **Role = artifact** → `use_cache=False` regardless of host.

> Keep this table in memory (from `cache_loader.py`) and print the effective map at startup for ops.

---

# 3) Applying TTL/SWrV/Body-key at **request** time

Hishel supports per-request **extensions** to override default behavior without touching headers. In the networking hub, just before sending a **cached** request:

* `extensions["hishel_ttl"] = ttl_s` (when you want to set/override TTL for this URL)
* `extensions["hishel_stale_while_revalidate"] = swrv_s` (**only** when `role=="metadata"`)
* `extensions["hishel_body_key"] = True` if you must cache a **POST** idempotent read (e.g., GraphQL; rare).
* For **offline mode**: add header `Cache-Control: only-if-cached` (Hishel returns a synthetic **504** if missing from cache). Also set a resolver/pipeline-visible flag so you can log `blocked_offline`.

**Important:** Do **not** set any of this for the raw client.

---

# 4) Request routing (cached vs raw) in the hub

For **every** call:

1. Get `role` (`metadata` | `landing` | `artifact`) from `request.extensions`, default `metadata`.
2. Normalize URL (`urls.canonical_for_request`); extract **host**.
3. `decision = resolve_cache_policy(host, role)`
4. Choose **client**:

   * `decision.use_cache=True` → **cached client**
   * else → **raw client**
5. If using cached client:

   * Apply `hishel_ttl`, `hishel_stale_while_revalidate` (metadata only), `hishel_body_key` per `decision`
   * If offline → header `Cache-Control: only-if-cached`
6. Send request (still wrapped by your breaker/limiter/Tenacity chain).

---

# 5) Offline mode (fast & deterministic)

* CLI flag `--offline` (or env `DOCSTOKG_OFFLINE=1`):

  * **Cached client** adds `Cache-Control: only-if-cached` to **every** request; Hishel serves from cache or returns 504 “unsatisfiable”.
  * **Raw client** calls should be guarded earlier in the pipeline (e.g., skip artifact downloads or mark them as `blocked_offline`).
  * **Telemetry**: emit `reason="blocked_offline"` whenever a request is skipped or a 504 stems from `only-if-cached`.

---

# 6) Instrumentation (how to measure)

Hook into HTTPX response:

* Read Hishel extensions (names are stable in Hishel):

  * `res.extensions.get("hishel_from_cache")    -> bool`
  * `res.extensions.get("hishel_revalidated")   -> bool`  (server returned 304)
  * `res.extensions.get("hishel_stored")        -> bool`  (fresh stored)
  * `res.extensions.get("hishel_stale")         -> bool`  (served stale)
  * `res.extensions.get("hishel_age_s")         -> int`   (age in seconds, if available)

Emit counters:

* `cache_hit_total{host,role}`  (from_cache=True)
* `cache_revalidated_total{host,role}`  (revalidated=True)
* `cache_stale_total{host,role}`  (stale=True)
* `cache_store_total{host,role}`  (stored=True)
* `cache_offline_504_total{host,role}` (offline miss)
* **Bandwidth saved estimate**: for hits/revalidations, add `content_length` (or estimate) to a running saved-bytes counter per host.

Add a **run summary** section:

* Hit rate = hits / (hits + network fetches)
* Revalidation rate = revalidated / network fetches
* % stale served
* “Saved MB” per host (rough)
* Top 5 hosts by hits

---

# 7) Edge cases & best practices

* **Respect `Vary`**: Do not set volatile headers (e.g., `Accept-Language`) on metadata requests; that would explode cache keys.
* **No PDFs** in cache: only metadata/HTML; PDFs always routed to raw client.
* **POST reads**: only cache **explicitly** when you set `hishel_body_key=True`. Default = do not cache POST.
* **Unknown hosts**: route to raw client by default (policy default is DO_NOT_CACHE), unless ops explicitly add them to the YAML.
* **Minimum TTL**: don’t invent TTL; rely on server Cache-Control/validators. TTL mapping is a **soft cap** or default when servers are silent—prefer hosts that set validators.
* **Storage**: store under `RUN_DIR/cache/http`; the loader should expand env vars at runtime.
* **Cleanup**: Hishel’s `check_ttl_every_s` will prune expired entries; add a CLI `cache vacuum` to remove dead files or nuke cache.

---

# 8) Loader and types (cache_loader.py)

Create `CacheConfig` dataclasses:

```text
CacheStorage { kind: "file"|"memory", path: str, check_ttl_every_s: int }
CacheRolePolicy { ttl_s?: int, swrv_s?: int, body_key?: bool }
CacheHostPolicy { ttl_s?: int, role?: { metadata?: CacheRolePolicy, landing?: CacheRolePolicy } }
CacheControllerDefaults { cacheable_methods, cacheable_statuses, allow_heuristics, default }
CacheConfig { storage, controller, hosts: Map[host -> CacheHostPolicy] }
```

Normalize **host keys** to lowercase + punycode (same helper you used for breakers) so policy lookup is stable.

Env/CLI overlays:

* Hosts: `DOCSTOKG_CACHE_HOST__api.openalex.org=ttl_s:259200`
* Host/role: `DOCSTOKG_CACHE_ROLE__api.openalex.org__metadata=ttl_s:259200,swrv_s:180`

Validation:

* `ttl_s>=0`, `swrv_s>=0`; if `controller.default=DO_NOT_CACHE`, ensure `hosts` is non-empty.

---

# 9) Tests (must-add)

1. **Cache selection**

   * Unknown host (metadata) → raw client chosen; known host → cached client chosen.

2. **Hit / store / revalidate**

   * First GET → `hishel_stored=True`
   * Second GET → `from_cache=True`
   * Third GET after server changes with validators → `revalidated=True` (304 flow)

3. **Stale-while-revalidate (metadata only)**

   * Expire an entry but allow `swrv_s=180` → response contains `hishel_stale=True`, and a background revalidation is triggered (Hishel semantics). Confirm subsequent request hits fresh.

4. **Offline**

   * With `only-if-cached`, first request → 504 & `blocked_offline`; second (after first stored) → hit.

5. **Role isolation**

   * landing with TTL 1d caches; artifact for same host does **not** cache.

6. **POST body keying**

   * POST with body A stored under body-key; POST with body B is a cache miss; POST again with body A is a hit.

7. **Vary sanity**

   * Set `Accept-Language` → distinct cache key; verify we don’t set it in hub by default.

---

# 10) Rollout plan

1. Land **cache_loader.py** + `cache.yaml` + policy resolver in hub.
2. Build **cached + raw** clients and route every request via `resolve_cache_policy(host, role)`.
3. Enable **offline** flag handling.
4. Add **telemetry** counters and run-summary section.
5. Canary with a small host list (OpenAlex, Crossref, Wayback).
6. Tune TTLs/SWrV from telemetry (hit/revalidate/stale %) after a few runs.

---

# 11) “Stretch” (optional, after it’s stable)

* **stale-if-error** (SIE): serve stale for N minutes when upstream 5xx; log `sie_used=true`. Keep off by default—enable per host if needed.
* **Shared cache** for multi-process/multi-host: later swap FileStorage for a Redis storage (if Hishel supports) or keep shared filesystem (ensure locking).
* **Cache directory cap**: periodic job to prune oldest entries by LRU when disk budget hit (simple file mtime heuristic).
* **Host allowlist auto-promoter**: if a currently raw host shows stable validators (ETag/Last-Modified) in telemetry for N requests, suggest adding it to cache.yaml.

---

## Acceptance criteria (DoD)

* **No PDFs** are cached; artifacts always raw.
* Known metadata/landing hosts are cached with per-host TTL & SWrV; unknown hosts are **not** cached.
* **Offline** runs use `only-if-cached` and fail fast with 504 on misses.
* Telemetry shows **hit**, **revalidated**, **stale**, and **stored** counts by host+role; run summary reports hit/revalidation/stale rates and estimated bandwidth saved.
* All requests still flow through limiter/breaker/Tenacity, and cache lookups **do not** burn limiter tokens.

If you want, I can also sketch `cache_loader.py` and the tiny `resolve_cache_policy(host, role)` helper signatures to plug straight into your hub.
