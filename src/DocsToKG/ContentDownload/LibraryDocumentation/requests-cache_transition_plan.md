Absolutely—here’s a **max-consolidation plan** to make `requests-cache` the only caching layer for metadata across `src/DocsToKG/ContentDownload`, with concrete placement, policy choices, and what you can delete afterward. I’m assuming the same “we don’t need compatibility with the old behavior” guidance you gave for Tenacity applies here too. (Feature names and behaviors are from your summary doc. )

---

# Goals (what “max consolidation” means here)

* **One cache** for all metadata HTTP traffic (GET/HEAD), owned by `requests-cache`.
* **No custom conditional request logic** (ETag/Last-Modified/304 handling) in your code.
* **No ad-hoc URL allowlists/TTLs** scattered across resolvers; one central mapping.
* **No PDF body caching** (we’ll keep streaming/atomic writes for artifacts but never store binary responses in the cache).
* **Cache intelligence lives in one place**: a single “cache policy + session factory” module, used by every caller.

---

# Where the cache lives (module boundaries)

**Authoritative home:** `src/DocsToKG/ContentDownload/networking.py`

* Replace your session factory with a **single `CachedSession` factory**. This returns the only session any resolver/pipeline uses.
* Move **all cache policy** (defaults, per-host overrides, exclusion rules, cache backend selection) into this file, alongside your Tenacity policy. Everything else—CLI, pipeline, resolvers—**stops thinking about caching** entirely and just calls `session.request(...)`.

**Touch points to enforce “one way in”:**

* `pipeline.py`: Ensure every HEAD/GET goes through the `networking` session. Remove any direct `requests.Session` creation or response-caching assumptions.
* `resolvers/*`: Replace any direct calls to `requests` or custom session creation with the centralized `networking` session import.
* `cli.py` (download path): Same rule—only use the centralized session for metadata probes (landing-page HTML, JSON APIs, etc.); the PDF stream path remains uncached.

---

# The cache surface (what is—and is not—cached)

**Cache these**:

* **Metadata GET/HEAD** from OpenAlex, Crossref, Unpaywall, arXiv OAI/metadata endpoints, PubMed/PMC/Europe PMC, Semantic Scholar, DOAJ, Zenodo, Figshare, OSF, OpenAIRE, HAL, and your “publisher landing page” HTML probes. (This is where nearly all repetition and quota waste happens.)

**Do *not* cache these**:

* **PDF bodies** (and any `application/octet-stream`/very large content). We’ll explicitly exclude by content-type/size. You still stream to temp + atomic rename as you do now.

---

# The cache policy (the knobs that replace your bespoke logic)

All of these are first-class options in `requests-cache`:

1. **Global defaults**

* **Backend**: SQLite (single-host runs). If you expect multi-host/multiprocess concurrency, choose Redis as a second profile.
* **TTL**: A conservative **default TTL** for all metadata (e.g., 24–72 hours) plus respect for server headers.
* **Respect HTTP freshness**: Enable `cache_control=True` so ETag/Last-Modified/Expires are honored; `always_revalidate=True` if you want to revalidate whenever validators exist.
* **Serve stale when upstream flakes**: `stale_if_error` (e.g., 5–15 minutes) to avoid collapsing when one provider hiccups.
* **Optionally revalidate in the background**: `stale_while_revalidate` (short window, e.g., 2–5 minutes) so a second call returns instantly while a refresh is kicked off.

2. **Method and code allowlist**

* **Allowable methods**: `GET`, `HEAD` (default). Add `POST` **only** for known idempotent “search” endpoints (some Graph APIs use POST for queries).
* **Allowable codes**: Default (200). You can extend to 3xx/404 for specific sources if that materially helps (e.g., stable “not found” from a provider).

3. **Per-host TTLs (central allowlist)**
   Use a single `urls_expire_after` mapping to encode your **source-aware TTL strategy**. Examples you can start with (tune to taste):

* Crossref & Semantic Scholar: **1–3 days** (fast-moving metadata)
* OpenAlex, DOAJ, OSF, Zenodo, Figshare: **3–7 days**
* PubMed, PMC, Europe PMC, OpenAIRE, HAL: **2–7 days**
* arXiv metadata & OAI: **7–14 days**
* Publisher landing-page HTML: **~1 day** (links and page templates change)
  Everything else: **DO_NOT_CACHE** by default so new/unrecognized hosts don’t accidentally get cached.

4. **Binary & large responses exclusion**

* Install a **response filter** that refuses to cache:

  * Any response with `Content-Type` beginning with `application/pdf` or other binary types
  * Any response larger than a threshold (e.g., > 5–10 MB)
    This guarantees you never persist big artifacts, while HEADs for PDFs remain cacheable.

5. **Key hygiene to maximize hit-rate**

* **Ignore volatile params** globally (e.g., `utm_*`, `nonce`, `timestamp`, `session`, `api_key` if you ever pass it in a query, `mailto` fields).
* **Header matching**: Minimal by default; only opt into headers (e.g., `Accept-Language`) if a provider actually varies by them. Rely on servers’ `Vary` header when present.
* **Custom key function** (optional): If you want DOI-centric keys that normalize multiple provider URLs to the same logical document, you can add a DOI-aware `key_fn`. Start without this; only add it if you see real duplication that the cache isn’t collapsing.

6. **Safety & serialization**

* Default **SQLite + pickle** is fine for single-host runs. If you will share caches across trust boundaries or want tamper detection, switch the serializer to JSON/YAML or use the signed pickle serializer.

7. **Operator controls (per request)**

* **`only_if_cached=True`** to run “offline mode” sections (no network at all; you’ll get a synthetic 504 if it’s not cached).
* **`refresh=True`** or **`force_refresh=True`** for explicit cache busting on critical paths.
* **Per-request TTL override** (`expire_after=`) for specific operations without changing the session defaults.

(These features and behaviors are straight from your summary doc: sessions vs global patching, TTL/expiration precedence, per-URL rules, cache headers support, stale behaviors, request matching and key control, backends & serializers, hooks & streaming, inspection/ops. )

---

# How this lands in your tree (module-by-module)

## `networking.py` (the hub)

* **Replace** the session factory with a **single `CachedSession` builder** that:

  * Chooses backend (SQLite path under your runtime cache dir; toggleable to Redis).
  * Sets the **global TTL** and enables `cache_control`.
  * Installs **`urls_expire_after`** with one mapping covering *all* providers/domains you use.
  * Installs the **binary/size filter** so PDFs never get cached.
  * Sets **method/codes allowlists** (GET/HEAD; optional POST for known idempotent reads).
  * Enables **stale** policies (if you want them).
  * Exposes **a few runtime toggles** (env or config): `CACHE_ENABLED`, `CACHE_BACKEND`, `CACHE_TTL_DEFAULT`, `CACHE_TTLS_PER_HOST`, `CACHE_STALE_IF_ERROR`, `CACHE_STALE_WHILE_REVALIDATE`, `CACHE_DO_NOT_CACHE_PDFS`, `CACHE_OFFLINE_MODE`.

* **Delete** any local conditional request helpers (ETag/Last-Modified parsing, `If-None-Match`/`If-Modified-Since` management). With `cache_control=True`, `requests-cache` both **stores** validators and **sends** them on revalidation for you.

* **Keep Tenacity** wrapping the session’s request call. Caching happens first; if a response is served from cache, Tenacity doesn’t run a network attempt at all. When a refresh is needed, Tenacity handles the retry/backoff timing.

## `pipeline.py`

* **Remove all cache-adjacent logic** (e.g., “don’t re-hit if we saw this recently” guards). If it isn’t a pure functional dedupe for business logic, it should disappear; the cache is the system of record for not re-hitting.
* Ensure every HTTP call (including HEAD prechecks) uses the centralized `CachedSession`. No alternate sessions, no “test” sessions.

## `resolvers/*`

* **Zero caching code**. Providers simply build URLs/headers and call the centralized session. If a resolver currently “remembers” that it fetched a landing page recently, delete that; the session cache makes that memory redundant.

## `cli.py` (download path)

* The **PDF download path remains uncached** (still stream → temp → atomic rename). The **pre-download HEAD** (if you perform it) becomes an ordinary `HEAD` through the cached session, so you won’t refetch the same headers for the same URL during a run (or between runs, within TTL).

## `telemetry/*`

* Start emitting `from_cache`/`created_at`/`expires` counters (available on every response). That gives you hit/miss, age, and expiry metrics without extra plumbing.
* Add one metric for **offline-served** (requests answered with `only_if_cached=True`) if you plan to use that mode.

---

# Policy starter set (so you don’t have to think about knobs on day 1)

**Defaults**

* Backend: SQLite; path `~/.cache/docstokg/http_cache.sqlite` (or your run context dir)
* Global TTL: 48h
* `cache_control=True`, `always_revalidate=True`
* `stale_if_error=10m`, `stale_while_revalidate=2m`
* Methods: GET, HEAD only
* Codes: 200 only
* Filter: don’t cache `application/pdf` or responses > 5 MB
* Ignored parameters: `utm_*`, `nonce`, `timestamp`, `session`, `auth`, `api_key`, `mailto`, `source`
* Match headers: none (let `Vary` govern)
* Offline toggle: `CACHE_OFFLINE_MODE` → sets `only_if_cached=True` for all calls

**Per-host TTLs (allowlist)**

* Crossref, Semantic Scholar: 24–72h
* OpenAlex, DOAJ, OSF, Zenodo, Figshare: 3–7d
* PubMed/PMC/EuropePMC/OpenAIRE/HAL: 2–7d
* arXiv metadata/OAI: 7–14d
* Publisher HTML: 24h
* `*` (everything else): DO_NOT_CACHE

(These are operational starting points; tune once you see hit-rates & staleness behavior in telemetry. )

---

# Operational posture

* **Observability**: Count `from_cache` vs network; track average `age` at return; watch how often `stale_if_error` is used (spikes often reveal upstream issues before failures hit).
* **Maintenance**: Periodically purge expired entries (`delete(expired=True)`) or lean on backend TTLs if you move to Redis/Mongo/Dynamo.
* **Testing**: Use an in-memory backend or disable caching entirely in tests that need to assert network behavior; use short per-request TTLs and `refresh`/`force_refresh` to probe cache transitions.
* **Multi-host scale**: If you orchestrate parallel runners across machines/containers, move the backend to **Redis** for a shared cache and native TTL eviction. Keep the same session code; only the backend changes.

---

# What you can delete after this

* Any **manual ETag/Last-Modified** handling, request validators, or 304 branching.
* Any **“recently fetched” guardrails** layered above HTTP (unless they’re deduping at the *document* level rather than the *request* level).
* Any **per-resolver TTL notes** in code or comments; consolidate as a single mapping in `networking.py`.
* Any **home-grown cache stores** or JSONL/SQLite “seen” tables that exist purely to avoid re-hitting metadata endpoints (keep document/business-level registries that serve other purposes).

---

# Rollout sequence (fast, low-risk)

1. Introduce the **central `CachedSession`** in `networking.py` with the policy above; switch all imports/uses to it.
2. Add **telemetry** on `from_cache` and simple hit/miss counters in your run summary.
3. Delete **custom caching/validator code** across the tree.
4. Tune the **per-host TTLs** and the **filter** thresholds after you see a day of traffic.
