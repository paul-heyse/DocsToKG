Awesome—here’s a **repo-shaped, junior-dev-friendly plan** to add **url-normalize** to `src/DocsToKG/ContentDownload` so URL canonicalization is centralized, predictable, and leveraged end-to-end (dedupe → cache keys → rate limiting → downloads). I’ll spell out *exactly* what to add, where to call it, what to delete, and how to test it.

---

## What url-normalize gives you (why we’re using it)

* **Single call → canonical string** that applies RFC-style normalization: lowercasing scheme/host; percent-encoding normalization; dot-segment removal; IDN (punycode) handling; defaulting missing bits sensibly.
* **Explicit knobs**, not ad-hoc helpers: `default_scheme`, `default_domain`, `filter_params`, and **param allowlists**.
* Important behaviors to pin in tests: **no query sorting** (by design), **default scheme is `https`** in v2.x, and **ports are dropped only if they match the scheme’s default**.
* “Port 80 on `https`” **gotcha**: with `default_scheme="https"`, `:80` is *not* default, so **it’s kept** (drop only scheme-default ports). Lock this in tests.

---

## The target end-state (after this refactor)

* **One tiny module** owns all URL canonicalization rules.
* **Two surfaces**:

  * `canonical_for_index()` → used for **manifests/dedupe/resume** keys.
  * `canonical_for_request(role=...)` → used **right before** any HTTP call (so Hishel cache keys, HTTPX mounts, and the rate limiter all see the same normalized string).
* **Zero “cleanup” code** anywhere else (providers/pipeline/download stop lowercasing, trimming ports, stripping `utm_*`, etc.).
* **Manifest schema** has both `original_url` and `canonical_url` (canonical is the unique key).

---

## Step 1 — Add a new module: `src/DocsToKG/ContentDownload/urls.py`

Create a small, documented policy module. It will be the **only** place that imports `url_normalize`.

### 1A) Policy and defaults (module constants)

* `DEFAULT_SCHEME = "https"` (matches library v2.x).
* `FILTER_FOR = {"landing": True, "metadata": False, "artifact": False}`
  Rationale: drop tracking noise only on publisher **landing pages**; **never** on metadata (**parameters matter**), **never** on artifacts (signed URLs).
* `PARAM_ALLOWLIST = {  # domain → [param names] }`
  Start empty. You can keep this map in one place if you later decide a few sites need specific params preserved when filtering.
* `DEFAULT_DOMAIN_PER_HOST = {}` (optional). If your scraper yields **relative paths** from landing pages, you’ll pass `default_domain=<origin_host>`; the library supports this.

> These knobs give you a single switchboard for **what** to filter and **when**; everything else becomes data flow.

### 1B) Public functions

* `canonical_for_index(url: str) -> str`

  * Use the library with: `default_scheme=DEFAULT_SCHEME`; **no `filter_params`** (index keys must reflect the true URL).
  * Keep behavior purely RFC-ish: case, percent-encoding, dot-segments, default `/`, IDN, drop only scheme-default port.

* `canonical_for_request(url: str, *, role: Literal["metadata","landing","artifact"], origin_host: str | None = None) -> str`

  * Always set `default_scheme=DEFAULT_SCHEME`.
  * If `origin_host` is provided (e.g., a relative `<a href>` on a landing page), set `default_domain=origin_host`.
  * If `role == "landing"` and `FILTER_FOR["landing"]` is `True`, set `filter_params=True` and pass `param_allowlist=PARAM_ALLOWLIST` (global or domain-specific mapping).
  * If `role in {"metadata","artifact"}`, **do not** set `filter_params` (we must not drop semantically relevant or signing params).
  * Never sort query params (the library doesn’t; that’s on purpose).

> Keep these two functions tiny; the library already performs the heavy lifting per RFC (normalizing case, escapes, dot-segments, default `/`, IDNA, default-port drop).

---

## Step 2 — Wire it everywhere (call sites to change)

> The rule: **every URL that enters or leaves your system** goes through `urls.py`.

### 2A) Providers: `src/DocsToKG/ContentDownload/resolvers/*`

* When a resolver **emits** any candidate URL (API JSON, landing page, artifact link):

  1. Save the **original string** for audit.
  2. Compute `canon = canonical_for_index(original)`.
  3. Pass `canon` downstream (and store both in the manifest row that gets built for that candidate).

* If a resolver calls the network directly (it shouldn’t now), remove that and go through `networking` (Step 2C).

### 2B) Pipeline + manifest/index code (e.g., where you build `ManifestUrlIndex`)

* Store **both** `original_url` and `canonical_url`; make **`canonical_url`** the **unique/primary** key used for dedupe/resume.
* Delete any “local normalization” helpers here: no lowercasing, no `:80` stripping, no manual UTM stripping—**all** of that moves to `urls.py` (or is handled by the library).
* For *relative* links captured during landing-page scraping, also attach `origin_host` so **later** network calls can call `canonical_for_request(..., origin_host=...)`.

### 2C) Networking hub: `src/DocsToKG/ContentDownload/networking.py`

* Right **before every HTTPX call** (HEAD/GET), canonicalize **with role**:

  * Metadata calls (OpenAlex/Crossref/etc.): `canonical_for_request(url, role="metadata")`
  * Landing-page HTML fetches: `canonical_for_request(url, role="landing", origin_host=...)`
  * Artifact (PDF) downloads: `canonical_for_request(url, role="artifact")`
* Pass the **canonical string** into HTTPX/Hishel. This ensures:

  * **Hishel** cache keys are built from normalized URLs (fewer duplicate entries).
  * Your **RateLimitedTransport** sees a normalized host (stable limiter keys).
  * HTTPX **mounts**/transports match consistently if you route on host.
* Delete any in-module URL cleanup (lowercasing, port tweaks, manual de-UTM). The hub shouldn’t alter URLs anymore—it just calls `urls.py`.

### 2D) Download strategies: `src/DocsToKG/ContentDownload/download/*`

* Ensure **artifact** GETs call `canonical_for_request(..., role="artifact")` and **never** filter parameters (CDN signatures!).

### 2E) Telemetry

* Where you summarize run outcomes, add counters:

  * `urls_normalized_total`, `urls_filtered_total` (landing only), and a sample of `(original → canonical)` pairs (redact query if sensitive).
* For cache stats, keep reading Hishel fields; normalization should **increase** hit-rate because trivial variants collapse. (The normalized URL feeds Hishel as input.)

---

## Step 3 — Delete dead code (and why it’s safe)

Remove **all** ad-hoc URL cleanup across the repo:

* Lowercasing scheme/host, percent-escape case fixes, dot-segment removal, default `/` path, default port dropping, IDN normalization → **now done by the library**.
* “Strip UTM” helpers → **use `filter_params=True` only for landing pages** via `urls.py`.
* Any “sort query params” code → **delete it** (out of scope; order can be meaningful, and the lib intentionally does not sort).

---

## Step 4 — Tests you must add (goldens you can copy/paste)

Create `tests/content_download/test_urls.py` with **golden IO**:

1. **Case & escapes**: `HTTP://User@ExAmple.COM/%7e/foo%2a` → `http://User@example.com/~ /foo%2A` (uppercased escapes, host lowercased).
2. **Dot segments + default “/”**:

   * `http://example.com` → `http://example.com/`
   * `http://example.com/a/./b/../c` → `http://example.com/a/c`
3. **IDN**: `http://münich.example/straße` punycodes host, encodes path.
4. **Default ports**:

   * `http://example.com:80/` → drop `:80`
   * `https://example.com:443/` → drop `:443`
5. **Port-80 gotcha under https**: `www.example.com:80/foo` (with default `https`) → **keep `:80`**; set `default_scheme="http"` if you want it dropped.
6. **Relative path → absolute**: `/img/logo.png` with `origin_host="example.com"` → `https://example.com/img/logo.png`.
7. **Landing param filtering**:

   * Input: `https://site.com/p.pdf?utm_source=x&ref=tw&q=test`
   * With `role="landing"` and `filter_params=True`, expect only `q=test` (allowlist optional).
8. **No param sorting**: assert the library preserves query order exactly (this protects cache keys & semantics).

> **Pin the library** (e.g., `==2.2.*`) so your goldens stay stable across CI/CD.

---

## Step 5 — CLI & config surface (small but useful)

* **Env flags** (read once at startup):

  * `DOCSTOKG_URL_DEFAULT_SCHEME` (default `https`)
  * `DOCSTOKG_URL_FILTER_LANDING` (`true|false`)
  * `DOCSTOKG_URL_PARAM_ALLOWLIST` (e.g., `site.com:page,id;example.org:id`)
* Optional **CLI** flags that flip the same switches (useful when you run batch jobs from the command line).

---

## Step 6 — Interactions with the rest of your stack

* **Hishel (HTTP caching)**: we normalize **before** calling the client, so cache keys collapse trivial variants (fewer duplicate entries, fewer revalidations).
* **pyrate-limiter**: limiter keys are based on the **normalized host**, so rate policies apply consistently even if inputs vary by case or accidental ports.
* **Tenacity**: unchanged—normalization doesn’t affect retry logic; it just makes each attempt talk to the same canonical URL.
* **Wayback fallback**: run normalization **before** querying CDX/Wayback; still keep the `original_url` around for audit.

---

## Step 7 — Migration & cleanup checklist (for the PR)

1. Add `urls.py` with the functions & policy constants above.
2. Update **providers** to emit `canonical_for_index(...)` and carry the `original_url` alongside it.
3. Update **pipeline/manifest** to store `canonical_url` as the unique key; add `original_url` for audit.
4. Update **networking** to call `canonical_for_request(...)` with the correct **role** before **every** HTTP call.
5. Update **download** to use `role="artifact"`; do **not** filter params.
6. Delete all bespoke URL normalization/cleanup helpers across the repo.
7. Add the **tests** outlined above.
8. Pin `url-normalize` in your environment and record test goldens.

---

### Final callouts that “move the needle”

* Put **all** URL logic in a **single file** (`urls.py`). This is the single biggest consolidation win—you’ll delete a surprising amount of string-munging across providers and pipeline.
* Treat **landing pages differently**: enable `filter_params` there and nowhere else; this preserves semantics/signatures while still collapsing noisy tracker variants.
* Lock in the **default-scheme and port rules** with tests, especially the **`:80` on `https`** trap.
* Normalize **before** Hishel and the limiter so **both** benefit from stable keys (more cache hits, fewer unexpected limiter keys).

If you’d like, I can follow this with a one-page docstring template for `urls.py` that spells out these policies in-line for agents.
