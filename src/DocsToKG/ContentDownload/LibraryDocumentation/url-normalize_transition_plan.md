Here’s a **max-consolidation implementation plan** to make **url-normalize** your *single source of truth* for URL canonicalization across `src/DocsToKG/ContentDownload`, with exactly where to plug it in, what to delete, and how it interacts with your resume/dedupe and requests-cache layers.

---

## What we’re adopting (and why)

* **url-normalize** converts any URL-ish string to a canonical form aligned with RFC 3986/3987: lowercases scheme/host, normalizes percent-escapes, removes dot-segments, drops default ports, handles IDNs, and can optionally filter “tracker” query params. One API: `url_normalize(...)`.
* 2.x changed defaults that matter: **default scheme is `https`**, **no query-param sorting** (order can be semantically meaningful), and **IDNA2008/UTS46** for internationalized hostnames—plan tests and policy accordingly.

---

## Design goal

* **One canonicalizer**, used everywhere a URL enters your system: provider outputs, landing-page scrapes, manifest/resume hydration, and request issuance.
* **No home-grown cleanup left** (no manual lowercasing, dot-segment removal, “:80” stripping, etc.). url-normalize alone does that; *you* only define policy (which scheme to assume, when to filter params, when to add a default domain).

---

## Where this lives in your tree

1. **Create a tiny, authoritative module**: `src/DocsToKG/ContentDownload/urls.py` (or `urlcanon.py`).
   Expose exactly two functions (names your choice):

   * `canonical_for_index(url: str) -> str` — used for **dedupe, manifests, resume**.
   * `canonical_for_request(url: str, *, role: Literal["metadata","landing","artifact"], origin_host: str | None = None) -> str` — used **right before** issuing the HTTP request, with per-role policy switches (below).

   Rationale: you keep one place for policy knobs, and you don’t spread per-role exceptions across resolvers and download paths.

2. **Call sites (make them all go through the new module):**

   * **ResolverPipeline & providers**: normalize immediately when a provider yields a candidate link; store the canonical in memory and forward it downstream (no per-provider cleanup anymore). Your pipeline already centralizes resolver orchestration; this is the right choke point.
   * **Global URL dedupe & indices**: feed `ManifestUrlIndex` with *canonical* only; persist both `original_url` and `canonical_url` in the manifest DB/JSONL, index on `canonical_url`. (Your README already calls out global dedupe and manifest indices; this change just makes canonicalization explicit and uniform.)
   * **Networking**: before **every** GET/HEAD, run `canonical_for_request(...)`, then call the cached session. This ensures requests-cache keys align (request-matching uses method+URL+params).
   * **Landing-page scraper**: when you resolve relative `<a href>`s, call `canonical_for_request(..., role="landing", origin_host=page_host)` so `default_domain` can be applied correctly. (The library supports default domain/scheme.)

---

## The policy (small, explicit, and safe)

### A) Defaults that apply everywhere

* **`default_scheme="https"`** to align with v2.x behavior (be conscious of port semantics—see the gotcha below).
* **No query sorting** (the library removed it by design). If you previously sorted for cache keys, keep that outside url-normalize (or better, stop sorting).

### B) Role-specific toggles

* **`role="metadata"` (APIs like OpenAlex, Crossref, etc.):**
  *Don’t* filter params; APIs often depend on exact param presence and order (and requests-cache keys include them). You still get case, path, port, percent-escape, and IDN normalization “for free.”
* **`role="landing"` (publisher HTML → PDF links):**
  Enable `filter_params=True` to drop common trackers (`utm_*`, etc.). Optionally add a conservative **allowlist** for params you *know* affect link resolution on a few domains. (url-normalize supports `filter_params` with a global or domain→allowlist map.)
* **`role="artifact"` (direct PDF endpoints):**
  *No param filtering*; some CDNs use signed query strings where *any* change breaks the URL. You still get safe normalization (lowercasing host, dot-segments, etc.), but avoid changing query content.

### C) Relative URLs

When the scraper yields `/pdfs/v1/123.pdf`, pass `default_domain=origin_host` so canonicalization yields a fully qualified URL. (Supported natively.)

### D) Port gotcha (pin it in tests)

With the **default scheme now `https`**, a URL like `www.example.com:80/foo` will *keep* `:80` (80 isn’t the default for HTTPS). Only **drop** a port when it equals the **scheme’s** default; don’t force-drop `:80` unless you explicitly set `default_scheme="http"`. Bake this into golden tests so behavior never surprises you.

---

## Interaction with requests-cache (this is a big win)

* requests-cache’s keying includes **method, URL, params, and optionally headers**; pre-normalizing the URL **raises hit rate** by collapsing trivial variants (case, default port, dot-segments, percent-escape case). Keep header matching minimal unless a provider truly varies by it.
* If you plan to ignore volatile params globally (e.g., `nonce`), do it **in requests-cache** via `ignored_parameters`, *not* in url-normalize—keep normalization and cache keying clearly separated.

---

## Concrete edits (module-by-module)

### 1) `networking.py` (the hub)

* **Add** a tiny call to `urls.canonical_for_request(...)` right before any session request is made (HEAD and GET). This keeps all networking and caching in lockstep with canonicalization. (Your README documents `request_with_retries()` and `head_precheck()` as the entry points—touch them both.)
* **Delete** any in-module URL cleanup (lowercasing host, removing ports, path dot-segments, etc.). url-normalize now owns that.

### 2) `resolvers/*` (all providers)

* **At the moment of URL emission**, call `urls.canonical_for_index(...)` and pass that forward. This ensures the same canonical form flows into global dedupe and manifests, regardless of which resolver produced it. (Resolver orchestration already centralizes flow through `ResolverPipeline`.)
* **Remove** any per-resolver “strip UTM”/“fix slashes” helpers—move domain-specific exceptions (if any) into a single allowlist map inside `urls.py`.

### 3) `pipeline.py`

* Where you insert into **`ManifestUrlIndex`** and other dedupe structures, ensure you store the **canonical URL** (and optionally the original as metadata for audit/debug). Your README already references that index and the resume/dedupe path; this change just standardizes the key.
* **Remove** any pre-insert normalization or param trimming present here; routes now trust `urls.py`.

### 4) `download/*` (strategies)

* **Before fetching HTML/PDF/XML**, run `canonical_for_request(...)`. Keep the existing streaming/atomic-write logic intact; normalization is orthogonal. (Your DownloadStrategy & finalization flow remains unchanged.)

### 5) Telemetry & schema

* Extend manifest entries to carry **`original_url`** and **`canonical_url`** where today you record only one “normalized URL” string. Make `canonical_url` the **unique index** used by resume/dedupe; treat `original_url` as informative. (Your sinks and schema versions are already called out in the README; bump the schema version accordingly.)

---

## What you can delete (and never revisit)

* Any code that:

  * lowercases scheme/host,
  * removes `.` / `..` path segments,
  * uppercases percent-escapes or decodes unreserved chars,
  * drops default ports,
  * strips common tracking params in landing-page code.
    url-normalize does **all** of this, and lets you toggle filtering when you want it.

---

## Guardrails (where to **not** normalize aggressively)

* **Signed/CDN URLs** (S3 query signatures, `X-Amz-*`, temporary `token=` params): do **not** filter or reorder params; pass `role="artifact"` (no filtering).
* **Site-specific canonicalization** (e.g., removing `index.html`, collapsing vanity aliases): url-normalize **won’t** do this and shouldn’t—keep such rules explicit and minimal if you need them.

---

## Testing & acceptance

1. **Golden cases** (add to `tests/content_download/test_urlcanon.py`):

   * Case/percent-escape: e.g., `%2a → %2A`, `HTTP://ExAmple.COM → http://example.com`.
   * Dot-segments removal & default `/` path.
   * Default port drop **only** when it matches scheme (443 for https, 80 for http).
   * IDN host to punycode; UTF-8 path preserved/encoded correctly.
   * **Port gotcha** locked: `www.example.com:80/foo` keeps `:80` under `https` default scheme.
   * Landing pages: `utm_*` are removed when `role="landing"`; preserve semantic params via `param_allowlist`.
2. **Dedupe improvement**: run a topic with and without url-normalize; measure fewer duplicate attempts (your summary/metrics & `ManifestUrlIndex` already surface counts).
3. **Cache hit-rate improvement**: confirm a higher `from_cache` ratio once normalization precedes requests-cache (hit/miss counters available on responses).

---

## Rollout steps

1. Land `urls.py` with policy defaults above; add a small map for **domain→param_allowlist** (empty to start).
2. Wire **resolvers → pipeline → networking → download** to call the two canonicalizer functions; delete any local cleanup.
3. Add **manifest schema** column for `canonical_url`; index it; keep `original_url` for audit. Update sinks and resume hydration accordingly.
4. Enable **requests-cache** *(you already planned this)* and confirm cache key stability alongside canonical URLs (request-matching rules documented).
5. Ship with a **feature flag** to bypass filtering for emergency domains (rarely needed, but good insurance).

---

## TL;DR

* Add **one canonicalization module**; call it everywhere a URL appears.
* **Don’t sort or “tidy” query strings**—v2.x deliberately doesn’t. Use optional **filtering only for landing pages**.
* **Default to `https`**, drop **only** scheme-default ports, and lean on the library for RFC-correct path and percent-encoding fixes.

If you want, I can follow up with a 1-page “policy doc” to drop into `urls.py` as module-level docstring so agents know exactly when to set `filter_params` and `default_domain`, plus a migration checklist for the manifest schema.
