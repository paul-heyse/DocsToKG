Perfect—let’s wire **Wayback Machine** fallback in a way that’s clean, testable, and maximally consolidated, using **waybackpy** for snapshot discovery and (optionally) **internetarchive** for full-item lookups. Below is a repo-shaped, junior-dev-friendly plan that tells you *exactly* what to add, where to call it, what to delete, and how to test it.

---

# What “good” looks like when we’re done

* **A single last-chance resolver** (`WaybackResolver`) that runs **only after** other sources fail and:

  1. Attempts **direct PDF snapshot** recovery for previously failed PDF URLs
  2. Otherwise, fetches an **archived landing page** and extracts a PDF from it (reusing your existing HTML→PDF heuristics)
  3. Optionally searches **Archive.org items** as a secondary fallback (stretch goal)

* HTTP is **only** done through your hub (HTTPX + Hishel + Tenacity, with the **pyrate-limiter** policy for `web.archive.org` already in place).

* URL normalization (your `urls.py`) is honored **before** cache/rate-limit/mounts.

* Telemetry records what happened (snapshot chosen, timestamp, content-type verification, size, etc.), and the manifest entry clearly indicates `source=wayback`.

* Resolver order keeps **Wayback last** (it already is).

---

## 0) Dependencies (pip/pyproject)

Add these runtime deps:

* `waybackpy` (CDX/Availability APIs, with helper methods). ([GitHub][1])
* `internetarchive` (optional stretch, Archive.org advanced search + item downloads). ([Internet Archive][2])

*(We will not cache binary PDFs via Hishel; only metadata/HTML/JSON are cacheable.)*

---

## 1) Where the code lives (and what to touch)

### A. `src/DocsToKG/ContentDownload/resolvers/wayback.py`

You already have a Wayback resolver stub that hits the “availability” endpoint. Replace that **requests-based** approach with **waybackpy** plus your hub client (HTTPX) for actual downloads. Keep the public surface (`is_enabled`, `iter_urls`) unchanged for the pipeline. (It’s already registered as the **last** resolver in the default order. )

### B. `src/DocsToKG/ContentDownload/resolvers/base.py`

Re-use your existing **HTML→PDF** discovery helpers on archived landing pages:

* `find_pdf_via_meta`, `find_pdf_via_link`, `find_pdf_via_anchor`.

### C. `src/DocsToKG/ContentDownload/core.py`

Leverage fields from `WorkArtifact`:

* `failed_pdf_urls` (primary input for Wayback) and, when available, `publication_year`, `landing_urls`.

---

## 2) Resolver enablement & inputs

**When should Wayback run?**

* `is_enabled()` should return **True** if:

  * `artifact.failed_pdf_urls` is non-empty (first priority), or
  * (optional) none of the upstream resolvers produced a viable PDF **and** `artifact.landing_urls` is non-empty (we can attempt landing-page recovery as a secondary mode).

Your current `is_enabled` checks `failed_pdf_urls`—keep that as the trigger; you can extend to landing URLs if you want the stretch mode.

---

## 3) Discovery algorithm (CDX-first, Availability as fast-path)

**Why CDX first?** CDX lists *all* snapshots with status code & mime; Availability returns only the “closest” one. Use Availability only as a **quick fast-path** (one request) before CDX, or skip it entirely if you want deterministic selection. (Availability examples & docs: `…/wayback/available?url=…`.) ([archivesupport.zendesk.com][3])

### 3.1 For each `failed_pdf_url` (direct PDF mode)

1. **Optional Availability check**: Query availability for the exact URL. If it returns a snapshot URL, try to **HEAD** it via HTTPX; if `Content-Type` looks like PDF, emit it immediately. (Endpoint details/examples. ) ([archivesupport.zendesk.com][3])
2. **CDX search** using `waybackpy.WaybackMachineCDXServerAPI`:

   * Key inputs: `url` (exact), `start_timestamp` and `end_timestamp` narrowed around the work’s `publication_year` when available (e.g., ±2 years). ([GitHub][1])
   * Iterate `snapshots()` and **prefer** entries with:

     * `statuscode == 200`
     * `mimetype` contains `application/pdf`
     * **closest** to `publication_year` (or **newest** if no year)
       *(waybackpy exposes `snapshots()` iteration and “near” helpers; examples in docs.)* ([GitHub][1])
   * For the top candidate’s `archive_url`, do an HTTPX **HEAD** (role=`artifact`), then **GET** if needed for final confirmation (small sniff); only emit if it’s PDF (or file signature starts with `%PDF-`).

**Why HEAD?** It’s cheap, avoids wasting bandwidth on dead captures, and your download path already knows how to classify PDF/HTML/XML.

### 3.2 If the **landing page** failed (or no PDF snapshots exist)

1. Use Availability or CDX on the **landing URL** (not the PDF):

   * Prefer HTML snapshots: `statuscode == 200` and `mimetype ∈ {text/html, application/xhtml+xml}`. ([GitHub][1])
2. **GET** the archived landing page (role=`landing`), **parse** with BeautifulSoup, then reuse your helpers to extract a PDF link: `find_pdf_via_meta/link/anchor()` (base module).
3. **Canonicalize** the discovered PDF (your `canonical_for_request(role="artifact")`) before testing it. Then **HEAD** the archived PDF to confirm it’s real; if yes, **emit**.

> “Archived URL” format is `https://web.archive.org/web/{timestamp}/{original_url}` (Memento path). We treat it like any other URL through the hub. The **CDX** and **Availability** APIs are the two primary discovery surfaces. ([Internet Archive][2])

---

## 4) Using HTTPX + Hishel + Tenacity correctly

* **Discovery** (Availability/CDX/archived HTML) uses your **cached client** (role=`metadata` for JSON/HTML CDX results; role=`landing` for archived HTML). That means CDX and Availability results will be cached per RFC rules by Hishel; retries/backoff handled by Tenacity; per-host rate limit enforced by your transport.
* **Artifact** (archived PDF) uses your **raw (non-cached) client** with role=`artifact`. (We never cache binary PDFs.)
* **HTTP/2** is OK against `web.archive.org`, but keep mounts/denylist logic ready if you observe throttling under multiplexing.

*(This matches the wiring you just set up for HTTPX/Hishel/Tenacity.)*

---

## 5) Rate-limit and politeness

* Add a **pyrate-limiter** policy for `web.archive.org`:

  * **Metadata/CDX/Availability**: modest caps (e.g., `5/s, 300/min`) with short `max_delay` (100–250 ms)
  * **Artifact PDFs**: stricter per-second cap (e.g., `1–2/s`) and a slightly larger `max_delay` (1–3 s) to smooth bursts
    *(This is exactly the limiter pattern you introduced in networking; the resolver does not sleep—**the hub controls** pacing.)*

---

## 6) Heuristics (how to pick “the right” snapshot)

* **Primary**: PDF snapshot exists → pick the one **closest to publication year**; tie-break on newest afterwards.
* **Secondary**: HTML snapshot exists → parse and emit discovered archive PDF URL.
* **Tertiary (Stretch)**: No snapshot for the exact URL? Try **prefix search** via CDX (URL without query; or canonicalized base) to find derived asset URLs that end with `.pdf`. (CDX supports filtering; waybackpy surfaces snapshot lists we can scan for `archive_url` suffix `.pdf`.) ([GitHub][1])

**Rejects / Skips**:

* `statuscode` not 200 (skip)
* Obvious block pages (robots denied, HTTP 403/451)
* PDFs under N bytes (e.g., < 4 KB) unless header proves it’s a valid `%PDF`

All decisions should produce **ResolverEvent** entries with reason codes for observability (SKIPPED_NO_SNAPSHOT, SKIPPED_HTML_NO_PDF, ERROR_CDX_TIMEOUT, etc.).

---

## 7) Manifest & telemetry

When `WaybackResolver` emits a URL, include **metadata**:

* `{"source": "wayback", "original": <failed_url>, "timestamp": <YYYYMMDDhhmmss>, "statuscode": 200, "mimetype": "application/pdf"}`

Also log **discovery path**: `mode="availability|cdx"`, `landing_parse=True|False`, and `html_snapshot_timestamp` if we parsed HTML.

Your manifest/attempt sinks already exist; just ensure the resolver emits `ResolverResult` with URL or an event, and the pipeline will log attempts accordingly.

---

## 8) Internet Archive Items (optional stretch, off by default)

If CDX fails, you *may* try the **Archive.org advanced search** via `internetarchive.search_items()` with a **well-scoped query** built from artifact metadata (DOI/title/creator/publisher) and ask for a small set of **fields** (identifier, mediatype, files count, year). Then:

* For each candidate **Item** (`get_item(identifier)`), list files and look for `.pdf` with sane size and plausible publication year; download **via item file URL** (not Wayback). ([Internet Archive][2])
* Query construction uses the Archive.org advanced search syntax (Lucene-like). Combine fields such as `title:"<title>" AND mediatype:texts AND year:<YYYY>`; if you carry DOI in metadata, try a literal `doi:"<doi>"` or `subject:"doi <doi>"` (fields vary by collection; *expect misses*). ([archivesupport.zendesk.com][4])

> Keep this behind a config toggle (`--wayback-item-search`) because collections vary in quality and the search space is wide.

---

## 9) Configuration surface (small, explicit)

In `ResolverConfig` (or your CLI mapping), add:

* `wayback_enabled: bool = True`
* `wayback_availability_first: bool = True` (fast-path check)
* `wayback_year_window: int = 2` (search ± this window around `publication_year`)
* `wayback_max_snapshots: int = 8` (cap CDX scan)
* `wayback_html_parse: bool = True` (allow landing-page parse)
* `wayback_item_search: bool = False` (stretch)
* `wayback_min_pdf_bytes: int = 4096` (sanity)
* `wayback_host_rps: float` (exposed via networking, not resolver)

---

## 10) Call flow and roles (very important)

* Each HTTP call must set the **role** in `request.extensions` so the hub picks the right client and limiter:

  * Availability/CDX JSON → `role="metadata"`
  * Archived HTML → `role="landing"`
  * Archived PDF → `role="artifact"`

This ensures **Hishel** caches only JSON/HTML requests, **never PDFs**, and **pyrate-limiter** enforces per-role quotas.

---

## 11) Tests you should add

Create `tests/resolvers/test_wayback.py` with deterministic fakes (or stub out waybackpy):

1. **Availability fast-path**

   * Simulate available JSON → `closest.url` points to a PDF → resolver emits that URL with timestamp/status. ([archivesupport.zendesk.com][3])

2. **CDX PDF selection**

   * Fake `snapshots()` with mixed `mimetype`/`statuscode`; pick the 200+PDF closest to `publication_year`. ([GitHub][1])

3. **CDX HTML → parse → PDF**

   * Provide an archived HTML body containing `<meta name="citation_pdf_url">` and `<a href="paper.pdf">` and assert the resolver returns the PDF URL discovered via your base helpers.

4. **No hits**

   * Availability empty AND CDX empty → emit `SKIPPED_NO_SNAPSHOT` event.

5. **Rate-limit awareness**

   * Ensure metadata calls do **not** consume artifact tokens and vice-versa (hook the limiter counters in the hub to assert role separation).

6. **URL normalization**

   * Assert the archived PDF request uses your canonicalization before send (stable keys for cache/limiter).

7. **Size/Type sanity**

   * HEAD `Content-Type` is not PDF OR small body → skip + emit reason.

---

## 12) Telemetry / Observability (what to log)

* Per attempt: `resolver="wayback"`, `mode="availability|cdx"`, `snapshot_ts`, `statuscode`, `mimetype`, `from_cache` (Hishel), `rate_delay_ms` (limiter), `revalidated=True|False`.
* Per run summary: counts of `wayback_candidates`, `wayback_emitted`, `wayback_from_html`, `wayback_timeouts`, `wayback_no_snapshot`.

---

## 13) What to **delete** / simplify

* The current `requests` call in Wayback resolver that hits only the **availability** endpoint repeatedly—replace with waybackpy + hub HTTPX for actual fetches.
* Any local JSON parsing/URL building for Wayback beyond waybackpy’s helpers (we’ll still build the Memento URL but typically receive it from waybackpy/availability).

---

## 14) Stretch “best possible” implementation ideas

* **CDX filter refinement**: When scanning, bias toward `mimetype:application/pdf`, then fall back to HTML. (CDX filtering parameters exist; waybackpy exposes `snapshots()` with fields we can filter client-side.) ([GitHub][1])
* **Memento “near” chooser**: If publication year exists, use waybackpy’s `near(...)` to select the closest capture to that date for both PDF and HTML. ([GitHub][1])
* **Offline mode**: If the run is in `--offline` (Hishel `only-if-cached`), ensure the resolver **does not** hit the network and instead emits a clear SKIPPED event with `offline=true`.
* **Backfill save (deferred)**: Consider wiring SavePageNow for later, not in the resolver path (can create operational/legal noise). Prefer read-only recovery.

---

## 15) Rollout checklist (single PR)

1. Replace `resolvers/wayback.py` internals with the **CDX-first** algorithm above; keep `name="wayback"` and existing resolver interface.
2. Add config flags (Section 9) to `ResolverConfig` and surface minimal CLI toggles.
3. Ensure all HTTP calls set **role** and go through the hub clients.
4. Add limiter policy for `web.archive.org` (metadata vs artifact).
5. Add the **tests** in Section 11.
6. Update docs (“last-chance resolver; CDX→HTML parse→PDF; optional Archive.org items”).
7. Verify **DEFAULT_RESOLVER_ORDER** still ends with `wayback`.

---

### References you’ll lean on

* Wayback **Availability API** (returns the “closest” archived snapshot URL). ([archivesupport.zendesk.com][3])
* Wayback **CDX Server API** (enumerate snapshots with status & mimetype; surfaced via **waybackpy**). ([GitHub][1])
* Wayback **Memento** URL shape for archived content (`/web/{timestamp}/{url}`). ([Internet Archive][2])
* **internetarchive** Python lib: `search_items()` and `get_item()` for Archive.org items (stretch). ([Internet Archive][2])

---

If you want, I can follow this with a tiny “field mapping” table (what metadata we record per discovery step) and a one-pager spec for the telemetry fields so your dashboards will reflect Wayback efficacy from day one.

[1]: https://github.com/akamhy/waybackpy?utm_source=chatgpt.com "GitHub - akamhy/waybackpy: Wayback Machine API interface & a command-line tool"
[2]: https://internetarchive.readthedocs.io/en/stable/api.html?utm_source=chatgpt.com "Developer Interface — internetarchive 1.8.0 documentation"
[3]: https://archivesupport.zendesk.com/hc/en-us/articles/360001495812-Developer-Resources?utm_source=chatgpt.com "Developer Resources – Internet Archive Help Center"
[4]: https://archivesupport.zendesk.com/hc/en-us/articles/360043648052-Search-Building-powerful-complex-queries?utm_source=chatgpt.com "Search - Building powerful, complex queries – Internet Archive Help Center"
