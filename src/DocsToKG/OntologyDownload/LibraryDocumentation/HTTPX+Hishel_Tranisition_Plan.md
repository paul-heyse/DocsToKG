Below is a **granular, narrative implementation plan** to **maximize consolidation under HTTPX + Hishel** inside `src/DocsToKG/OntologyDownload`, so we can delete bespoke networking/caching and keep public behavior stable. I anchored each step to today’s module layout and call-sites so an agent can implement straight through.

---

# North-star outcome (what “done” looks like)

* A **single, long-lived `httpx.Client`** (or `AsyncClient` if you later go all-async) powers *all* HTTP operations in OntologyDownload. It’s configured once with explicit **timeouts**, **pool limits**, **HTTP/2**, **SSL**, and optional **proxy mounts**; reused across planning, HEAD/GET probes, and downloads. HTTPX is intentionally strict (timeouts, redirects off by default); we keep that for clarity.
* The client is decorated with a **Hishel RFC-9111 cache** (disk-backed under your `CACHE_DIR/http/`) so **ETag / Last-Modified** conditionals and **304** handling are automatic. We remove bespoke conditional/ETag plumbing and most HEAD preflights.
* All **file downloads** stream from HTTPX directly to the final destination; we **drop the `pooch.HTTPDownloader` subclass** and its session pool, and we stop double-copying (cache→dest) to cut IO in half. Status mapping and manifest fields remain unchanged.
* **URL security** stays centralized (deny traversal, enforce allow-listed registrable domains/IDN), but the *callers* don’t know how it’s implemented; they just call `validate_url_security` before network I/O.
* **Media-type checks** remain (RDF aliases/labels), but they now read `Response.headers["Content-Type"]` from HTTPX/Hishel results.
* **Public types stay stable** (`DownloadResult`, `DownloadFailure`, etc.). Callers in `pipeline.py` and `ontology_download.py` continue to receive the same fields (path/status/sha256/etag/last_modified/content_type/content_length).

---

# Snapshot of where to operate (today’s code paths)

* You currently have downloader/planner logic spread across **`io/network.py`** (session pool, downloader, URL checks), **`io/__init__.py`** (aggregate exports), plus **`download.py` / `net`** references during the ongoing re-org. We will replace these with an HTTPX+Hishel stack, keeping the same exports.
* `StreamingDownloader` and bespoke pool/HEAD logic live under `io/network.py` and are invoked from `download_stream` and planner probes; we will retire these in favor of HTTPX streaming + cache semantics.
* `ontology_download.py` & `pipeline.py` already import from a `net`/`io_safe` surface; we will keep those import sites stable while changing implementation behind them.
* RDF MIME alias/label consolidation exists and should continue to be used to validate server responses.

---

# Phase 0 — Prepare the ground (small PR, no behavior change)

1. **Add dependencies & pins.** Add `httpx[http2]` and `hishel` to your project. (Leave `requests` installed for the moment to avoid breaking other subpackages mid-refactor.)
2. **Create a new module boundary for the transport** (e.g., `OntologyDownload/net.py` or consolidate under the existing `net` package you’re already importing). Keep public exports identical to what `io/__init__.py` currently exposes for networking: `download_stream`, `DownloadResult`, `validate_url_security`, RDF alias sets, etc. Callers keep their imports.
3. **Introduce a feature flag** inside `settings/config` like `network.engine = "httpx"` (default: httpx). This gives you an easy rollback if needed while tests are updated. (No public CLI change yet.)

**Acceptance**: Build passes; nothing calls HTTPX yet.

---

# Phase 1 — Install a single HTTPX + Hishel client (no call-site changes yet)

Goal: centralize client creation and cache semantics in one place.

1. **Client factory**. In `net.py`, create a *single* long-lived client (module-level or managed singleton). Configure:

   * **Timeouts**: explicit connect/read/write/pool phases that reflect your SLOs (HTTPX defaults are strict; keep this clarity).
   * **Pool limits**: set `max_connections`, `max_keepalive_connections`, and `keepalive_expiry` for your workload.
   * **HTTP/2**: enable for concurrency benefits (esp. async later).
   * **Redirects**: leave **off** globally; explicitly follow only where your resolvers expect it. (HTTPX defaults to off.)
   * **SSL**: use a proper OS/Certifi trust store via an `SSLContext`. (This also aligns with HTTPX’s 0.28 deprecations around string `verify=`.)
   * **Proxy routing**: if you require per-scheme/per-host routing, mount transports accordingly; otherwise respect `HTTP(S)_PROXY/NO_PROXY` via `trust_env=True`.
2. **Attach Hishel caching** at the client layer using a disk store under `CACHE_DIR/http`. Make GET/HEAD cacheable; ensure revalidation so `If-None-Match/If-Modified-Since` are applied automatically. Remove any bespoke ETag/Last-Modified logic in the *client* layer; we’ll keep only **result interpretation** at higher layers.
3. **Event hooks**: install request/response hooks that:

   * Annotate telemetry (method, URL, service/host tags, correlation id).
   * Enforce `raise_for_status()` universally so you map errors in one place.
   * Stamp a consistent **User-Agent** and any resolver-requested headers. (Your code already tracks service in the adapter; preserve that behavior.)

**Acceptance**: You can unit-construct the client; no production path calls it yet.

---

# Phase 2 — Replace the bespoke **session pool** with the shared HTTPX client

Goal: delete `SESSION_POOL` and any per-host/service `requests.Session` leasing in `io/network.py`.

1. **Identify all `SESSION_POOL.lease(...)` call-sites** (planner probes, downloader). Replace them with direct use of the single HTTPX client. This removes stack allocation, pool maps, and factory indirection in `lease`.
2. **Remove the pool implementation** and its registry. Keep a thin indirection (e.g., `get_client()`) so tests can swap a `MockTransport`.
3. **Keep rate limits intact for now** (your `TokenBucket`/registry) since they’re orthogonal; they remain at the *call* boundary (before requests). We’ll revisit once network is stable.

**Acceptance**: Existing planner/download unit tests run against the HTTPX client via a drop-in `get_client()`.

---

# Phase 3 — Retire `StreamingDownloader` (pooch) in favor of HTTPX streaming + Hishel

Goal: delete the custom `pooch.HTTPDownloader` subclass and its resume/HEAD dance.

1. **Map behaviors that must persist** from `StreamingDownloader`:

   * **Max size guard** (GB cap) → enforce by counting bytes while streaming; raise your existing `PolicyError` when exceeded. (You already distinguish Policy vs. filesystem failures.)
   * **Progress logging** → keep (emit events every ~10% or similar, but do it in response iter loop, not via pooch callbacks).
   * **Media type validation** → compare server `Content-Type` against `RDF_MIME_ALIASES`/labels *after* GET, not via a separate HEAD. (Use HEAD only when servers are known to behave.)
   * **ETag/Last-Modified** capture → take from `Response.headers`; on **304**, emit `status="cached"` and short-circuit.
   * **Manifest fields** → continue to populate `content_type`, `content_length`, `etag`, `last_modified` on `DownloadResult` consistently for both cache hits and fresh downloads.
2. **Reimplement `download_stream(...)`** to:

   * Run `validate_url_security` up front (same API/semantics; the implementation may live in `io_safe`/`net`, but callers are unchanged).
   * If rate-limited, **consume a token** before starting I/O (leave your registry in place for now).
   * Issue a **single GET** (prefer avoiding HEAD unless configured) using the shared client; if Hishel returns a 304 on revalidation, construct a `DownloadResult(status="cached")`. Otherwise stream chunks to a temp file, atomically move into place, **then** compute `sha256_file` as you do now. (You already centralize atomic write and hashing helpers in `io_safe`.)
   * Map HTTP errors to `DownloadFailure` and size violations to `PolicyError`. Keep your existing exception taxonomy so upstream doesn’t change.
3. **Remove pooch dependency** from OntologyDownload (keep if used elsewhere; otherwise drop). Update tests that referenced pooch stubs to use HTTPX `MockTransport` fixtures.

**Acceptance**: All existing `download_stream_*` tests pass with HTTPX; cache hits return `status="cached"` with correct manifest telemetry; no regressions in `ontology_download.py` call-paths.

---

# Phase 4 — Consolidate planner probes on HTTPX (+ cache awareness)

Goal: planner’s “probe” logic uses the same client and benefits from cache and strict timeouts.

1. **Replace planner’s `requests` use** (HEAD/GET) with the shared HTTPX client. Keep retry budget decisions at this layer if you have them; otherwise rely on explicit timeouts and, if you must, the lightweight `retries=` at the HTTPX transport level for *connect* errors only. (HTTPX intentionally keeps richer retries out of core.)
2. **Stop bespoke conditional logic**—let Hishel handle validators and 304; only keep planner-level **policy checks** (e.g., max size, allowed media types) and **domain allow-list**.
3. **Persist the same plan metadata fields** you already compute (`_populate_plan_metadata` etc.); their *source* is now the HTTPX+Hishel response rather than a HEAD call. (You already centralized those parsers.)

**Acceptance**: `plan`/`pull --dry-run` produce identical or better metadata without extra network round-trips.

---

# Phase 5 — Remove legacy pool & downloader code, keep the same public exports

Goal: delete bespoke layers, keep import surfaces intact.

1. **Delete** or mark deprecated: `SESSION_POOL`, `StreamingDownloader`, and any `requests.Session` factories in `io/network.py`. Keep thin shims that forward to HTTPX for one release if you want a soft transition; otherwise remove directly.
2. **Keep `__all__` exports** stable. Your `io/__init__.py` aggregator currently re-exports downloader bits; update it to re-export from the HTTPX implementation (or collapse the aggregator and point callers at `net`). Validate that `ontology_download.py` still imports the same names.
3. **Prune duplicate helpers** migrated into `io_safe` (e.g., `extract_archive_safe`, `sha256_file`, url validation) so there is a **single** source. You’ve already started moving these.

**Acceptance**: `grep` shows no `requests.Session` use in OntologyDownload; `pooch` is no longer imported; public API remains.

---

# Phase 6 — Tests & fixtures on HTTPX semantics

Goal: port tests from bespoke downloader to HTTPX idioms and cache.

1. **Replace fake `requests` servers** with HTTPX **`MockTransport`** test doubles; assert:

   * 304 pathways (revalidation via Hishel) → `status="cached"`.
   * Content-type mismatches surface the existing log & warning path but still allow RFC-equivalent RDF formats per your alias map.
   * Max size violations raise `PolicyError`.
2. **Planner probe tests**: assert no redundant HEAD calls when a GET suffices; keep a single GET with cache validation where safe.
3. **CLI output**: your `cli` tests already assert JSON/table fields; ensure fields are unchanged with the new path (e.g., `content_type`, `content_length` present on results).

**Acceptance**: All OntologyDownload tests pass *without* monkey-patching `requests` pools or pooch; coverage on cache and error branches is maintained.

---

# Phase 7 — Clean-ups & follow-ons (optional, after green)

1. **Rate-limits**: either keep your `TokenBucket` registry or swap to a library later; it’s orthogonal to HTTPX/Hishel consolidation and can be addressed separately if desired. (Not required for “max consolidation under HTTPX/Hishel”.)
2. **Retry policy**: leave **transport-level connect retries** at small values; if you need status-code–aware backoff (429/5xx) add it *once* at the call boundary rather than re-implementing per function. (HTTPX intentionally recommends separate retry layers for rich policies.)
3. **HEAD usage**: keep a host allow-list for pre-flight HEAD only when a provider is known to return accurate size/type; otherwise prefer a single GET with early bailouts.
4. **Docs**: update the OntologyDownload README and API docs to reflect HTTPX client + Hishel cache as the **authoritative** transport/caching layer (you’re already curating module docs heavily).

---

## File-by-file actions (explicit edits the agent should make)

* **`src/DocsToKG/OntologyDownload/net.py` (new or replace existing `net`)**

  * Build the **shared HTTPX client** (timeouts, limits, HTTP/2, SSL, optional mounts).
  * Attach **Hishel cache** (disk store under `CACHE_DIR/http/`).
  * Implement **`download_stream(...)`** (streaming GET, atomic write, manifest fields, 304→cached).
  * Keep/forward **`validate_url_security`** to the existing implementation in `io_safe` to avoid duplication.
  * Re-export **`DownloadResult`, `DownloadFailure`, `RDF_MIME_ALIASES`, `RDF_MIME_FORMAT_LABELS`** so current imports remain valid.

* **`src/DocsToKG/OntologyDownload/io/network.py`**

  * **Remove** `SESSION_POOL` and the `lease()` API; callers will use the shared HTTPX client.
  * **Remove** `StreamingDownloader`; migrate progress/mime/size logic to `net.py`.
  * Keep tiny shims for one release if you want a soft deprecation; logs should warn when shims are used.

* **`src/DocsToKG/OntologyDownload/io/__init__.py`**

  * Update exports so `download_stream` et al. point to the new `net` implementation (or delete the aggregator if you standardize on `net`).

* **`src/DocsToKG/OntologyDownload/pipeline.py` & `ontology_download.py`**

  * No signature changes; ensure they import from the stable surface (`net` / `io_safe`).
  * Where you previously captured `etag/last_modified/content_type/length`, now read them from HTTPX responses or the 304 pathway (unchanged call sites).

* **`tests/ontology_download/*`**

  * Replace pooch/request mocks with **HTTPX `MockTransport`** fixtures.
  * Preserve tests for: retryable/non-retryable HTTP errors, rate-limit token consumption, size guard, RDF content-type alias acceptance, manifest field population, and cache hits. (You already have tests around `download_stream_*`, zip/tar extraction, and CLI outputs; retain them.)

---

## Behavioral decisions (explicit, so there’s no ambiguity)

* **Redirects**: remain **off** globally; enable per request only when a resolver expects it. This matches HTTPX defaults and makes provenance clearer.
* **Conditional requests**: defer to **Hishel**. Do not manually set `If-None-Match`/`If-Modified-Since`; only interpret 304 at the result layer.
* **Resume/Range**: we **drop custom resume** for now to reduce bespoke surface; large downloads still benefit from streaming. If resume is required later, we can add a tiny Range helper behind one function. (Neatness over niche feature, per your direction.)
* **Media types**: keep your **`RDF_MIME_ALIASES`** set and **label map**; accept equivalent RDF formats with warnings; reject others unless explicitly allowed.
* **Policy vs. network errors**: file size and allow-list violations remain **Policy** failures; socket/TLS/status errors map to **DownloadFailure** with `retryable` set per your current rules. (You already separate these.)

---

## “Definition of done” checklist (agent can tick these)

* [ ] `grep -R "requests.Session" src/DocsToKG/OntologyDownload` → **0 matches**
* [ ] `grep -R "pooch.HTTPDownloader" src/DocsToKG/OntologyDownload` → **0 matches**
* [ ] A single `httpx.Client` with **Hishel** is constructed once and reused across planner + downloader.
* [ ] All **existing public exports** retained (e.g., `DownloadResult`, `download_stream`, RDF alias constants).
* [ ] `download_stream(...)` still returns the same shape; `status ∈ {"fresh","cached"}`; manifest fields populated consistently (etag/last_modified/content_type/content_length).
* [ ] Planner probes use the same client; unnecessary HEADs removed.
* [ ] Tests validate **304** cache hits, **size guard**, **RDF alias acceptance**, and **CLI JSON/table** remain stable.

---

## Why HTTPX + Hishel for this job (for reviewers)

* HTTPX gives us *first-class* **client lifecycle**, **timeouts per phase**, **pool sizing**, **HTTP/2**, and **event hooks** in one place; it’s strong on correctness and explicit control. Redirects default off; timeouts are strict; and transports/mounts let you route by scheme/host when needed.
* Hishel adds **standards-compliant, validator-aware caching** to HTTPX (RFC 9111), handling `ETag`/`Last-Modified`, age, freshness, and revalidation so we can delete homemade 304 plumbing and most HEADs.

---

If you want, I can turn this into a mini series of PRs (Phase 0→5) with exact file diffs per step. For now, the plan above gives your agent everything needed to implement end-to-end.

**Sources (project-specific anchors):**

* Aggregated IO & downloader surface (exports).
* Session pool and downloader call-sites slated for removal.
* Newer `net`/`io_safe` import surface in `ontology_download.py`/`pipeline.py`.
* RDF MIME alias/label mapping to preserve behavior.
* CLI/tests expectations for outputs and errors.

**References (attached library briefs):**
HTTPX quick-ref & behaviors (timeouts, limits, HTTP/2, redirects, SSL, transports, exceptions).
Hishel RFC-style HTTP caching on top of HTTPX.
