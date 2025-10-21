Here’s a plain-English, component-by-component explanation of everything we just outlined for PR #5, so you can sanity-check that our mental models match.

---

# Telemetry layer

**What it is:** A thin, explicit interface for recording every network “attempt” and the final per-artifact manifest row—no globals, no hidden side effects.

**Key pieces**

* **`AttemptRecord`** (from P1): the canonical event for *one* decision or network action. Stable fields: `ts`, `run_id`, `resolver`, `url`, `verb`, `status`, `http_status`, `content_type`, `elapsed_ms`, `bytes_written`, `content_length_hdr`, `reason`, `extra`.
* **`CsvAttemptSink`**: appends attempts to a CSV with a fixed header. Advantages: grep-able, diff-able, cheap.
* **`JsonlManifestSink`**: appends one JSON object per completed artifact (“final outcome”). Advantages: schematizable, round-trippable into analytics.
* **`RunTelemetry`**: the façade the pipeline calls. It has two jobs:

  * `log_attempt(**kwargs)` → materialize an `AttemptRecord` and write it.
  * `record_pipeline_result(...)` → normalize outcome + context into a manifest row.

**How it behaves**

* **Best-effort**: telemetry errors must never crash the pipeline.
* **Stable tokens**: we treat `status` and `reason` strings as a public contract (they appear in CSV/JSONL, scripts rely on them). We only *add* tokens over time.
* **Config-aware**: paths come from config; sinks are swappable later (e.g., OTLP, Prometheus push).

**Why it exists:** It gives us confident post-hoc answers to “what happened?” (latency by resolver, error distributions, bytes downloaded, policy skips, etc.) without cluttering execution code.

---

# HTTP base session (`requests.Session`)

**What it is:** A single process-wide connection pool with good defaults.

**Key behaviors**

* **Headers**: sets a sane `User-Agent` and folds in a `mailto` (some providers require it).
* **Connection pooling**: mounts an `HTTPAdapter` with tuned `pool_connections/pool_maxsize` (defaults we picked are safe for most hosts; we can surface them later if needed).
* **TLS/Proxies**: toggled from config (`verify_tls` and `proxies`). We leave cert path/SSL context customization out of scope for now, but the seam exists.

**Why one session:** Reusing TCP/TLS connections lowers latency, avoids socket churn, and reduces TLS handshake load on providers.

---

# Per-resolver HTTP client (rate-limit + retry)

**What it is:** A tiny wrapper around the shared `Session` that applies **per-resolver** rate limits and retry/backoff, and emits telemetry for sleeps/retries.

**Policies it enforces**

* **Rate limit** (`TokenBucket`): capacity, refill rate, and small burst. Before each request it either proceeds or sleeps exactly long enough to regain a token; we log the sleep as `status="retry"` with `reason="backoff"`.
* **Retry/backoff**:

  * Retries only on configured statuses (e.g., 429/500/502/503/504) or network exceptions.
  * Honors `Retry-After` if present (overrides exponential backoff for that attempt).
  * Exponential backoff with jitter—bounded between `base_delay_ms` and `max_delay_ms`.
  * Emits `status="retry"` on each retry, with `reason="retry-after"`, `"backoff"`, or `"conn-error"`, and includes the sleep duration in `elapsed_ms`.

**Interface**

* Looks like `requests.Session` for the operations we use (`head`, `get`), so the execution code doesn’t care which client it receives.
* Merges default timeouts from config unless explicitly overridden at call-site.

**Why per-resolver:** Different upstreams have different politeness requirements and error patterns; keeping policy at the resolver granularity makes tuning safe and explicit.

---

# Resolver registry & resolvers (context in bootstrap)

**What it is:** A registry that materializes concrete resolver instances *in the configured order*, each optionally constructed from its section of the config.

**What we expect of resolvers**

* **Pure resolution**: they compute one or more `DownloadPlan`s—*they don’t fetch the content*. That separation preserves reuse and lets the shared HTTP layer enforce policies uniformly.
* **Predictable return**: a `ResolverResult` with `plans: Sequence[DownloadPlan]`. Zero plans is valid; the pipeline moves on.

**Why this shape:** Keeps resolver logic simple and focused, and lets the bootstrap layer inject shared concerns (HTTP, telemetry, policies).

---

# `bootstrap.run_from_config(cfg)` (the glue)

**What it does (step-by-step)**

1. Builds **telemetry sinks** from `cfg.telemetry` and threads `run_id`.
2. Builds a shared **`requests.Session`** from `cfg.http`.
3. Materializes **resolvers** from the registry using `cfg.resolvers`.
4. For each resolver, constructs a **per-resolver `HttpClient`** preconfigured with its retry/rate policies (and default timeouts from `cfg.http`).
5. Constructs the **`ResolverPipeline`** with:

   * The ordered resolver instances,
   * A **client map** (`resolver_name` → `HttpClient`),
   * Telemetry, `run_id`,
   * Policy knobs (`cfg.robots`, `cfg.download`).
6. Optionally iterates **artifacts** (if provided), letting the pipeline process each and record the manifest; or simply validates/warm-starts the wiring.

**Why a seam here:** The CLI’s `run` needs a single entrypoint that’s trivial to test and evolve; we keep CLI/UI separate from runtime concerns.

---

# Pipeline changes (policy-aware, per-resolver client)

**What it now does**

* **Selects the right client** for each plan via `plan.resolver_name` (falls back to a default if needed).
* Passes **policy knobs** to execution:

  * Robots behavior (if your `prepare_*` consumes it),
  * Download behavior (chunk size, `verify_content_length`, atomic move).
* Maintains **at-most-once success** semantics: as soon as one plan produces a successful outcome, the pipeline returns (and records the final manifest).
* Converts **short-circuit exceptions** into outcomes:

  * `SkipDownload(reason)` → `classification="skip"`,
  * `DownloadError(reason)` → `classification="error"`.

**Why here:** The pipeline is the nexus where plans become downloads; it’s the right layer to pick the client and propagate policies to the execution stage.

---

# Download execution functions (prepare → stream → finalize)

**Responsibilities**

* **`prepare_candidate_download`**: fast, side-effect-free checks that can *skip early* (robots disallow, obvious content-type mismatch based on plan hints, too-large policy via HEAD, etc.). It can:

  * return an adjusted plan, or
  * raise `SkipDownload(reason)`/`DownloadError(reason)`.
* **`stream_candidate_payload`**: actually performs the network transfers with the injected client. It is the **source of truth for attempt telemetry** (HEAD, GET, 304, retries). It writes to a temp file and returns `DownloadStreamResult`.
* **`finalize_candidate_download`**: performs integrity checks (e.g., `Content-Length` match, PDF tail if applicable), **atomically renames** the temp to final, and returns the `DownloadOutcome`.

**Invariants**

* No partial files at final path: on any failure, temp is removed.
* Every network roundtrip should emit **exactly one** attempt record (plus retry attempts as needed).
* The outcome’s `meta` remains small and structured (content type, bytes written, plus a few optional fields).

**Why this split:** It keeps each function small, testable, and single-purpose; it also aligns with the telemetry taxonomy.

---

# CLI artifact ingestion (optional now, easy later)

**What it adds (optionally)**

* `contentdownload run --input file.txt` (or `-` for stdin) to read newline-delimited artifact descriptors (URLs/DOIs/IDs).
* A tiny adapter that maps each line to your `Artifact` representation and hands it to `run_from_config`.

**Why**: Makes local/manual runs easy without baking artifact ingestion into the bootstrap (which stays programmatic).

---

# Tests (what they prove)

**End-to-end smoke**

* Fakes a resolver and HTTP to ensure `run_from_config` builds telemetry, resolves a plan, streams a small body, finalizes successfully, and writes both attempts and a manifest line.
* Asserts that the CSV/JSONL files are created and non-empty.

**Retry/backoff + rate limit**

* Replaces `time.sleep` to capture sleeps without waiting.
* Drives a `429` + `Retry-After` response followed by a `200`, asserts we slept ~1s and the final response is `200`.
* (You can extend the test to assert retry attempt telemetry events were emitted; the seam is there.)

**Why these two:** Together they validate the two most brittle behaviors (bootstrap wiring and polite networking) deterministically.

---

# Acceptance (what “done” looks like)

* A real `run_from_config(cfg)` that does: telemetry → session → per-resolver clients → resolvers → pipeline; works on an artifact iterator.
* Execution functions accept the **policy knobs** and the **per-resolver client**.
* The **attempt CSV** and **manifest JSONL** fill with sensible rows on runs.
* CLI can call `run_from_config(cfg)` and (optionally) read artifacts from a file/stdin.

---

# Guardrails & design tenets

* **One shared session, many policy clients:** reuse connections, vary behavior per resolver safely.
* **No resolver-local HTTP**: resolvers do not fetch content; *all* fetches go through the injected client so rate limits and retries are applied uniformly and telemetry is complete.
* **Telemetry is non-intrusive:** never raises; if a sink fails, the run continues.
* **Explicit DI everywhere:** `telemetry`, `run_id`, `cfg.*` are passed in—no globals, no hidden context.
* **Stable vocabulary:** the set of telemetry `status`/`reason` tokens is a compatibility contract.

---

# Edge cases & how they’re handled

* **Redirect loops / too many redirects**: surfaced by `requests`; counted as a `DownloadError("conn-error")` with a final manifest row (you can extend to a specific reason if you want).
* **No `Content-Length`**: `verify_content_length` skips the equality check—atomic write still prevents partial finals.
* **`206 Partial Content`**: not expected; treat as error unless you add resumable logic later.
* **Robots fetch fails**: treat as “could not determine policy”; we currently default to *allow* or *deny* based on your stance—document your default and log an attempt with a clear reason.
* **Disk full / rename fails**: `finalize_*` must raise `DownloadError`, delete temp, emit a final outcome with `reason="download-error"`.
* **Different filesystems for temp/final**: keep temp in the destination directory so `os.replace` stays atomic.

---

# Why this layering works (and scales)

* **Traceable**: Every decision point emits a compact, structured event; the final row links all attempts via `run_id` and `url`.
* **Tuneable**: Per-resolver policies allow safe experimentation without changing shared behavior.
* **Extensible**: New sinks (OTLP), new resolvers, conditional GETs (ETag/IMS), or parallel artifact processing can be added without rewriting core pieces.
* **Testable**: Each piece has thin seams and pure signatures, enabling small, deterministic tests.

---

If you want me to annotate this with *specific* values from your current `models.py` (e.g., default retry sets per resolver) or walk through one artifact end-to-end with example attempt/manifest lines, I can draft that too.
