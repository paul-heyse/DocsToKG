We’re going for **maximal consolidation into Tenacity** (and we don’t need to preserve old semantics), here’s the concrete plan—module-by-module—so you can replace the custom backoff logic with Tenacity’s primitives everywhere that matters.

---

# The target shape (after refactor)

* **One place** defines all retry behavior: the `DocsToKG.ContentDownload.networking` module (this is already your consolidated “HTTP session, retry, and conditional request helpers” surface). It will expose a **single Tenacity policy** (or a tiny family of them) and a **single thin entrypoint** to run a request under that policy. Your own loops/jitter/sleeps go away.

* **Everything else** (CLI/download path, resolvers, head-precheck) must call that entrypoint; no local sleeps or retry loops remain in those call sites. Your docs already frame `networking` as canonical; keep it that way.

---

# What Tenacity building blocks to use (and why)

* **Retry predicate:** `retry_if_exception_type` for `requests` network exceptions **OR** `retry_if_result` for HTTP responses whose `.status_code` is in your retryable set (e.g., {429, 500, 502, 503, 504}). This replaces every ad-hoc “if status in … then sleep/backoff” branch you have in networking & resolvers.

* **Stop policy:** `stop_after_attempt(N)` **combined with** `stop_after_delay(T)` so you can express both “budget of attempts” and “budget of time” in one place.

* **Wait policy:** `wait_random_exponential(multiplier=…, max=…)` (jittered exponential). To keep politeness, **prefer `Retry-After`** when present: you’ll plug it in via a small Tenacity `wait_base` subclass (a few lines) that checks the last response and returns its header value; otherwise it defers to the exponential policy. (This is the only unavoidable “glue”: Tenacity doesn’t parse `Retry-After`; you already have a robust parser to reuse.)

* **Sleep hook + logging:** Use `sleep=time.sleep` so your existing tests that monkeypatch `time.sleep` still work, and `before_sleep_log(LOGGER, DEBUG)` for consistent telemetry across all callers.

You already centralised this surface under `networking` and documented the public entrypoints (`create_session`, `request_with_retries`, `ConditionalRequestHelper`). We’ll keep the same names, but the implementation inside becomes pure-Tenacity.

---

# Exact edits (by module)

## 1) `src/DocsToKG/ContentDownload/networking.py`  ➜  **Become the Tenacity hub**

**What to do**

* **Replace** the custom backoff loop inside `request_with_retries(…)` with a Tenacity `Retrying` policy (and delete local jitter/sleep helpers). Keep only a **thin wrapper** that:

  * Builds the Tenacity `Retrying` object from function args (attempts, total time, backoff multiplier/max, retryable statuses),
  * Invokes the underlying `session.request(…)`,
  * Returns the final `requests.Response`.

* **Honor `Retry-After`**: implement a tiny Tenacity `wait` strategy that checks the last response and returns the parsed header seconds when present; otherwise defer to the exponential jitter wait. Reuse your existing header parser (it already supports both integer and HTTP-date forms).

* **Leave session pooling intact** in `create_session(…)` (adapters with `max_retries=0`) so retries don’t double-stack; your docs and tests already point to this arrangement after the consolidation.

* **Keep** `ConditionalRequestHelper` as-is; it is orthogonal to retries and already lives here after consolidation.

**Why here?** This file is the canonical networking surface (your Sphinx/API pages and tests have converged on it). You’ve already migrated imports toward `networking` and re-exported from `network` only as a shim. Make `networking` authoritative for Tenacity policies.

---

## 2) `src/DocsToKG/ContentDownload/cli.py` (the former `download_pyalex_pdfs.py`)  ➜  **Delete local loops, let Tenacity drive**

**What to do**

* In the **download path** (`download_candidate(…)` and helpers), remove any while/for retry cycles and `time.sleep` calls. Every place that touches HTTP must call `networking.request_with_retries(…)` and **return on first non-retryable result**. Your recent refactors already moved conditional caching and result shaping out, so this change doesn’t conflict with ETag/`304` handling.

* Keep range-resume and streaming logic as-is; Tenacity will simply re-invoke the GET when it decides another attempt is warranted (e.g., `429/503`). (Your current design holds off on append-mode resume by default; that decision remains independent of retries.)

* Ensure the local **HEAD preflight** path (when enabled) also calls the Tenacity entrypoint (no direct `session.head`). This is already the case in tests; with Tenacity in `networking`, no additional loops are needed here.

**Why here?** This is where lingering one-off sleeps tend to hide. The goal is: **no sleeps, no loops** in download code; only Tenacity orchestrates attempts.

---

## 3) `src/DocsToKG/ContentDownload/pipeline.py` (resolver orchestration)  ➜  **Enforce call-through only**

**What to do**

* In resolver execution paths that probe URLs (e.g., `_attempt_url`, head-precheck helpers), **remove** any inline backoff/sleep and call `networking.request_with_retries(…)` exclusively. Your consolidation already routes most of these through networking; this step is mainly a **code deletion / cleanup** pass.

* Keep your breaker/token-bucket integrations for now; Tenacity handles *when to try again*, the breaker/limiter decides *whether we’re allowed to try*. (You’ll swap those to `pybreaker` / `pyrate-limiter` in later steps.)

**Why here?** The pipeline previously contained small “sleep then requeue” branches for retries. Those go away entirely once Tenacity governs the timing.

---

## 4) `src/DocsToKG/ContentDownload/resolvers/providers/*`  ➜  **No local retries anywhere**

**What to do**

* Replace any lingering `session.get` / `session.head` calls in providers with the central Tenacity entrypoint. The open-spec notes show you already did this for Crossref and others; now make it universal across the provider folder so **zero** provider implements its own sleep/retry.

**Why here?** Providers should be pure request factories + result interpreters. Timing belongs to Tenacity.

---

## 5) Tests & docs  ➜  **Point everything at `networking`**

**What to do**

* Tests that patch `DocsToKG.ContentDownload.networking.time.sleep` (to measure or freeze backoff) should continue to pass because we’ll hand Tenacity that sleep function. Your tree already moved most imports from the legacy `network` shim to `networking`; make sure any stragglers are updated.

* Your docs already state that `networking` is the consolidated reference; keep the Sphinx pages pointing there.

---

# Tenacity policy knobs (map your existing flags → Tenacity)

* **Attempts:** map your former `max_retries` to `stop_after_attempt(max_retries + 1)`. (Attempts = initial try + retries.)

* **Wall time:** expose an optional `--max-retry-seconds` to set `stop_after_delay(total_seconds)` (you have inline time budgeting comments in code; this turns it into a first-class control).

* **Retryable statuses:** keep your existing default (e.g., `{429, 500, 502, 503, 504}`) in one constant in `networking` and make it configurable per resolver type if you need it later.

* **Backoff shape:** map your existing `backoff_factor/backoff_max` into Tenacity’s `wait_random_exponential(multiplier=<factor>, max=<max>)`. No local jitter calls remain.

* **Header-aware wait:** wire `Retry-After` preference through the custom `wait` implementation that consults your existing `parse_retry_after_header`. That small parser is the *only* non-Tenacity component we keep; everything else becomes Tenacity config.

---

# What you can delete after this

* All per-callsite sleep/backoff loops in **CLI/download**, **pipeline**, and **providers** (keep only the Tenacity call). The recent consolidation already reduced a lot of this, so this pass is mostly deleting dead branches that tests still stub.

* Any helper that computes “exponential delay with jitter”; Tenacity owns that now.

* Any adapter-level retries (ensure adapters use `max_retries=0`); your own docs/tests already moved in this direction.

---

# Acceptance checklist (quick to verify)

1. **Only one** implementation of retries remains (in `networking.request_with_retries`).
   *Search confirms centralisation and the `networking` logger is the canonical one.*

2. **Providers** never call `session.get/head` directly—every call goes through `request_with_retries`.
   *Crossref migration is already documented; repeat for all providers.*

3. **HEAD precheck** goes through the same Tenacity entrypoint; no local sleeps.
   *Tests already monkeypatch `networking.request_with_retries` in head-precheck paths.*

4. **Docs & tests** import from `networking`, not `network`.
   *Your Sphinx “Module: networking” page is in place.*

---

## Where to touch (short list)

* **Authoritative refactor:** `src/DocsToKG/ContentDownload/networking.py` (replace retry loop with Tenacity; keep parser + head-precheck).

* **Callers that must stop sleeping:**

  * `src/DocsToKG/ContentDownload/cli.py` (download path)
  * `src/DocsToKG/ContentDownload/pipeline.py` (resolver orchestration)
  * `src/DocsToKG/ContentDownload/resolvers/providers/*` (all provider HTTP calls)

* **Tests that assume old jitter/sleep:** update to patch Tenacity via the `networking` surface (they mostly already do).

---

### Bottom line

* Put **all** retry/backoff semantics into **Tenacity** inside `networking`.
* Treat **every HTTP attempt** (HEAD/GET) as “one Tenacity call” from the rest of the code.
* Keep only the small **`Retry-After` parser** as glue; everything else is Tenacity configuration.

This gets you the maximal deletion of custom backoff code with the least surface area change—and it cleanly aligns with how your repo has already consolidated the networking surface.
