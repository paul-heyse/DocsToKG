Awesome—here’s a tight, implementation-ready “what to log” package for the Wayback fallback, split into:

1. a **Field-mapping** (what each record should carry, where it comes from, and how to compute it), and
2. a one-page **Telemetry Spec** you can drop into the repo (e.g., `docs/telemetry/wayback.md`) so junior devs and agents have a single source of truth.

---

# Field mapping (record-by-record)

## A. Correlation & identity (present on **every** record)

| Field          | Type       | Source / How to populate          | Notes                                         |
| -------------- | ---------- | --------------------------------- | --------------------------------------------- |
| `run_id`       | str (uuid) | Provided by runner at start       | Correlates the entire execution window.       |
| `work_id`      | str        | Your Work/Document id             | Whatever you already use (e.g., OpenAlex ID). |
| `artifact_id`  | str        | Stable id for the target artifact | E.g., DOI or your internal artifact key.      |
| `resolver`     | enum       | Constant `"wayback"`              | Distinguishes from other resolvers.           |
| `attempt_id`   | str (uuid) | New uuid per resolver attempt     | Lets you aggregate all sub-events.            |
| `ts`           | RFC3339    | `now()` wall-clock                | Use timezone-aware timestamps.                |
| `monotonic_ms` | int        | `monotonic()` since attempt start | For duration math immune to clock skew.       |

> These fields go in **all** event rows below.

---

## B. Attempt start / end

**Record name:** `wayback_attempt` (one **start** row, one **end** row per attempt)

| Field                | Type | Source / How                                                                                                                               | Notes                                      |
| -------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| `event`              | enum | `"start"` | `"end"`                                                                                                                        | Two rows per attempt.                      |
| `original_url`       | str  | The URL we failed on upstream                                                                                                              | Keep raw input for audit.                  |
| `canonical_url`      | str  | `urls.canonical_for_index()`                                                                                                               | Stable dedupe key.                         |
| `publication_year`   | int? | From artifact metadata                                                                                                                     | Optional; improves ranking.                |
| `mode_selected`      | enum | `"pdf_direct"` | `"html_parse"` | `"none"`                                                                                                 | Which branch ended up used. (`end` only)   |
| `result`             | enum | `"emitted_pdf"` | `"emitted_pdf_from_html"` | `"skipped_no_snapshot"` | `"skipped_non_pdf"` | `"error_http"` | `"error_cdx"` | `"timeout"` | Final outcome. (`end` only)                |
| `total_duration_ms`  | int  | `now() - attempt_start`                                                                                                                    | (`end` only)                               |
| `candidates_scanned` | int  | Counter you keep                                                                                                                           | Sum of CDX entries looked at. (`end` only) |

---

## C. CDX / Availability queries (discovery phase)

**Record name:** `wayback_discovery`

You’ll likely emit 1–3 of these per attempt (e.g., Availability check, then CDX scan).

| Field           | Type | Source / How                             | Notes                                         |
| --------------- | ---- | ---------------------------------------- | --------------------------------------------- |
| `stage`         | enum | `"availability"` | `"cdx"`               | Which discovery interface.                    |
| `query_url`     | str  | Original or landing URL used             | Canonicalize for requests.                    |
| `year_window`   | str  | e.g., `"-2..+2"`                         | The ± window applied around publication year. |
| `limit`         | int  | Config (e.g., 8)                         | Max CDX entries scanned.                      |
| `http_status`   | int  | From HTTPX                               | 200/204/4xx/5xx.                              |
| `from_cache`    | bool | `response.extensions.hishel_from_cache`  | Only for Availability/CDX/HTML.               |
| `revalidated`   | bool | `response.extensions.hishel_revalidated` | Ditto.                                        |
| `rate_delay_ms` | int  | From your limiter hook                   | Delay before request (per role).              |
| `retry_after_s` | int? | If Tenacity honored header               | For 429/503 backoff cases.                    |
| `retry_count`   | int  | From Tenacity                            | Total internal retries made.                  |
| `error`         | str? | Exception class/message                  | On non-HTTP failures.                         |

> For CDX, also log the **count** of entries returned and the **first/last timestamps** present.

---

## D. Snapshot candidate evaluation (selection)

**Record name:** `wayback_candidate`

Emit for the **one** (or two) candidates you actually evaluate (to keep cardinality low). If you want full trace, make it configurable and default to “sampled”.

| Field                  | Type | Source / How                                           | Notes                                   |
| ---------------------- | ---- | ------------------------------------------------------ | --------------------------------------- |
| `archive_url`          | str  | From waybackpy (`archive_url`)                         | Full Memento URL.                       |
| `memento_ts`           | str  | `YYYYMMDDhhmmss`                                       | From CDX row or availability.           |
| `statuscode`           | int  | From CDX row                                           | Pre-check before HEAD.                  |
| `mimetype`             | str  | From CDX row                                           | Often `application/pdf` or `text/html`. |
| `distance_to_pub_year` | int  | `abs(ts.year - publication_year)`                      | If year is known.                       |
| `source_stage`         | enum | `"availability"` | `"cdx"`                             | Where it came from.                     |
| `decision`             | enum | `"head_check"` | `"skipped_status"` | `"skipped_mime"` | What you did next.                      |

---

## E. Archived HTML fetch & parse (html→pdf path)

**Record name:** `wayback_html_parse`

| Field                        | Type | Source / How                     | Notes                 |
| ---------------------------- | ---- | -------------------------------- | --------------------- |
| `archive_html_url`           | str  | Memento URL for landing page     |                       |
| `html_http_status`           | int  | From HTTPX                       |                       |
| `from_cache` / `revalidated` | bool | Hishel extensions                |                       |
| `html_bytes`                 | int  | `Content-Length` or len(body)    | Don’t store the body. |
| `pdf_link_found`             | bool | Result of your parser            | High-level flag only. |
| `pdf_discovery_method`       | enum | `"meta"` | `"link"` | `"anchor"` | Which helper matched. |
| `discovered_pdf_url`         | str? | What you found (canonicalized)   | Null if none.         |

---

## F. Archived PDF verification (final artifact)

**Record name:** `wayback_pdf_check`

| Field              | Type  | Source / How                                    | Notes                       |
| ------------------ | ----- | ----------------------------------------------- | --------------------------- |
| `archive_pdf_url`  | str   | Candidate PDF memento URL                       |                             |
| `head_status`      | int   | HEAD via artifact client                        |                             |
| `content_type`     | str   | From HEAD                                       | Expect `application/pdf`.   |
| `content_length`   | int?  | From HEAD                                       | Optional.                   |
| `is_pdf_signature` | bool? | If you did a small GET sniff                    | True if `%PDF-` magic seen. |
| `min_bytes_pass`   | bool  | `content_length >= min_bytes`                   | E.g., ≥ 4096.               |
| `decision`         | enum  | `"emit"` | `"reject_small"` | `"reject_nonpdf"` | Final gate.                 |

---

## G. Emission (success path)

**Record name:** `wayback_emit`

| Field              | Type | Source / How                    | Notes                        |
| ------------------ | ---- | ------------------------------- | ---------------------------- |
| `emitted_url`      | str  | The URL handed to downloader    | Always the archived PDF URL. |
| `source_mode`      | enum | `"pdf_direct"` | `"html_parse"` | Which path produced it.      |
| `memento_ts`       | str  | The chosen snapshot ts          |                              |
| `http_ct_expected` | str  | `"application/pdf"`             | Hard-coded expectation.      |

---

## H. Skip / error signals (explicit reasons)

**Record name:** `wayback_skip`

| Field     | Type | Value                                                                                                                       |
| --------- | ---- | --------------------------------------------------------------------------------------------------------------------------- |
| `reason`  | enum | `"no_snapshot"` | `"all_non_pdf"` | `"below_min_size"` | `"blocked_offline"` | `"timeout"` | `"cdx_error"` | `"http_error"` |
| `details` | str? | Short, sanitized message                                                                                                    |

---

# One-pager telemetry spec (drop-in)

> **File:** `docs/telemetry/wayback.md` (or `src/DocsToKG/ContentDownload/docs/wayback_telemetry.md`)

## Purpose

Track the effectiveness, latency, and safety of the Wayback last-chance resolver, including discovery pathways (Availability/CDX), HTML-parse fallback, and archived-PDF verification. All events are JSONL rows with a shared correlation envelope and strict, low-cardinality enums.

## Streams

* `wayback_attempt` (start/end)
* `wayback_discovery` (availability, cdx)
* `wayback_candidate` (evaluated candidates; sampled)
* `wayback_html_parse` (if HTML path used)
* `wayback_pdf_check` (HEAD/GET sniff)
* `wayback_emit` (success)
* `wayback_skip` (explicit failure reasons)

> **Naming:** snake_case keys; enums as lowercase strings.
> **Timestamps:** wall clock `ts` (RFC3339) + `monotonic_ms` for durations.
> **PII:** never log HTML bodies; store counts/bytes only.

## Required envelope (on every row)

`run_id`, `work_id`, `artifact_id`, `resolver="wayback"`, `attempt_id`, `ts`, `monotonic_ms`.

## Roles & networking fields

For each HTTP call, the networking hub enriches records with:

* `http_status`, `from_cache`, `revalidated`, `rate_delay_ms`, `retry_after_s`, `retry_count`.

Roles:

* Availability/CDX JSON → `role="metadata"`
* Archived HTML → `role="landing"`
* Archived PDF → `role="artifact"`

## Success criteria (SLOs)

* **Coverage:** ≥ X% of otherwise-failed artifacts get a valid archived PDF.
* **Median selection latency:** ≤ 500 ms for direct PDF snapshots; ≤ 2 s when HTML parse needed.
* **False-positives:** < 1% non-PDF misclassifications (guard via HEAD CT + `%PDF-` sniff + min bytes).

## Derived metrics (from logs)

* **Wayback yield** = `wayback_emit` / `wayback_attempt(end)`
* **Path mix** = share `"pdf_direct"` vs `"html_parse"`
* **Cache assist rate** = % discovery calls with `from_cache=true`
* **Rate-limit smoothing** = p95 `rate_delay_ms` by role
* **Backoff impact** = mean `retry_after_s` on 429/503 responses
* **Quality gates** = reject counts by `reason` (below_min_size, non_pdf, no_snapshot)

## Storage guidance

* **JSONL sinks:** append rows; rotate daily or per-size.
* **SQLite sink (optional):** create narrow tables with required columns only to keep indices small:

  * `wayback_attempts(run_id, work_id, artifact_id, attempt_id, start_ts, end_ts, result, mode_selected, total_duration_ms, candidates_scanned)`
  * `wayback_emits(run_id, artifact_id, attempt_id, emitted_url, memento_ts, source_mode)`
  * `wayback_discovery_sum(run_id, attempt_id, stage, queries, returned, first_ts, last_ts, cache_hits, http_errors)`
* **Indices:** `(run_id)`, `(artifact_id)`, `(attempt_id)`; for perf dashboards also index `(result)`, `(source_mode)`.

## Query examples (SQLite)

* Yield by path:

  ```sql
  SELECT source_mode, COUNT(*)
  FROM wayback_emits
  WHERE run_id = ?
  GROUP BY source_mode;
  ```

* P95 selection latency:

  ```sql
  SELECT percentile_approx(total_duration_ms, 0.95)
  FROM wayback_attempts
  WHERE run_id = ?;
  ```

* Top reasons for skip:

  ```sql
  SELECT reason, COUNT(*)
  FROM wayback_skips
  WHERE run_id = ?
  GROUP BY reason ORDER BY COUNT(*) DESC;
  ```

## Implementation checklist

1. Emit `wayback_attempt(start)` when resolver begins evaluating a failed artifact URL.
2. For **Availability**: log `wayback_discovery(stage="availability")`.
3. For **CDX**: log `wayback_discovery(stage="cdx")` with counts and first/last ts.
4. For each **evaluated** snapshot (1–2 max): log `wayback_candidate`.
5. If **HTML parse** path is taken: log `wayback_html_parse` with discovery method.
6. For archived PDF **HEAD/sniff**: log `wayback_pdf_check`.
7. On success: `wayback_emit` (and `wayback_attempt(end)` with `result="emitted_*"`).
8. On failure: `wayback_skip` (explicit `reason`) and `wayback_attempt(end)` with the same result string.

## Guardrails

* Don’t cache PDFs (raw client only).
* Don’t log HTML bodies (size only).
* Keep candidate emission ≤ 2 per attempt to avoid high cardinality.
* Always canonicalize URLs before logging (`canonical_url`) and preserve `original_url` for audit.

---

## “Stretch” enhancements (if you want best-in-class)

* **Offline awareness:** if `only-if-cached` is set globally, short-circuit discovery, emit `wayback_skip(reason="blocked_offline")`.
* **Backpressure visibility:** add `breaker_open=true|false` if your circuit breaker blocked network access to Wayback.
* **Latency buckets:** maintain Prom-style buckets client-side (e.g., `<100ms`, `<250ms`, `<1s`, `>=1s`) to visualize path health quickly.

---

If you want, I can convert this spec into a stubbed `TelemetryWayback` class with `emit_*()` method signatures and docstrings so the agent can wire calls without guessing field names.
