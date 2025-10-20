# Wayback Machine Telemetry Specification

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

* Don't cache PDFs (raw client only).
* Don't log HTML bodies (size only).
* Keep candidate emission ≤ 2 per attempt to avoid high cardinality.
* Always canonicalize URLs before logging (`canonical_url`) and preserve `original_url` for audit.

## Event Types

### wayback_attempt

Start and end events for each resolver attempt.

**Start event:**

```json
{
  "event_type": "wayback_attempt",
  "event": "start",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "resolver": "wayback",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 0,
  "original_url": "https://example.com/paper.pdf",
  "canonical_url": "https://example.com/paper.pdf",
  "publication_year": 2023
}
```

**End event:**

```json
{
  "event_type": "wayback_attempt",
  "event": "end",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "resolver": "wayback",
  "ts": "2023-01-01T00:00:01Z",
  "monotonic_ms": 1000,
  "mode_selected": "pdf_direct",
  "result": "emitted_pdf",
  "total_duration_ms": 1000,
  "candidates_scanned": 5
}
```

### wayback_discovery

Availability and CDX API queries.

**Availability:**

```json
{
  "event_type": "wayback_discovery",
  "stage": "availability",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 100,
  "query_url": "https://example.com/paper.pdf",
  "year_window": "-2..+2",
  "http_status": 200,
  "from_cache": true,
  "revalidated": false,
  "rate_delay_ms": 100,
  "retry_after_s": null,
  "retry_count": 0,
  "error": null
}
```

**CDX:**

```json
{
  "event_type": "wayback_discovery",
  "stage": "cdx",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 200,
  "query_url": "https://example.com/paper.pdf",
  "year_window": "-2..+2",
  "limit": 8,
  "returned": 3,
  "first_ts": "20230101000000",
  "last_ts": "20231201000000",
  "http_status": 200,
  "from_cache": false,
  "revalidated": false,
  "rate_delay_ms": 200,
  "retry_after_s": null,
  "retry_count": 0,
  "error": null
}
```

### wayback_candidate

Evaluation of individual snapshot candidates.

```json
{
  "event_type": "wayback_candidate",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 300,
  "archive_url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
  "memento_ts": "20230101000000",
  "statuscode": 200,
  "mimetype": "application/pdf",
  "source_stage": "cdx",
  "decision": "head_check",
  "distance_to_pub_year": 0
}
```

### wayback_html_parse

HTML parsing to discover PDF links.

```json
{
  "event_type": "wayback_html_parse",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 400,
  "archive_html_url": "https://web.archive.org/web/20230101000000/https://example.com/",
  "html_http_status": 200,
  "from_cache": true,
  "revalidated": false,
  "html_bytes": 5000,
  "pdf_link_found": true,
  "pdf_discovery_method": "meta",
  "discovered_pdf_url": "https://example.com/paper.pdf"
}
```

### wayback_pdf_check

PDF verification via HEAD/GET.

```json
{
  "event_type": "wayback_pdf_check",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 500,
  "archive_pdf_url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
  "head_status": 200,
  "content_type": "application/pdf",
  "content_length": 50000,
  "is_pdf_signature": true,
  "min_bytes_pass": true,
  "decision": "head_check"
}
```

### wayback_emit

Successful PDF emission.

```json
{
  "event_type": "wayback_emit",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 600,
  "emitted_url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
  "memento_ts": "20230101000000",
  "source_mode": "pdf_direct",
  "http_ct_expected": "application/pdf"
}
```

### wayback_skip

Explicit failure reasons.

```json
{
  "event_type": "wayback_skip",
  "run_id": "uuid",
  "work_id": "work-123",
  "artifact_id": "artifact-456",
  "attempt_id": "attempt-789",
  "ts": "2023-01-01T00:00:00Z",
  "monotonic_ms": 700,
  "reason": "no_snapshot",
  "details": "No snapshots found for this URL"
}
```

## Enums

### AttemptResult

- `emitted_pdf` - Direct PDF snapshot found and emitted
* `emitted_pdf_from_html` - PDF found via HTML parsing and emitted
* `skipped_no_snapshot` - No snapshots available
* `skipped_non_pdf` - Snapshots exist but none are PDFs
* `skipped_below_min_size` - PDF too small
* `skipped_blocked_offline` - Offline mode blocked network access
* `error_http` - HTTP error occurred
* `error_cdx` - CDX API error
* `timeout` - Request timeout

### ModeSelected

- `pdf_direct` - Direct PDF snapshot used
* `html_parse` - HTML parsing path used
* `none` - No suitable method found

### DiscoveryStage

- `availability` - Wayback Availability API
* `cdx` - Wayback CDX API

### CandidateDecision

- `head_check` - Proceed to HEAD check
* `skipped_status` - Skipped due to status code
* `skipped_mime` - Skipped due to MIME type

### PdfDiscoveryMethod

- `meta` - Found via citation_pdf_url meta tag
* `link` - Found via alternate link tag
* `anchor` - Found via anchor element

### SkipReason

- `no_snapshot` - No snapshots available
* `all_non_pdf` - All snapshots are non-PDF
* `below_min_size` - PDF below minimum size
* `non_pdf` - Not a PDF
* `blocked_offline` - Blocked by offline mode
* `timeout` - Request timeout
* `cdx_error` - CDX API error
* `http_error` - HTTP error
