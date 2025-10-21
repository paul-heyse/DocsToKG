Awesome—here’s a concrete “filmstrip” of how one run looks on the wire, with **exact Attempt CSV rows** and a **Manifest JSONL row** for each outcome. I’m using the stable tokens and shapes we agreed on:

* Attempt CSV header (once per file):
  `ts,run_id,resolver,url,verb,status,http_status,content_type,elapsed_ms,bytes_written,content_length_hdr,reason`
* Manifest JSONL fields (one line per artifact):
  `{ "ts","run_id","artifact_id","resolver","url","outcome","ok","reason","path","content_type","bytes","html_paths","config_hash","dry_run" }`

I’ll walk through a common **success** case first, then add two short variants (**304 skip** and **robots skip**) plus one **integrity error** case. Timestamps and sizes are illustrative but consistent.

---

## Effective config (excerpt used in these examples)

```yaml
run_id: "2025-10-21T23:12:45Z-abc123"
http:
  user_agent: "DocsToKG/ContentDownload (+mailto:data@example.org)"
  timeout_connect_s: 10.0
  timeout_read_s: 60.0
robots:
  enabled: true
  ttl_seconds: 3600
download:
  atomic_write: true
  verify_content_length: true
  chunk_size_bytes: 1048576    # 1 MiB
telemetry:
  csv_path: "logs/attempts.csv"
  manifest_path: "logs/manifest.jsonl"
resolvers:
  order: ["unpaywall","crossref","landing","wayback"]
  unpaywall:
    enabled: true
    retry: { max_attempts: 4, retry_statuses: [429,500,502,503,504], base_delay_ms: 200, max_delay_ms: 4000, jitter_ms: 100 }
    rate_limit: { capacity: 5, refill_per_sec: 1.0, burst: 2 }
```

For reference, **config_hash** below is a placeholder for `sha256(model_dump_json(..., sort_keys=True))`:

```
config_hash: "sha256:8f5b1a9c6f9d4f3c6c1f0e1a3a77be2a7e4f9d6c0c1d3a1b2c3d4e5f6a7b8c9d"
```

---

## A) Success — Unpaywall → direct PDF (200 OK)

**Artifact**

* `artifact_id`: `doi:10.1234/abcd.5678`
* Resolver returns plan:

  * `resolver_name="unpaywall"`
  * `url="https://cdn.publisher.org/article/abcd5678.pdf"`
  * `expected_mime="application/pdf"`

**Attempts appended to `logs/attempts.csv`**
*(CSV header appears once at file creation; shown here for clarity)*

```
ts,run_id,resolver,url,verb,status,http_status,content_type,elapsed_ms,bytes_written,content_length_hdr,reason
2025-10-21T23:12:46.120Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/abcd5678.pdf,HEAD,http-head,200,application/pdf,92,,,
2025-10-21T23:12:46.325Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/abcd5678.pdf,GET,http-get,200,application/pdf,145,,,
2025-10-21T23:12:46.980Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/abcd5678.pdf,GET,http-200,200,application/pdf,,1245721,1245721,
```

* Row 1 = HEAD probe (`status=http-head`) with server-declared MIME.
* Row 2 = GET response arrival (`status=http-get`) and handshake latency.
* Row 3 = Stream finished and fsync/rename done (`status=http-200`) with `bytes_written` matching `Content-Length`.

**Final file**

* Path (example policy): `/data/docs/2025/10/abcd5678.pdf`
* Size: `1,245,721` bytes (matches `Content-Length`)

**Manifest JSONL appended to `logs/manifest.jsonl`**

```json
{
  "ts": "2025-10-21T23:12:47.005Z",
  "run_id": "2025-10-21T23:12:45Z-abc123",
  "artifact_id": "doi:10.1234/abcd.5678",
  "resolver": "unpaywall",
  "url": "https://cdn.publisher.org/article/abcd5678.pdf",
  "outcome": "success",
  "ok": true,
  "reason": null,
  "path": "/data/docs/2025/10/abcd5678.pdf",
  "content_type": "application/pdf",
  "bytes": 1245721,
  "html_paths": [],
  "config_hash": "sha256:8f5b1a9c6f9d4f3c6c1f0e1a3a77be2a7e4f9d6c0c1d3a1b2c3d4e5f6a7b8c9d",
  "dry_run": false
}
```

---

## B) Skip — Conditional GET → 304 Not Modified

**Artifact**

* `artifact_id`: `doi:10.7777/unchanged.0001`
* Plan includes prior cache hints: `etag="W/\"abc123\""`, `last_modified="Mon, 20 Oct 2025 12:00:00 GMT"`
* Client sends conditional GET; server returns `304`.

**Attempts**

```
2025-10-21T23:14:12.210Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/unchanged0001.pdf,GET,http-get,304,application/pdf,88,,,
2025-10-21T23:14:12.210Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/unchanged0001.pdf,GET,http-304,304,application/pdf,,,,"not-modified"
```

**Manifest JSONL**

```json
{
  "ts": "2025-10-21T23:14:12.235Z",
  "run_id": "2025-10-21T23:12:45Z-abc123",
  "artifact_id": "doi:10.7777/unchanged.0001",
  "resolver": "unpaywall",
  "url": "https://cdn.publisher.org/article/unchanged0001.pdf",
  "outcome": "skip",
  "ok": false,
  "reason": "not-modified",
  "path": null,
  "content_type": "application/pdf",
  "bytes": null,
  "html_paths": [],
  "config_hash": "sha256:8f5b1a9c6f9d4f3c6c1f0e1a3a77be2a7e4f9d6c0c1d3a1b2c3d4e5f6a7b8c9d",
  "dry_run": false
}
```

> Notes: two lines are shown for clarity. Your implementation might emit just the `http-304` line (either is fine as long as the tokens stay stable).

---

## C) Skip — Robots disallowed (landing page)

**Artifact**

* `artifact_id`: `url:https://example-journal.org/articles/56789`
* Resolver `landing` would fetch the page to look for a PDF, but **robots** disallow this URL.

**Attempts**

```
2025-10-21T23:16:03.015Z,2025-10-21T23:12:45Z-abc123,landing,https://example-journal.org/articles/56789,ROBOTS,robots-fetch,,,35,,,ok
2025-10-21T23:16:03.020Z,2025-10-21T23:12:45Z-abc123,landing,https://example-journal.org/articles/56789,ROBOTS,robots-disallowed,,,,,,robots
```

* The first line shows we fetched `/robots.txt` (`status=robots-fetch`); `elapsed_ms=35`.
* The second line records the policy decision (`status=robots-disallowed`)—**no GET** was attempted.

**Manifest JSONL**

```json
{
  "ts": "2025-10-21T23:16:03.045Z",
  "run_id": "2025-10-21T23:12:45Z-abc123",
  "artifact_id": "url:https://example-journal.org/articles/56789",
  "resolver": "landing",
  "url": "https://example-journal.org/articles/56789",
  "outcome": "skip",
  "ok": false,
  "reason": "robots",
  "path": null,
  "content_type": null,
  "bytes": null,
  "html_paths": [],
  "config_hash": "sha256:8f5b1a9c6f9d4f3c6c1f0e1a3a77be2a7e4f9d6c0c1d3a1b2c3d4e5f6a7b8c9d",
  "dry_run": false
}
```

---

## D) Error — Truncated body (size mismatch)

**Artifact**

* `artifact_id`: `doi:10.31337/truncated.9000`
* Server advertises `Content-Length: 500000`, but connection drops after ~100 KB.

**Attempts**

```
2025-10-21T23:18:41.100Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/truncated9000.pdf,GET,http-get,200,application/pdf,97,,,
2025-10-21T23:18:41.540Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/truncated9000.pdf,GET,size-mismatch,200,application/pdf,,,500000,size-mismatch
```

* The atomic writer **removes the temp file** and no final path is produced.

**Manifest JSONL**

```json
{
  "ts": "2025-10-21T23:18:41.565Z",
  "run_id": "2025-10-21T23:12:45Z-abc123",
  "artifact_id": "doi:10.31337/truncated.9000",
  "resolver": "unpaywall",
  "url": "https://cdn.publisher.org/article/truncated9000.pdf",
  "outcome": "error",
  "ok": false,
  "reason": "size-mismatch",
  "path": null,
  "content_type": "application/pdf",
  "bytes": null,
  "html_paths": [],
  "config_hash": "sha256:8f5b1a9c6f9d4f3c6c1f0e1a3a77be2a7e4f9d6c0c1d3a1b2c3d4e5f6a7b8c9d",
  "dry_run": false
}
```

---

## (Optional) Variation — Backoff then success (429 with Retry-After)

If you want a “politeness” example too, here’s the short pattern (no manifest shown, since the final success row would look like **(A)**):

**Attempts**

```
2025-10-21T23:20:12.000Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/queued123.pdf,GET,http-get,429,application/json,55,,,
2025-10-21T23:20:12.000Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/queued123.pdf,GET,retry,429,application/json,1000,,,retry-after
2025-10-21T23:20:13.120Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/queued123.pdf,GET,http-get,200,application/pdf,120,,,
2025-10-21T23:20:13.760Z,2025-10-21T23:12:45Z-abc123,unpaywall,https://cdn.publisher.org/article/queued123.pdf,GET,http-200,200,application/pdf,,987654,987654,
```

* Line 2 is the **retry attempt** emission with `status=retry`, `reason=retry-after`, and `elapsed_ms=1000` (sleep time).
* Then normal GET/stream completes as in **(A)** and the manifest would show `success`.

---

## Cross-check against our vocabulary

* **status** used: `http-head`, `http-get`, `http-200`, `http-304`, `robots-fetch`, `robots-disallowed`, `retry`, `size-mismatch`.
* **reason** used: `ok`, `not-modified`, `robots`, `retry-after`, `size-mismatch`.
* **classification** used in manifest: `success`, `skip`, `error`.

All tokens match the stable sets we locked in earlier.

---

If you want me to **emit these as ready-to-load fixtures** (CSV + JSONL files) for a test or a demo dashboard, say the word and I’ll print them verbatim with consistent timestamps and IDs.
