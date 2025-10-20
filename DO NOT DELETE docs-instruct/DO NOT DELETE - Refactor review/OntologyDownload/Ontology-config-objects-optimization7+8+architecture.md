heck yes—here’s the final pair, rendered so you can “see” the platform at a glance. Drop these straight into your docs.

---

# E) **Observability that answers questions** + **Safety & policy (defense-in-depth)**

## E1. Big-board: how safety gates and events thread through the system

```
                          ┌───────────────────────────────────────────┐
                          │          Traced Settings (Pydantic)       │
                          │  {submodels, source_fingerprint, hash}    │
                          └───────────────┬───────────────────────────┘
                                          │ normalized policies & hash
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION PIPELINE  (one command run_id)                         │
│                                                                                          │
│  CLI (Typer) ──►  policy.gate(config) ──✓/✗──►   NETWORK   ───►  EXTRACT   ───►   DB     │
│  (plan/pull/      (strict settings)               (HTTPX)         (libarchive)       (DuckDB)
│   extract/…)                                                                             │
│         │                  ▲                      ▲    ▲          ▲    ▲              ▲  │
│         │                  │                      │    │          │    │              │  │
│         ▼                  │                      │    │          │    │              │  │
│   cli.command.*      policy.gate(url) ──✓/✗  policy.gate(redirect)  policy.gate(path)│  │
│   start/done/error                          & dns/ports              & entry/type    │  │
│                                                                                          │
│         ├───────────────► EVENTS BUS (structlog JSON + DuckDB events/Parquet) ◄──────────┤
│         │            (every event carries: ts, type, level, run_id, config_hash)         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key idea:** every boundary is a **gate**; every gate emits a **structured event** whether it **passes** or **rejects**—so you can answer questions without grep.

---

## E2. Event grammar + policy taxonomy (contracts you can depend on)

### E2.1 Event envelope (present on *every* event)

```json
{
  "ts": "2025-10-20T01:23:45.678Z",
  "type": "net.request | extract.done | policy.gate | cli.command.start | db.tx.commit | ...",
  "level": "INFO|WARN|ERROR",
  "run_id": "uuid-…",
  "config_hash": "sha256:…",
  "context": { "app_version": "…", "os": "…", "python": "…", "libarchive_version": "…" },
  "ids": { "service": "ols", "version_id": "2025-10-20T01-23-45Z",
           "artifact_id": null, "file_id": null, "request_id": "uuid-…" },
  "payload": { "… event-specific fields …" }
}
```

### E2.2 Policy error catalog (singular source of truth)

```
Network/TLS:  E_NET_CONNECT  E_NET_READ  E_NET_PROTOCOL  E_TLS
URL/DNS:      E_SCHEME  E_USERINFO  E_HOST_DENY  E_PORT_DENY  E_DNS_FAIL  E_PRIVATE_NET
Redirect:     E_REDIRECT_DENY  E_REDIRECT_LOOP
Filesystem:   E_TRAVERSAL  E_CASEFOLD_COLLISION  E_DEPTH  E_SEGMENT_LEN  E_PATH_LEN  E_PORTABILITY
Extraction:   E_SPECIAL_TYPE  E_BOMB_RATIO  E_ENTRY_RATIO  E_FILE_SIZE  E_FILE_SIZE_STREAM  E_ENTRY_BUDGET
Storage:      E_STORAGE_PUT  E_STORAGE_MOVE  E_STORAGE_MARKER
DB/Boundaries:E_DB_TX  E_DB_MIGRATION  E_LATEST_MISMATCH
```

**All rejections travel one road:** `policy.errors.raise_with_event(code, details)` → emits `*.error` with `error_code` + context, then raises the right exception.

---

## E3. Safety gates—where they sit and what they enforce

```
┌────────────────────────────────────────────────────────────────┐
│ policy.gate(config)                                            │
│  - strict types & bounds; JSON Schema exists                   │
│  - normalized allowlists/ports/globs; emit policy.gate (OK)    │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ policy.gate(url)   (pre-request, and per redirect hop)         │
│  - scheme http/https only; userinfo forbidden                  │
│  - IDN→punycode; PSL allowlist (exact/suffix/ip/cidr)          │
│  - per-host ports + global port set                            │
│  - http→https upgrade unless allowlisted plain-http            │
│  - DNS resolve (strict/lenient); private nets policy           │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ policy.gate(path)  (before write)                              │
│  - encapsulation root; dirfd/openat; O_NOFOLLOW|O_EXCL         │
│  - NFC normalization, forbid .. / absolute / drive letters     │
│  - casefold collision; depth/segment/path length; win reserved │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ policy.gate(entry) (pre-scan + per entry)                      │
│  - type allowlist: regular files only                          │
│  - zip-bomb guards: total ratio + per-entry ratio              │
│  - per-file size; max entries; include/exclude filters         │
│  - explicit perms (0644/0755); strip suid/sgid/sticky          │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────┐
│ policy.gate(storage/db) (boundaries)                           │
│  - Latest set = DB TX + atomic marker write                    │
│  - Commit DB only after FS success (no torn writes)            │
│  - Doctor flags latest mismatch; Prune deletes by staging diff │
└────────────────────────────────────────────────────────────────┘
```

Each gate emits `policy.gate` **OK** with `ms` + summary; on reject it emits **ERROR** with `error_code` + compact `details` (never secrets).

---

## E4. Run timeline—events + gates (one happy path; one reject path)

### E4.1 Happy path (redirect with cache revalidate; extraction succeeds)

```
cli.command.start
  │
  ├─ policy.gate(config) → OK
  │
  ├─ ratelimit.acquire → {blocked_ms:0, outcome:"ok"}
  │
  ├─ net.request (302)  ──► policy.gate(url:Location) → OK  ──► net.request (200 revalidated)
  │                                    │
  │                                    └─ cache:"revalidated", ttfb_ms, elapsed_ms
  │
  ├─ policy.gate(path root) → OK
  ├─ policy.gate(entry pre-scan) → OK (totals, ratios)
  │
  ├─ extract.start → stream→hash→fsync→rename
  ├─ extract.done  → {entries_total, entries_included, bytes_written, ratio_total, duration_ms}
  │
  ├─ db.tx.commit (extracted_files bulk insert) → OK
  │
  └─ cli.command.done {elapsed_ms, files, bytes}
```

### E4.2 Reject path (redirect to disallowed host)

```
… net.request (302) → Location: http://evil.example/…
         │
         └─ policy.gate(url:Location) → ERROR {error_code:"E_HOST_DENY", host:"evil.example", ...}
                   │
                   └─ net.request ends with ERROR (mapped to E_REDIRECT_DENY)
                       + cli.command.error → exit code 3 (policy)
```

---

## E5. Where the data lands (so you can answer questions fast)

```
Events (JSON) ──► stdout / file.jsonl
      └───────► DuckDB events table (or Parquet logs)
                     │
                     ├─ Stock queries / CLI:
                     │   • SLO: p95 net.request per service; cache hit ratio
                     │   • Safety: policy.gate rejects by error_code (top offenders)
                     │   • Capacity: sum(bytes_written) by service/day
                     │   • Stability: error_code histogram last N runs
                     │   • Delta guard: correlate extract.done with delta summaries
                     │
                     └─ Dashboards: "Top blocked keys", "Redirect denies", "Zip-bomb hits"
```

**IDs for joins**

* `run_id` (correlates CLI start/done with all nested events)
* `request_id` (join download.fetch↔net.request)
* `version_id` / `artifact_id` / `file_id` (join to catalog rows)

---

## E6. SLOs & guardrails (ready-to-use dashboards/queries)

* **Network SLO**
  `SELECT service, approx_quantile(payload.elapsed_ms, 0.95) AS p95 FROM events WHERE type='net.request' GROUP BY 1;`
* **Cache hit ratio**
  `SELECT service, avg(payload.cache IN ('hit','revalidated')) AS hit_ratio FROM events WHERE type='net.request' GROUP BY 1;`
* **Rate-limit pressure**
  `SELECT substr(payload.key,1,40) AS key, sum(payload.blocked_ms) AS blocked_ms FROM events WHERE type='ratelimit.acquire' GROUP BY 1 ORDER BY 2 DESC LIMIT 10;`
* **Safety heatmap**
  `SELECT payload.error_code, count(*) FROM events WHERE type LIKE '%.error' GROUP BY 1 ORDER BY 2 DESC;`
* **Zip-bomb sentinel**
  `SELECT ts, payload.ratio_total FROM events WHERE type='extract.done' AND payload.ratio_total > 10.0;`

---

## E7. “Why this is hard to break” (defense-in-depth recap)

* Gates wrap **every I/O boundary**; no direct writes or follows without passing the gate.
* One **error catalog**; one **emission path**; one **query surface**—no mystery failures.
* Provenance baked in: `source_fingerprint` (who set what) → `config_hash` (exact config) → `run_id` (what happened).
* Doctor/Prune are **safe-by-default** (dry-run + typed confirm) and operate from a **staging diff**, not guesses.

---

## E8. Acceptance “at a glance”

* [ ] All subsystems emit namespaced events with `{run_id, config_hash}`.
* [ ] Every gate emits `policy.gate` **OK/ERROR** with timing and codes.
* [ ] Redirects are *audited* (deny on policy breach); file writes are *encapsulated* (no escapes).
* [ ] DuckDB (or Parquet) accumulates events; stock queries return SLOs, safety heatmaps, capacity charts.
* [ ] CLI `obs stats|tail|export` and `delta summary` answer Monday-morning questions in seconds.

---

If you want, I can package these visuals into a single **“System Architecture”** page (with linkable anchors), plus a **one-page laminated runbook** (top queries + meanings) you can keep in the repo root.
