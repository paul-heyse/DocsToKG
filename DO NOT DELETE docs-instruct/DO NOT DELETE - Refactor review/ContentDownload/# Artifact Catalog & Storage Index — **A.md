# Artifact Catalog & Storage Index — **Architecture & Narrative Companion (PR #9)**

> Save this as `src/DocsToKG/ContentDownload/ARCHITECTURE_catalog.md`.
> It explains **what the catalog is**, how it **fits** with the pipeline/orchestrator/hishel, the **storage layouts** (policy path vs CAS), **dedup**, **verification**, **GC/retention**, and exactly what each new artifact in the implementation plan does. It’s meant to build intuition so you can operate and extend the system confidently.

---

## 0) Why a catalog? (intuition first)

Your pipeline already downloads files safely (hishel caching, retries, atomic writes) and logs auditable events (CSV / JSONL). The **catalog** adds:

* A **queryable index** of what you have: *what file did we store for this DOI/URL? what’s its hash? where is it?*
* **Deduplication** across resolvers and runs via content hash (**sha-256**).
* **Verification**: detect corruption or silent changes (re-hash vs catalog).
* **Retention & GC**: delete orphans or aged content confidently, with provenance.

The catalog is not a second telemetry system; it’s the **source of truth for stored artifacts.**

---

## 1) Big picture (how it all fits)

```mermaid
flowchart LR
  subgraph Runtime["ContentDownload process"]
    PPL[ResolverPipeline]
    FIN[Finalize (atomic move, optional sha256)]
    CAT[(CatalogStore\nSQLite)]
    STG[/Storage/]
    TEL[RunTelemetry\n(CSV/JSONL+OTel)]
    ORCH[Orchestrator\n(PR #8)]
  end

  PPL --> FIN
  FIN -->|policy path or CAS| STG
  FIN -->|register_or_get(...)| CAT
  FIN --> TEL
  ORCH --> PPL

  subgraph Ops["CLI / Ops"]
    IMP[import-manifest]
    SHOW[show/where]
    DEDUP[dedup-report]
    VER[verify]
    GC[gc]
  end

  IMP --> CAT
  SHOW --> CAT
  DEDUP --> CAT
  VER --> CAT
  GC --> CAT
  GC --> STG
```

**Key ideas**

* Finalization computes **sha-256** (configurable), chooses a **final path** (policy path or **CAS**), performs an **atomic rename**, then **registers** the record in the catalog.
* Catalog records store a **URI** (`file:///...`, `s3://bucket/key`), size, MIME, resolver, URL, artifact id, and run id.
* CLI tools operate **against the catalog** (for insight) and **the storage** (for verify/GC).

---

## 2) Core data model (ER view)

```mermaid
erDiagram
  DOCUMENTS ||--o{ VARIANTS : has
  DOCUMENTS {
    int id PK
    string artifact_id
    string source_url
    string resolver
    string content_type
    int    bytes
    string sha256
    string storage_uri
    string created_at
    string updated_at
    string run_id
  }
  VARIANTS {
    int id PK
    int document_id FK
    string variant       // "pdf" | "html" | "supp" ...
    string storage_uri
    int    bytes
    string content_type
    string sha256
    string created_at
  }
```

* **Idempotence**: `(artifact_id, source_url, resolver)` is unique — reruns won’t duplicate rows.
* **sha-256** may be null if hashing is disabled; when enabled it unlocks CAS and dedup reporting.
* **Variants** let you attach multiple outputs to one document (optional now; schema is ready).

---

## 3) Storage layouts (how files are placed on disk or S3)

### A) Policy path (default; simple)

* Final path is derived from a policy (e.g., basename from URL or your existing layout).
* One file per artifact per resolver+URL; duplicates might exist across resolvers unless dedup is applied separately.
* Easiest when you need predictable human paths.

### B) CAS (content addressable storage)

* Final path depends on **sha-256** only:
  `data/cas/sha256/ab/cdef...` (two-level fan-out to avoid hot directories).
* **Dedup is free**: same content → same path.
* You can:

  * store only the CAS file and record that path in the catalog, or
  * keep policy paths as **hardlinks** to the CAS file (`hardlink_dedup=true`) for human-friendly locations.

**Choosing:** CAS is ideal for large-scale dedup and integrity. Policy paths are ideal for human browsing. Both are supported; you can even do **CAS-only** for pure de-dup.

---

## 4) Lifecycle flows (how a file becomes a catalog record)

### A) Successful download (200 OK) — with CAS

1. **Stream** to temp file (`.part-XXXX.tmp`) and count bytes.
2. **Compute sha-256** (stream-time or post-write).
3. **Pick final path**: `cas_path(root, sha256)`; if it exists → **dedup hit**.
4. **Move** temp → final (`os.replace`) or **hardlink** and remove temp.
5. **Register** in catalog: `(artifact_id, url, resolver, content_type, bytes, sha, storage_uri, run_id)`.
6. Emit `http-200` attempt; `manifest.jsonl` includes `bytes`, `content_type`, and `sha`.

### B) Revalidation (304)

* No bytes streamed; **no new file**.
* No catalog record registered unless you choose to record a “seen” variant (not typical).
* Telemetry shows `http-304`; manifest outcome = `skip`.

### C) Cache hit (hishel)

* No network; **no write**; no catalog change unless you re-register old runs (not typical).
* Telemetry shows `cache-hit`.

### D) Verify (later CLI)

* Recompute sha over `storage_uri`; compare to catalog’s `sha256`.
* If mismatch → **verify failure** metric increments; actionable alert.

### E) GC and Retention

* **GC**: compute set of files under `root_dir` – referenced catalog `storage_uri`s; delete extras (optionally, older than N days).
* **Retention**: delete catalog rows older than policy; delete associated files. (Use with care; default off.)

---

## 5) Concurrency & safety

* Finalization uses **atomic rename** or hardlink then delete.
* **No partial finals**: on any error, temp is removed; final is untouched.
* Hashing can be **stream-time** (preferred) or **post-write**; if post-write, always verify once after move when `verify_on_register=true`.
* With orchestrator (PR #8), many workers may finalize concurrently — CAS dedup logic must be **race-safe**:

  * Try `os.link` or `os.replace`; handle `EEXIST` by deleting temp.
  * Always verify final exists and records the same sha.

---

## 6) What each artifact in the plan *actually does*

### `catalog/schema.sql`

* **What:** DB schema for SQLite (Postgres-friendly).
* **Why:** Define canonical tables (`documents`, `variants`) and indexes (idempotence, sha, run_id).

### `catalog/models.py`

* **What:** small dataclasses (`DocumentRecord`) used by the store and callers.
* **Why:** typed edges; clear field meaning.

### `catalog/store.py` → `SQLiteCatalog`

* **What:** imperative CRUD with `register_or_get()` idempotent write.
* **Why:** one place to talk to the DB; easy to swap to Postgres later.
* **Key contracts:** `register_or_get`, `get_by_artifact`, `get_by_sha256`, `find_duplicates`, `verify`, `stats`.

### `catalog/fs_layout.py`

* **What:** build **CAS** path; build **policy path**; perform **hardlink-or-copy** dedup.
* **Why:** unify path decisions; isolate FS primitives.

### `catalog/s3_layout.py`

* **What:** seam for S3: generate keys, do puts, return `s3://` URIs.
* **Why:** you can keep FS staging and upload after; catalog doesn’t change.

### `catalog/gc.py`

* **What:** list orphans (files in root not referenced by catalog) and apply deletion (with dry-run).
* **Why:** safe cleanup; keeps store slim.

### `catalog/migrate.py`

* **What:** **import-manifest** helper — backfill older runs from `manifest.jsonl`.
* **Why:** keep history continuity.

### Config (`StorageConfig` / `CatalogConfig`)

* **What:** typed knobs (layout, root paths, hashing, verification, retention).
* **Why:** a single source of truth — pipeline just reads.

### Pipeline / finalize changes

* **What:** compute sha-256, choose final path, atomic write, register to catalog, build runtime outcome meta.
* **Why:** single responsibility for file integrity and persistence.

### CLI `catalog` commands

* **What:** `import-manifest`, `show`, `where`, `dedup-report`, `verify`, `gc`.
* **Why:** operator ergonomics; no ad-hoc scripts necessary.

### Telemetry metrics (OTel)

* **What:** `contentdownload_dedup_hits_total`, `contentdownload_gc_removed_total`, `contentdownload_verify_failures_total`.
* **Why:** quick signals for storage health and efficiency.

---

## 7) Example: one file, two resolvers (dedup story)

* Run A (`unpaywall`) downloads `abcd5678.pdf` → sha `e3b0…` → CAS path `…/cas/sha256/e3/b0…` → catalog record #101 points to `file:///…/cas/sha256/e3/b0…`.
* Run B (`crossref`) finds the *same* bytes: finalization sees CAS target exists → **hardlink** or reuse path; register record #154 with a **different** `(artifact_id, url, resolver)` but **same `sha256` and `storage_uri`**.
* `catalog dedup-report` shows `e3b0…: 2`.
* Only **one physical copy** exists (inode count shows 2 links if hardlinking; else one file referenced twice if you store only CAS).

---

## 8) Ops cookbook

**Register everything from an old manifest:**

```bash
contentdownload catalog import-manifest -c cd.yaml /path/to/manifest.jsonl
```

**Find where a file lives by sha:**

```bash
contentdownload catalog where e3b0c44298...
```

**Show what we’ve stored for an artifact:**

```bash
contentdownload catalog show doi:10.1234/abcd.5678
```

**Get a dedup report:**

```bash
contentdownload catalog dedup-report
```

**Verify a specific record:**

```bash
contentdownload catalog verify 101
```

**Garbage collect orphans (dry-run first):**

```bash
contentdownload catalog gc -c cd.yaml --dry-run
contentdownload catalog gc -c cd.yaml --apply
```

---

## 9) Recommended defaults & tuning

* **Hashing:** `compute_sha256=true` (on), `verify_on_register=false` (off) — enable verify for S3 or high-risk storage.
* **Layout:** start with **CAS**, `hardlink_dedup=true` on POSIX; set `false` on Windows or cross-FS.
* **Retention:** leave `retention_days=0` until you’re confident; run **GC dry-run** weekly.
* **S3:** keep **FS staging** (CAS) and upload asynchronously; record the `s3://…` URI in catalog once put succeeds.

---

## 10) Edge cases & guardrails

* **Cross-filesystem moves:** atomic `os.replace` may fail across devices. Our layouts create final paths in the **same root** as temp to keep renames atomic. If that’s not feasible, use **copy → fsync → rename** fallback.
* **Concurrent CAS writes:** two workers computing the same hash may race to create the same CAS path. We treat `EEXIST` as a **dedup hit** and safely delete temp.
* **Hash mismatch after move:** treat as a critical error; increment `verify_failures_total`; leave file in quarantine path; do not register.
* **Path hygiene:** catalog stores **URIs** (not raw paths) to make backend transitions (FS → S3) painless.
* **Metrics cardinality:** don’t put `artifact_id` or `url` in metric labels; use `resolver`, maybe `content_type`.

---

## 11) Testing heuristics (what to prove)

* **Idempotent register**: same `(artifact_id,url,resolver)` yields the same row id.
* **Dedup race**: run two finalizations with the same sha in parallel → only one CAS file physically; both catalog records present.
* **Verify**: tamper a file → `verify` fails and metric increments.
* **GC**: create an orphan in `root_dir` → `gc --apply` removes it and records a metric.
* **S3 smoke** (when implemented): upload stub; catalog URI is `s3://…`.

---

## 12) How this interacts with hishel & the orchestrator

* **hishel** only affects **network behavior**; the catalog records **what we persisted** — independent of cache hits/revalidations.
* **Orchestrator** introduces concurrency; catalog registration remains **thread-safe**. CAS logic must be race-safe (as described).
* **Telemetry** continues to be the operational feedback loop; the catalog is your **inventory & provenance**.

---

## 13) Migration guidance (from just manifest files)

1. Roll out catalog **disabled** (`compute_sha256=false`) to land schema.
2. Run `catalog import-manifest` for recent manifests (or the full history).
3. Enable `compute_sha256=true` and **CAS** on a subset of new runs; confirm dedup stats and path correctness.
4. Switch new runs to **CAS + (optional) hardlink policy paths**; keep verifying selectively (`verify_on_register=true`) if risk warrants.
5. Set up a weekly **GC dry-run**, then **apply** after reviewing logs.

---

## 14) Glossary

* **CAS (Content Addressable Storage)**: file path is derived from its hash; identical content → identical path.
* **Policy path**: human-friendly path based on artifact metadata (e.g., URL basename).
* **Dedup**: avoid storing identical bytes more than once (hardlink or single CAS copy).
* **Orphan**: a file under storage root not referenced by any catalog record.
* **Retention**: policy to remove **old** catalog records and their files.

---

### TL;DR

* The **catalog** tells you *what* you have and *where* it lives.
* **CAS + sha-256** gives you **dedup & integrity** by construction.
* **GC/retention** keeps storage lean and safe.
* It snaps into place with your existing pipeline, orchestrator, and hishel — no behavioral changes, only **new capabilities**.
