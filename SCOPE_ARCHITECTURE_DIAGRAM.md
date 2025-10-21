# ARCHITECTURE DIAGRAMS: Pillars 5 & 6 Integration

## Diagram 1: High-Level System Flow (Post-Implementation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OntologyDownload System (v6.0.0)                │
│                                                                     │
│  INPUT: configs/sources.yaml + CLI flags                            │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 1: PLANNING (Phases 5.5–5.9: COMPLETE)               │  │
│  │                                                              │  │
│  │  planning.plan_all(specs)                                    │  │
│  │    ├─ settings.build_resolved_config()                      │  │
│  │    ├─ Polite HTTP Client (rate-limit aware)                │  │
│  │    ├─ Policy gates (URL/path/extraction validation)         │  │
│  │    └─ emit: net.request, ratelimit.acquire events           │  │
│  │                                                              │  │
│  │  Output: PlannedFetch list                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 2: DOWNLOAD (Phases 5.5–5.9 + NEW: Phase 5A bounds)  │  │
│  │                                                              │  │
│  │  io.network.download_stream(urls)                            │  │
│  │    ├─ HTTP client + Hishel caching                           │  │
│  │    ├─ Rate limiting (acquire before I/O)                     │  │
│  │    ├─ Redirect audit (URL security gate)                    │  │
│  │    └─ Stream → file.tmp → rename                            │  │
│  │                                                              │  │
│  │  ┌─ NEW: @db.download_boundary() TX ────────────────┐       │  │
│  │  │  • Insert artifact (hash, size, etag, fs_relpath)│       │  │
│  │  │  • Emit: download.fetch event                     │       │  │
│  │  └──────────────────────────────────────────────────┘       │  │
│  │                                                              │  │
│  │  Output: DownloadResult (checksum, bytes)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 3: EXTRACTION (NEW: Phase 5A + 5C)                   │  │
│  │                                                              │  │
│  │  io.extraction_throughput.extract_archive()                  │  │
│  │    ├─ Pre-scan policy gates (size limits, patterns)          │  │
│  │    ├─ Stream files + hashing                                 │  │
│  │    ├─ Write: data/** + .extract.audit.json (atomic)         │  │
│  │    └─ Collect: extracted_files rows (Arrow)                │  │
│  │                                                              │  │
│  │  ┌─ NEW: @db.extraction_boundary() TX ──────────────┐       │  │
│  │  │  • Bulk insert extracted_files (Arrow appender)  │       │  │
│  │  │  • Record audit JSON deterministically            │       │  │
│  │  │  • Emit: extraction.done event                    │       │  │
│  │  └──────────────────────────────────────────────────┘       │  │
│  │                                                              │  │
│  │  Output: extraction audit + file rows                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 4: VALIDATION (NEW: Phase 5A bounds)                 │  │
│  │                                                              │  │
│  │  validation.run_validators(extracted_files)                  │  │
│  │    ├─ rdflib, pronto, owlready2, ROBOT, Arelle              │  │
│  │    ├─ Collect: ValidationResult list                         │  │
│  │    └─ Emit: validation events                                │  │
│  │                                                              │  │
│  │  ┌─ NEW: @db.validation_boundary() TX ──────────────┐       │  │
│  │  │  • Insert validations (file_id, validator, status)│       │  │
│  │  │  • Link to extracted_files                         │       │  │
│  │  │  • Emit: validation.complete event                │       │  │
│  │  └──────────────────────────────────────────────────┘       │  │
│  │                                                              │  │
│  │  Output: ValidationResult map                                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 5: FINALIZATION (NEW: Phase 5A bounds)               │  │
│  │                                                              │  │
│  │  ┌─ NEW: @db.set_latest_boundary() TX ──────────────┐       │  │
│  │  │  • Upsert latest_pointer (version_id, ts)         │       │  │
│  │  │  • Write LATEST.json (atomic rename)              │       │  │
│  │  │  • Emit: finalization.complete event              │       │  │
│  │  └──────────────────────────────────────────────────┘       │  │
│  │                                                              │  │
│  │  manifests.write_lockfile()                                  │  │
│  │    └─ Lock + manifest.json (schema v1.0)                    │  │
│  │                                                              │  │
│  │  Output: Manifest (attestation of all operations)            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 6: QUERIES & ANALYTICS (NEW: Phases 6A + 6B)         │  │
│  │                                                              │  │
│  │  DuckDB Catalog (metadata brain):                            │  │
│  │    ├─ SELECT versions, artifacts, extracted_files           │  │
│  │    ├─ SELECT validations → per-validator rates              │  │
│  │    ├─ Views: v_version_stats, v_latest_files                │  │
│  │    └─ Views: v_fs_orphans (for doctor/prune)                │  │
│  │                                                              │  │
│  │  Polars Analytics (zero-loop reporting):                     │  │
│  │    ├─ Latest Summary (bytes/files by format + top-N)         │  │
│  │    ├─ Growth A→B (delta + churn metrics)                     │  │
│  │    ├─ Validation Health (FIXED/REGRESSED)                    │  │
│  │    └─ Hotspots (power law contributors)                      │  │
│  │                                                              │  │
│  │  CLI: ontofetch report {latest|growth|validation|hotspots}  │  │
│  │    └─ Output: table|json|parquet                             │  │
│  │                                                              │  │
│  │  Output: Analytics DataFrames + CLI tables                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 7: OPERATIONS (NEW: Phases 5B + maintenance)         │  │
│  │                                                              │  │
│  │  CLI: ontofetch doctor --fix                                 │  │
│  │    └─ Reconcile FS ↔ DB, auto-heal common drifts            │  │
│  │                                                              │  │
│  │  CLI: ontofetch prune --dry-run / --apply                   │  │
│  │    └─ Safe GC of orphans with audit trail                    │  │
│  │                                                              │  │
│  │  CLI: ontofetch db backup/restore                            │  │
│  │    └─ Transactional snapshots + schema versions              │  │
│  │                                                              │  │
│  │  Output: Audit logs + events table                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  INVARIANTS (Enforced Throughout):                                │
│  ✓ Filesystem is source-of-truth for blobs; DB is metadata brain  │
│  ✓ One writer at a time (file lock); many readers                 │
│  ✓ Two-phase choreography: FS write → DB commit                   │
│  ✓ Idempotence via content hashing (sha256)                        │
│  ✓ Determinism: audit JSON + config_hash in events                │
│  ✓ No torn writes: rollback on any failure                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: DuckDB Schema & Boundaries (Pillar 5)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DuckDB Catalog (.duckdb)                     │
│                                                                 │
│  Tables:                                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ schema_version                                         │    │
│  │  • migration_name (PK)                                 │    │
│  │  • applied_at (TIMESTAMP)                              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ versions                                               │    │
│  │  • version_id (PK) — timestamp-based                   │    │
│  │  • service (FK)                                        │    │
│  │  • latest_pointer (BOOL)                               │    │
│  │  • ts (TIMESTAMP)                                      │    │
│  │  → idx: (service, latest_pointer), (ts DESC)           │    │
│  └────────────────────────────────────────────────────────┘    │
│        ↑                                                         │
│        │ Foreign key                                            │
│        │                                                         │
│  ┌─────┴────────────────────────────────────────────────────┐  │
│  │ artifacts  ← Download Boundary writes here              │  │
│  │  • artifact_id (PK) = sha256(archive)                   │  │
│  │  • version_id (FK)                                      │  │
│  │  • fs_relpath (source-of-truth → FS blobs)             │  │
│  │  • size, etag, status (fresh|cached)                    │  │
│  │  • downloaded_at (TIMESTAMP)                            │  │
│  │  → idx: (version_id, status), (artifact_id)            │  │
│  └────────────────────────────────────────────────────────┘    │
│        │                                                        │
│        │ artifact_id                                           │
│        ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ extracted_files  ← Extraction Boundary writes here     │    │
│  │  • file_id (PK) = sha256(file bytes)                   │    │
│  │  • artifact_id (FK)                                    │    │
│  │  • relpath (within extraction)                         │    │
│  │  • size, format, sha256, mtime                         │    │
│  │  • extracted_at (TIMESTAMP)                            │    │
│  │  → idx: (artifact_id, relpath), (format), (file_id)   │    │
│  └────────────────────────────────────────────────────────┘    │
│        │                                                        │
│        │ file_id                                               │
│        ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ validations  ← Validation Boundary writes here          │    │
│  │  • validation_id (PK)                                  │    │
│  │  • file_id (FK)                                        │    │
│  │  • validator (rdflib|pronto|owlready2|ROBOT|Arelle)   │    │
│  │  • status (pass|fail|timeout), details (JSON)          │    │
│  │  • validated_at (TIMESTAMP)                            │    │
│  │  → idx: (file_id, validator), (status), (validated_at) │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ latest_pointer  ← Set Latest Boundary writes here      │    │
│  │  • version_id (PK, FK)                                 │    │
│  │  • set_at (TIMESTAMP)                                  │    │
│  │  → idx: set_at DESC                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ events  ← All phases emit here                          │    │
│  │  • run_id, ts, type, level, payload (JSON)             │    │
│  │  • All boundaries append operational events             │    │
│  │  → idx: (run_id, ts DESC)                              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Views:                                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ v_version_stats                                        │    │
│  │  SELECT version_id, COUNT(files), SUM(size), …        │    │
│  └────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ v_validation_failures                                  │    │
│  │  SELECT file_id, validator, status WHERE status='fail' │    │
│  └────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ v_fs_orphans (for Doctor/Prune)                        │    │
│  │  LEFT JOIN with staging_fs_listing                      │    │
│  │  → DB rows missing from FS                             │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Transactional Boundaries (Choreography):                      │
│                                                                 │
│  1️⃣ Download Boundary                                           │
│     Filesystem:   file.tmp → rename → final path              │
│     DuckDB TX:    INSERT INTO artifacts (...)                 │
│     Atomicity:    FS commit happens BEFORE DB commit          │
│                   If DB fails → ROLLBACK (FS unchanged)       │
│                                                                 │
│  2️⃣ Extraction Boundary                                         │
│     Filesystem:   Write data/** + .extract.audit.json        │
│     DuckDB TX:    BULK INSERT extracted_files (Arrow)         │
│     Atomicity:    FS writes finalized BEFORE DB TX             │
│                   If DB fails → ROLLBACK (FS already written) │
│                                                                 │
│  3️⃣ Validation Boundary                                         │
│     Filesystem:   Write validation/*.json (optional)          │
│     DuckDB TX:    INSERT INTO validations (...)               │
│     Atomicity:    No FS ops; just DB insert                   │
│                                                                 │
│  4️⃣ Set Latest Boundary                                         │
│     Filesystem:   Write LATEST.json (tmp → rename)            │
│     DuckDB TX:    UPSERT latest_pointer                        │
│     Atomicity:    LATEST.json write + DB commit together       │
│                   If either fails → ROLLBACK                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diagram 3: Polars Analytics Pipeline (Pillar 6)

```
┌──────────────────────────────────────────────────────────────┐
│              Polars Analytics Engine (Pillar 6)              │
│                                                              │
│  Data Sources:                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ DuckDB Catalog (metadata)                              │ │
│  │  SQL: SELECT * FROM extracted_files                    │ │
│  │  ├─ duckdb.execute(sql).arrow() → Arrow table         │ │
│  │  └─ pl.from_arrow() → LazyFrame                        │ │
│  │                                                         │ │
│  │ Audit JSON files (.extract.audit.json)                │ │
│  │  ├─ pl.scan_ndjson(path) → LazyFrame                 │ │
│  │  └─ collect(streaming=True) for large files           │ │
│  │                                                         │ │
│  │ Event Logs (Parquet or JSONL)                          │ │
│  │  └─ pl.scan_parquet(path) with predicate pushdown     │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Pipeline Builders (Zero-Loop, Lazy)                   │ │
│  │                                                         │ │
│  │  def scan_audit_json(path: Path) → LazyFrame:         │ │
│  │    lf = pl.scan_ndjson(path)                          │ │
│  │    return (lf                                          │ │
│  │      .select(['path_rel', 'size', 'sha256', 'format'])│ │
│  │      .with_columns([                                   │ │
│  │        pl.col('size').cast(pl.Int64),                 │ │
│  │        pl.col('format').cast(pl.Categorical)          │ │
│  │      ])                                                │ │
│  │    )                                                    │ │
│  │                                                         │ │
│  │  def fetch_version_arrow(db, version_id) → Arrow:     │ │
│  │    sql = f"SELECT * FROM extracted_files ..."         │ │
│  │    return db.execute(sql).arrow()                      │ │
│  │                                                         │ │
│  │  → All pipelines start with scan_* (lazy evaluation)  │ │
│  │  → Predicates pushed down to storage                  │ │
│  │  → Only collect when final output needed               │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Report Builders (Return DataFrames)                    │ │
│  │                                                         │ │
│  │ Report 1: Latest Version Summary                       │ │
│  │  Inputs: extracted_files + optional validations       │ │
│  │  Outputs:                                              │ │
│  │    • files_by_format (format, count, size_bytes)      │ │
│  │    • top_10_largest_files (path, size_bytes)          │ │
│  │    • pass_fail_rates (validator, passed, failed, %)   │ │
│  │    • total_bytes (single row)                         │ │
│  │                                                         │ │
│  │ Report 2: Version Growth (A → B Delta)                │ │
│  │  Inputs: extracted_files[version_a], extracted_files[v_b]
│  │  Outputs:                                              │ │
│  │    • added_files (path, size_bytes)                   │ │
│  │    • removed_files (path, size_bytes)                 │ │
│  │    • modified_files (path, old_size, new_size, delta) │ │
│  │    • renamed_files (old_path → new_path, size)        │ │
│  │    • summary (added_count, removed_count, churn_bytes)│ │
│  │                                                         │ │
│  │ Report 3: Validation Health                           │ │
│  │  Inputs: validations join extracted_files             │ │
│  │  Outputs:                                              │ │
│  │    • per_validator_rates (validator, pass%, fail%, avg_ms)
│  │    • fixed_issues (path, prev_status → pass)          │ │
│  │    • regressed_issues (path, prev_pass → fail)        │ │
│  │    • new_issues (path, first_fail)                    │ │
│  │                                                         │ │
│  │ Report 4: Hotspots                                    │ │
│  │  Inputs: extracted_files (optional: events, validators)
│  │  Outputs:                                              │ │
│  │    • top_N_by_bytes (path, size_bytes) [Pareto]       │ │
│  │    • top_N_by_failures (path, fail_count) [Pareto]    │ │
│  │    • format_distribution (format, count, bytes, %)    │ │
│  │    • format_failure_rates (format, fail%)             │ │
│  │                                                         │ │
│  │  → All return DataFrames w/ consistent schemas        │ │
│  │  → Typed columns (Int64, Float64, Categorical, UTC)  │ │
│  │  → Sorted for reproducibility                         │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Renderers (DataFrame → Output Format)                 │ │
│  │                                                         │ │
│  │  def as_table(df) → str:                              │ │
│  │    # Rich ASCII table with colors + alignment          │ │
│  │    # auto-format numbers (K, M, G suffix)              │ │
│  │    # handle wide tables w/ wrapping                    │ │
│  │                                                         │ │
│  │  def as_json(df) → str:                               │ │
│  │    # JSONL (one row per line) or compact JSON         │ │
│  │    # Preserve types + nulls correctly                  │ │
│  │                                                         │ │
│  │  def as_parquet(df, path: Path) → None:               │ │
│  │    # Direct DataFrame → Parquet (binary)               │ │
│  │    # For dashboard ingestion                           │ │
│  │                                                         │ │
│  │  → All renderers preserve data fidelity               │ │
│  │  → No truncation or lossy compression                  │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ CLI: `ontofetch report <type> [options]`              │ │
│  │                                                         │ │
│  │ $ ontofetch report latest \                           │ │
│  │   --version <v> --format table|json|parquet \         │ │
│  │   [--profile]                                          │ │
│  │   → Latest Summary DataFrame → render                 │ │
│  │                                                         │ │
│  │ $ ontofetch report growth \                           │ │
│  │   --a <v1> --b <v2> \                                │ │
│  │   --by format|path --top 50 \                         │ │
│  │   [--profile]                                          │ │
│  │   → Growth Report → render                            │ │
│  │                                                         │ │
│  │ $ ontofetch report validation \                       │ │
│  │   --version <v> --by validator|format \               │ │
│  │   [--profile]                                          │ │
│  │   → Health Report → render                            │ │
│  │                                                         │ │
│  │ $ ontofetch report hotspots \                         │ │
│  │   --version <v> --top 20 \                            │ │
│  │   [--profile]                                          │ │
│  │   → Hotspots Report → render                          │ │
│  │                                                         │ │
│  │ [--profile] option: Print lazy plan (EXPLAIN output)  │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│      Output:  table | json | parquet                       │
│                                                              │
│  Performance Targets:                                      │
│  • Report latest (100k files): < 5s                        │
│  • Report growth (2 versions, 1M files each): < 10s        │
│  • Streaming audit scan (1M rows): < 10s                   │
│  • Validation report (500k validations): < 8s              │
│                                                              │
│  Zero-Loop Philosophy:                                     │
│  ✓ No Python for-loops; all SQL/Polars                     │
│  ✓ Lazy evaluation; predicate pushdown to source           │
│  ✓ Streaming for large files (> 1M rows)                   │
│  ✓ Efficient memory usage via chunking                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Diagram 4: Integration Points Summary

```
┌─────────────────────────────────────────────────────────────┐
│           Integration Points (Pillar 5 + 6 + Existing)      │
│                                                             │
│  Layer 1: Existing (Phases 5.5–5.9) ✅ DO NOT MODIFY      │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ • network/ (HTTP + Hishel + polite headers)           │ │
│  │ • ratelimit/ (multi-window token buckets)             │ │
│  │ • policy/ (gates + error handling)                    │ │
│  │ • io/network.py (download_stream)                     │ │
│  │ • planning.py (fetch_all orchestration)               │ │
│  │ • manifests.py (write_lockfile)                       │ │
│  └───────────────────────────────────────────────────────┘ │
│         │                                                   │
│         ▼                                                   │
│  Layer 2: Boundaries (NEW Pillar 5A+5C) ⏳ IMPLEMENT       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Insertion Points (decorate with TX wrappers):          │ │
│  │                                                        │ │
│  │ 1. planning.fetch_all() entry:                        │ │
│  │    db = Database(config)                              │ │
│  │    db.bootstrap()  # Idempotent migrations            │ │
│  │                                                        │ │
│  │ 2. After io.network.download_stream() success:       │ │
│  │    @db.download_boundary()                           │ │
│  │    INSERT artifacts + emit event                      │ │
│  │                                                        │ │
│  │ 3. After io.extraction_throughput() success:         │ │
│  │    @db.extraction_boundary()                         │ │
│  │    BULK INSERT extracted_files (Arrow) + audit JSON  │ │
│  │                                                        │ │
│  │ 4. After validation.run_validators() success:        │ │
│  │    @db.validation_boundary()                         │ │
│  │    INSERT validations + emit event                    │ │
│  │                                                        │ │
│  │ 5. After manifests.write_lockfile():                │ │
│  │    @db.set_latest_boundary()                        │ │
│  │    UPSERT latest_pointer + write LATEST.json         │ │
│  └───────────────────────────────────────────────────────┘ │
│         │                                                   │
│         ▼                                                   │
│  Layer 3: Operations (NEW Pillar 5B) ⏳ IMPLEMENT          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ CLI Commands (add to cli.py):                          │ │
│  │                                                        │ │
│  │ 1. doctor:                                             │ │
│  │    db.doctor(dry_run=True) → print actions            │ │
│  │    db.doctor(dry_run=False) → apply + audit           │ │
│  │                                                        │ │
│  │ 2. prune:                                              │ │
│  │    db.list_orphans() → candidates                     │ │
│  │    db.prune(dry_run=True) → preview                   │ │
│  │    db.prune(dry_run=False) → delete + record          │ │
│  │                                                        │ │
│  │ 3. backup/restore:                                     │ │
│  │    db.backup(target_dir) → snapshot                   │ │
│  │    db.restore(source_dir) → restore + migrate         │ │
│  └───────────────────────────────────────────────────────┘ │
│         │                                                   │
│         ▼                                                   │
│  Layer 4: Analytics (NEW Pillar 6A+6B) ⏳ IMPLEMENT        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ New CLI Subcommand: `report`                           │ │
│  │                                                        │ │
│  │ Insertion Points (add to cli.py):                     │ │
│  │                                                        │ │
│  │ 1. List reports:                                       │ │
│  │    from analytics.reports import (                    │ │
│  │      report_latest_summary,                           │ │
│  │      report_version_growth,                           │ │
│  │      report_validation_health,                        │ │
│  │      report_hotspots                                  │ │
│  │    )                                                   │ │
│  │                                                        │ │
│  │ 2. Render output:                                      │ │
│  │    from analytics.renderers import (                  │ │
│  │      render_as_table,                                 │ │
│  │      render_as_json,                                  │ │
│  │      render_as_parquet                                │ │
│  │    )                                                   │ │
│  │                                                        │ │
│  │ 3. CLI pattern:                                        │ │
│  │    df = report_latest_summary(db, version_id)         │ │
│  │    output = render_as_table(df)                       │ │
│  │    print(output)  OR  write to file                   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Data Flow Summary:                                       │
│                                                             │
│  FS Blobs          DuckDB Metadata    Polars Analytics    │
│  ────────────      ────────────────    ──────────────     │
│  data/**           versions            Latest Summary     │
│  archives          artifacts           Growth Report      │
│  .audit.json   ←→  extracted_files ←→  Validation Health │
│  LATEST.json       validations         Hotspots           │
│  ...               events              ...                │
│                    plans                                 │
│                    plans_diffs                           │
│                                                             │
│  Invariants Maintained:                                  │
│  ✓ FS is source-of-truth for blobs                       │
│  ✓ DB is source-of-truth for metadata + lineage          │
│  ✓ Polars reads DB/audit; never modifies                 │
│  ✓ All writes atomic or guarded with rollback            │ │
│  ✓ Doctor/Prune are safe reconciliation tools            │ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

1. **Two-Phase Choreography**: Filesystem write → DB transaction (guard with lock)
2. **Idempotence via Hashing**: Content-based keys (sha256) prevent duplicates
3. **Zero Loops**: All analytics use LazyFrame + SQL; no Python loops
4. **Determinism**: Audit JSON + config_hash enable replay and debugging
5. **Safe Operations**: Doctor/Prune always dry-run first; `--apply` guarded
6. **Rich Observability**: Events table captures all operation flows
7. **Graceful Degradation**: `--continue-on-error` lets partial runs succeed

---

*End of Architecture Diagrams*
