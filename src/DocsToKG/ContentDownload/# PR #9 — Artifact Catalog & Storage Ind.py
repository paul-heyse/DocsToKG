# PR #9 — Artifact Catalog & Storage Index

**(Content hashing, deduplication, retention, GC, and S3-ready storage layout)**

> Paste this into `docs/pr9-artifact-catalog.md` (or your PR description).
> This PR introduces a **persistent artifact catalog** and a **storage index** that records every successful download (path, bytes, MIME, SHA-256), enables **deduplication**, **retention & garbage collection**, and **fast lookups** by artifact id, URL, resolver, or content hash.
> It integrates cleanly with the existing pipeline, hishel cache, telemetry, and the PR #8 orchestrator — with explicit DI and no globals.

---

## Goals

1. Add a **catalog database** (SQLite by default; Postgres-ready) to register **every successful outcome** with strong metadata & provenance.
2. Compute and persist **SHA-256** for dedup and verification (streaming-friendly).
3. Support **content-addressable storage** (CAS) option and conventional “policy path” layout; allow **S3** later without code churn.
4. CLI tools: **import-from-manifest**, **show / search / where**, **dedup report**, **verify hashes**, **gc** orphaned files, **retention** (age/size policies).
5. Observability: counters for **dedup hits**, **GC removals**, **verification failures**.

No change to user-facing telemetry tokens; only **additive** metrics and a new catalog.

---

## New/updated file tree

```text
src/DocsToKG/ContentDownload/
  catalog/
    __init__.py
    schema.sql                # NEW: DB schema for SQLite/Postgres
    models.py                 # NEW: dataclasses for Records
    store.py                  # NEW: CatalogStore (SQLite) + protocol
    fs_layout.py              # NEW: Local FS path strategies (CAS vs policy path)
    s3_layout.py              # NEW: S3 object layout adapter (stub-ready)
    gc.py                     # NEW: Retention/GC utilities
    migrate.py                # NEW: Import from manifest.jsonl into catalog
  config/
    models.py                 # MOD: StorageConfig and CatalogConfig
  download_execution.py       # MOD: compute sha256 (stream or post), return/path
  pipeline.py                 # MOD: on success, register in CatalogStore
  bootstrap.py                # MOD: build CatalogStore + FS layout from config and inject
  cli/
    app.py                    # MOD: new command group 'catalog' (show/search/verify/dedup/gc/import)
  telemetry/
    otel.py                   # MOD: metrics for dedup_hits_total, gc_removed_total, verify_failures_total
tests/
  contentdownload/
    test_catalog_register.py  # NEW: register, lookup by id/url/hash
    test_dedup_and_layout.py  # NEW: CAS path reuse and hardlink/copy behavior
    test_catalog_verify.py    # NEW: hash verification failures
    test_catalog_gc.py        # NEW: GC removes orphans; retention windows honored
    test_cli_catalog.py       # NEW: smoke tests for CLI
```

---

## 1) Config additions

```python
# src/DocsToKG/ContentDownload/config/models.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional

class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["fs", "s3"] = "fs"
    root_dir: str = "data/docs"                # base for final files (fs) or local staging for s3
    layout: Literal["policy_path", "cas"] = "policy_path"
    cas_prefix: str = "sha256"                 # e.g., data/cas/sha256/xx/...
    hardlink_dedup: bool = True                # if FS supports hardlinks
    s3_bucket: Optional[str] = None            # for backend="s3"
    s3_prefix: str = "docs/"
    s3_storage_class: str = "STANDARD"         # optional

class CatalogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["sqlite"] = "sqlite"      # Postgres-ready; start with sqlite
    path: str = "state/catalog.sqlite"
    wal_mode: bool = True
    # verify/hashing options
    compute_sha256: bool = True                # compute on success
    verify_on_register: bool = False           # re-open final file & verify sha after move
    # retention
    retention_days: int = 0                    # 0 = disabled; else GC older than X days (optional CLI)
    orphan_ttl_days: int = 7                   # files not referenced for X days are eligible for GC

class ContentDownloadConfig(BaseModel):
    # ...
    storage: StorageConfig = Field(default_factory=StorageConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
```

---

## 2) Database schema (SQLite; Postgres-ready)

> File: `src/DocsToKG/ContentDownload/catal og/schema.sql`

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id TEXT NOT NULL,               -- e.g., doi:..., url:..., any unique identifier
  source_url  TEXT NOT NULL,
  resolver    TEXT NOT NULL,
  content_type TEXT,
  bytes       INTEGER NOT NULL,
  sha256      TEXT,                        -- lowercase hex; may be NULL if disabled
  storage_uri TEXT NOT NULL,               -- "file:///.../path" or "s3://bucket/key"
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  run_id      TEXT                         -- provenance
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_unique
  ON documents(artifact_id, source_url, resolver);

CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_ct  ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id);

-- Optional table to model multiple variants per artifact (e.g., pdf/html/supplement)
CREATE TABLE IF NOT EXISTS variants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  variant TEXT NOT NULL,                   -- e.g., "pdf", "html", "supp"
  storage_uri TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  content_type TEXT,
  sha256 TEXT,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_variants_unique
  ON variants(document_id, variant);
```

Design notes:

* The **unique** composite `(artifact_id, source_url, resolver)` ensures idempotence (same resolver+url won’t duplicate records).
* **`sha256`** is optional (config). When enabled, use it for **dedup report** and **CAS** linking.
* `storage_uri` abstracts **fs** vs **s3** (use `file:///` and `s3://` URIs).

---

## 3) Catalog interfaces & models

```python
# src/DocsToKG/ContentDownload/catalog/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: int
    artifact_id: str
    source_url: str
    resolver: str
    content_type: Optional[str]
    bytes: int
    sha256: Optional[str]
    storage_uri: str
    created_at: datetime
    updated_at: datetime
    run_id: Optional[str]
```

```python
# src/DocsToKG/ContentDownload/catalog/store.py
import sqlite3
from pathlib import Path
from typing import Optional, Iterable
from datetime import datetime
from .models import DocumentRecord

class CatalogStore:
    """Protocol-ish base for injection."""
    def register_or_get(self, *, artifact_id: str, source_url: str, resolver: str,
                        content_type: Optional[str], bytes: int, sha256: Optional[str],
                        storage_uri: str, run_id: Optional[str]) -> DocumentRecord: ...
    def get_by_artifact(self, artifact_id: str) -> list[DocumentRecord]: ...
    def get_by_sha256(self, sha256: str) -> list[DocumentRecord]: ...
    def find_duplicates(self) -> list[tuple[str, int]]: ...  # (sha256, count)
    def verify(self, record_id: int) -> bool: ...
    def stats(self) -> dict: ...

class SQLiteCatalog(CatalogStore):
    def __init__(self, path: str, wal_mode: bool = True):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        if wal_mode: self.conn.execute("PRAGMA journal_mode=WAL")
        # load schema.sql
        schema = (Path(__file__).parent/"schema.sql").read_text()
        self.conn.executescript(schema); self.conn.commit()

    def register_or_get(self, **kw) -> DocumentRecord:
        now = datetime.utcnow().isoformat(timespec="seconds")
        try:
            self.conn.execute(
                """INSERT INTO documents (artifact_id, source_url, resolver, content_type, bytes, sha256, storage_uri, created_at, updated_at, run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (kw["artifact_id"], kw["source_url"], kw["resolver"], kw["content_type"],
                 kw["bytes"], kw["sha256"], kw["storage_uri"], now, now, kw["run_id"])
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Already present — fetch it
            pass
        cur = self.conn.execute(
            """SELECT id, artifact_id, source_url, resolver, content_type, bytes, sha256, storage_uri,
                      created_at, updated_at, run_id
               FROM documents
               WHERE artifact_id=? AND source_url=? AND resolver=?""",
            (kw["artifact_id"], kw["source_url"], kw["resolver"])
        )
        row = cur.fetchone()
        return DocumentRecord(
            id=row[0], artifact_id=row[1], source_url=row[2], resolver=row[3],
            content_type=row[4], bytes=row[5], sha256=row[6], storage_uri=row[7],
            created_at=datetime.fromisoformat(row[8]), updated_at=datetime.fromisoformat(row[9]),
            run_id=row[10]
        )

    def get_by_artifact(self, artifact_id: str) -> list[DocumentRecord]: ...
    def get_by_sha256(self, sha256: str) -> list[DocumentRecord]: ...
    def find_duplicates(self) -> list[tuple[str,int]]: ...
    def verify(self, record_id: int) -> bool: ...
    def stats(self) -> dict: ...
```

---

## 4) Storage layouts (FS + CAS; S3-ready)

```python
# src/DocsToKG/ContentDownload/catalog/fs_layout.py
from __future__ import annotations
import os, pathlib, hashlib, shutil

def cas_path(root_dir: str, sha256_hex: str) -> str:
    # e.g., data/cas/sha256/ab/cdef... (two-level dirs for fan-out)
    p = pathlib.Path(root_dir) / "cas" / "sha256" / sha256_hex[:2] / sha256_hex[2:]
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

def policy_path(root_dir: str, *, artifact_id: str, url_basename: str) -> str:
    # simple default policy; you may already have a policy path generator
    p = pathlib.Path(root_dir) / url_basename
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

def dedup_hardlink_or_copy(src_tmp: str, dst_final: str, hardlink: bool = True) -> None:
    if hardlink:
        try:
            os.link(src_tmp, dst_final); os.remove(src_tmp); return
        except OSError:
            pass
    # fallback: move/replace
    if os.path.exists(dst_final):
        os.remove(dst_final)
    shutil.move(src_tmp, dst_final)
```

```python
# src/DocsToKG/ContentDownload/catalog/s3_layout.py
# Low-effort seam for later: expose put_file(), build_uri(), and optional existence checks.
```

---

## 5) Pipeline integration (register on success)

### 5.1 finalize phase: compute sha256 & decide final path

* If `cfg.catalog.compute_sha256`: compute streaming (preferred) or after write.
* If `cfg.storage.layout == "cas"` and `sha256` is available → compute **CAS path** under `root_dir/cas/sha256/xx/...`.
* Else → use existing policy path (as today).
* If CAS is in use and `hardlink_dedup=True`:

  * If CAS file already exists (same hash) → **hardlink** to policy path or keep single CAS file and record *that* path; configurable.
* Atomicity preserved: write to tmp, fsync, rename.

### 5.2 register to Catalog

```python
# src/DocsToKG/ContentDownload/download_execution.py (snippet)
from DocsToKG.ContentDownload.catalog.fs_layout import cas_path, policy_path, dedup_hardlink_or_copy

def finalize_candidate_download(..., *, catalog=None, cfg_storage=None, cfg_catalog=None, ...):
    # choose final_path based on cfg_storage.layout
    sha = None
    if cfg_catalog.compute_sha256:
        sha = _sha256_file(stream.path_tmp)     # or stream-time computation buffer
    if cfg_storage.layout == "cas" and sha:
        final_path = cas_path(cfg_storage.root_dir, sha)
        dedup_hardlink_or_copy(stream.path_tmp, final_path, hardlink=cfg_storage.hardlink_dedup)
    else:
        final_path = policy_path(cfg_storage.root_dir, artifact_id=artifact.id, url_basename=_basename(plan.url))
        _atomic_move(stream.path_tmp, final_path)

    # Optional verify
    if cfg_catalog.verify_on_register and sha:
        assert sha == _sha256_file(final_path), "sha mismatch after finalize"

    # Register record
    if catalog is not None:
        storage_uri = f"file://{final_path}"     # or s3://bucket/key later
        catalog.register_or_get(
            artifact_id=artifact.id, source_url=plan.url, resolver=plan.resolver_name,
            content_type=stream.content_type, bytes=stream.bytes_written,
            sha256=sha, storage_uri=storage_uri, run_id=run_id
        )

    # outcome
    return DownloadOutcome(ok=True, classification="success", path=final_path,
                           meta={"content_type": stream.content_type, "bytes": stream.bytes_written, "sha256": sha})
```

> `bootstrap.py` should construct and inject `catalog` & `cfg_storage/cfg_catalog` into the pipeline (similar to how we injected hishel/telemetry).

---

## 6) CLI: catalog tools

```python
# src/DocsToKG/ContentDownload/cli/app.py
catalog_app = typer.Typer(help="Artifact Catalog")
app.add_typer(catalog_app, name="catalog")

@catalog_app.command("import-manifest")
def import_manifest(config: Optional[str] = typer.Option(None, "--config", "-c"),
                    path: str = typer.Argument(..., help="manifest.jsonl from older runs")):
    """One-time import: for each success line, compute sha256 (optional) and register."""
    ...

@catalog_app.command("show")
def show(artifact_id: str, config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Print records for an artifact_id."""
    ...

@catalog_app.command("where")
def where(sha256: str, config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Find all records with a given sha256."""
    ...

@catalog_app.command("dedup-report")
def dedup_report(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """List sha256 values with count > 1."""
    ...

@catalog_app.command("verify")
def verify(record_id: int, config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Compute sha256 of current storage_uri and compare with catalog."""
    ...

@catalog_app.command("gc")
def gc(config: Optional[str] = typer.Option(None, "--config", "-c"),
       dry_run: bool = typer.Option(True, "--dry-run/--apply"),
       orphan_days: Optional[int] = typer.Option(None, "--orphans", help="days since last seen (override config)")):
    """Remove unreferenced files under root_dir (or in S3) with age > orphan_days."""
    ...
```

---

## 7) GC & Retention

```python
# src/DocsToKG/ContentDownload/catalog/gc.py
from datetime import datetime, timedelta
from pathlib import Path

def find_orphans(root_dir: str, referenced_paths: set[str]) -> list[str]:
    # Walk root_dir; compare against referenced file list; return extras
    ...

def retention_filter(records: list[DocumentRecord], days: int) -> list[DocumentRecord]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    return [r for r in records if r.created_at < cutoff]
```

* **GC** removes files **not referenced** by any catalog record (plus optional age filter).
* **Retention** is catalog-level (delete records older than retention and their files). Do it **opt-in**.

---

## 8) Observability

New OTel metrics:

* `contentdownload_dedup_hits_total{resolver}` — count when CAS/hardlink avoided duplicate write.
* `contentdownload_gc_removed_total` — files removed by GC.
* `contentdownload_verify_failures_total` — hash mismatches.

Add log lines in CSV? No — we keep attempts CSV about network/download; catalog ops are better in metrics + CLI.

---

## 9) Tests

1. **Register & lookup**

   * Create a fake successful stream (temp file, known bytes & MIME).
   * Finalize → CAS path chosen; catalog record created.
   * `get_by_artifact`, `get_by_sha256` return expected rows.

2. **Dedup (hardlink or copy)**

   * Second identical file → detect CAS target exists; `hardlink_dedup=True` → create hardlink; assert inode equality on POSIX.
   * `hardlink_dedup=False` → copy/replace; sizes match.

3. **Verify**

   * Tamper with file; `catalog verify record_id` returns False and increments `verify_failures_total`.

4. **GC**

   * Create orphan file under root_dir; run `gc --apply` → file removed; metric increments.

5. **CLI smoke**

   * `catalog import-manifest` loads a minimal JSONL and creates rows.
   * `catalog where <sha>` prints entries.
   * `catalog dedup-report` prints duplicates.

---

## 10) Acceptance checklist

* [ ] Success outcomes register in catalog with `(artifact_id, url, resolver)` idempotence.
* [ ] SHA-256 computation integrated; CAS or policy path used per config; atomic writes preserved.
* [ ] Dedup works (hardlink/copy based on config); **no extra download** or re-stream required.
* [ ] CLI supports **show/where/verify/dedup-report/gc/import-manifest**.
* [ ] OTel metrics for dedup, GC, verify failures.
* [ ] Tests green; behavior unchanged for users who disable the catalog (feature-flag by setting `catalog.compute_sha256=False` and avoiding registration).

---

## 11) Migration & compatibility

* Existing **manifest.jsonl** remains the audit log; catalog is the **queryable index**.
* Provide a one-time `catalog import-manifest` to backfill older runs into the DB.
* Paths in catalog are **URIs**; switch to S3 later without record shape changes.

---

## 12) Design notes & guardrails

* **Compute SHA at finalize**, not earlier: cheapest and guaranteed stable after **atomic move**.
* **CAS layout** keeps one canonical copy per hash; you can either **point** policy paths at CAS (symlink/hardlink) or drop policy paths altogether and let clients resolve via catalog.
* **No URL labels in metrics**: as always, keep labels low-cardinality (`resolver`, maybe `content_type`).
* **Safe defaults**: `hardlink_dedup=True` on POSIX; flip to `False` for Windows or cross-filesystem moves.

---

## 13) Extension ideas

* **Postgres** backend: swap `SQLiteCatalog` with `PGCatalog` (SQL is simple).
* **Variant support**: populate `variants` for HTML/supplement; multiple rows per artifact.
* **Manifest v2**: enrich final manifest to include catalog `document_id` for easy joins.
* **S3 backend**: stream final file to S3 and keep **local CAS** as staging (optional).
* **Content signatures**: add **MIME sniff**, **PDF linearization check**, **text extraction preview** for indexing.
* **Provenance chain**: add a `sources` table capturing resolver ladder (who found the plan) for deep lineage.

---

### Minimal “diff guide” (what changes where)

* **Config**: add `storage` + `catalog` sections, defaults off for intrusive bits (verify/retention).
* **Bootstrap**: create `SQLiteCatalog` and pass `(catalog, cfg.storage, cfg.catalog)` into pipeline finalize.
* **Finalize**: compute SHA-256; choose path (CAS or policy path); register in catalog; build outcome meta.
* **CLI**: add `catalog` subcommands.
* **Telemetry**: increment dedup/gc/verify metrics when triggered.

---

### TL;DR

With PR #9, you get a **first-class artifact index** and a clean **storage abstraction** that makes dedup, verification, and retention **operationally trivial** — without touching your resolver/HTTP/hishel stack.
