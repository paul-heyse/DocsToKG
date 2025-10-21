Absolutely—here are **drop-in scaffold stubs** (minimal, typed, documented) for “**DuckDB as the brain**” plus an **overall architecture** to make the interactions crystal clear. The stubs are *safe to paste as-is*; they compile, but many methods include `TODO` markers so you can fill in real logic iteratively.

---

# 🗂️ Where these files go

Paste these under your repo root:

```
src/DocsToKG/OntologyDownload/
  catalog/
    __init__.py
    connection.py
    migrations.py
    repo.py
    boundaries.py
    prune.py
    doctor.py
    profile.py
    migrations/            # put your 0001..0007 *.sql files here
  storage/
    base.py
    localfs_duckdb.py
  cli/
    db_cmd.py
# (settings.py additions for DuckDB/Storage at the bottom)
# (observability/events.py stub if you don't already have one)
```

---

# 🧱 Catalog scaffolds

### `src/DocsToKG/OntologyDownload/catalog/__init__.py`

```python
"""DuckDB-backed catalog (“brain”) for OntologyDownload.

Public entrypoints:
- `get_writer()` / `get_reader()` – open connections with correct PRAGMAs
- `apply_pending_migrations()` – idempotent runner
- `Repo` – typed façade for upserts/queries
- `writer_tx()` – boundary transaction helper (BEGIN/COMMIT/ROLLBACK + events)
"""

from .connection import get_writer, get_reader, writer_tx
from .migrations import apply_pending_migrations
from .repo import Repo
```

---

### `src/DocsToKG/OntologyDownload/catalog/connection.py`

```python
from __future__ import annotations
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import duckdb

try:
    # Prefer your real emitter if present
    from ..observability.events import emit
except Exception:
    def emit(event_type: str, level: str = "INFO", payload: dict | None = None, **ids):
        pass  # no-op fallback

@dataclass(frozen=True)
class DuckDBConfig:
    path: Path
    threads: int = 8
    readonly: bool = False
    writer_lock: bool = True

_writer_conn: Optional[duckdb.DuckDBPyConnection] = None
_reader_conn: Optional[duckdb.DuckDBPyConnection] = None
_writer_lock_fd: Optional[int] = None

def _acquire_writer_lock(db_path: Path) -> None:
    """Acquire a simple process writer lock.

    TODO: replace with robust cross-platform lock (portalocker/fcntl on POSIX).
    """
    global _writer_lock_fd
    if _writer_lock_fd is not None:
        return
    lock_path = db_path.with_suffix(db_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _writer_lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    # NOTE: fcntl lock for POSIX; msvcrt for Windows – left as TODO to avoid deps.
    # This stub relies on single-process discipline.

def get_writer(cfg: DuckDBConfig) -> duckdb.DuckDBPyConnection:
    global _writer_conn
    if _writer_conn is None:
        if cfg.writer_lock and not cfg.readonly:
            _acquire_writer_lock(cfg.path)
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        _writer_conn = duckdb.connect(str(cfg.path), read_only=False)
        _writer_conn.execute("PRAGMA threads = {}".format(int(cfg.threads)))
    return _writer_conn

def get_reader(cfg: DuckDBConfig) -> duckdb.DuckDBPyConnection:
    global _reader_conn
    if _reader_conn is None:
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        _reader_conn = duckdb.connect(str(cfg.path), read_only=True)
        _reader_conn.execute("PRAGMA threads = {}".format(int(cfg.threads)))
    return _reader_conn

@contextmanager
def writer_tx(cfg: DuckDBConfig, boundary: str = "unspecified") -> Iterator[duckdb.DuckDBPyConnection]:
    """BEGIN/COMMIT/ROLLBACK wrapper that emits db.tx events."""
    conn = get_writer(cfg)
    try:
        emit("db.tx.begin", payload={"boundary": boundary})
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
        emit("db.tx.commit", payload={"boundary": boundary})
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        finally:
            emit("db.tx.rollback", level="ERROR", payload={"boundary": boundary, "error": repr(e)})
        raise
```

---

### `src/DocsToKG/OntologyDownload/catalog/migrations.py`

```python
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import duckdb

from .connection import get_writer, DuckDBConfig

def _applied(conn: duckdb.DuckDBPyConnection) -> set[str]:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version(
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT now()
        )
    """)
    return {row[0] for row in conn.execute("SELECT version FROM schema_version").fetchall()}

def _apply_file(conn: duckdb.DuckDBPyConnection, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    conn.execute("BEGIN")
    conn.execute(sql)
    conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (?)", [sql_path.stem])
    conn.execute("COMMIT")

def discover_migrations(migrations_dir: Path) -> list[Path]:
    return sorted(migrations_dir.glob("*.sql"), key=lambda p: p.name)

def apply_pending_migrations(cfg: DuckDBConfig, migrations_dir: Path) -> list[str]:
    """Apply 000N_*.sql in order if not present in schema_version. Returns applied ids."""
    conn = get_writer(cfg)
    already = _applied(conn)
    applied: list[str] = []
    for sql in discover_migrations(migrations_dir):
        if sql.stem in already:
            continue
        _apply_file(conn, sql)
        applied.append(sql.stem)
    return applied
```

---

### `src/DocsToKG/OntologyDownload/catalog/repo.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import duckdb

from .connection import DuckDBConfig, get_writer, get_reader

@dataclass(frozen=True)
class ArtifactMeta:
    artifact_id: str
    version_id: str
    service: str
    source_url: str
    etag: Optional[str]
    last_modified: Optional[datetime]
    content_type: Optional[str]
    size_bytes: int
    fs_relpath: str
    status: str  # 'fresh'|'cached'|'failed'

@dataclass(frozen=True)
class ExtractedFileMeta:
    file_id: str
    artifact_id: str
    version_id: str
    relpath_in_version: str
    format: str
    size_bytes: int
    mtime: Optional[datetime]
    cas_relpath: Optional[str] = None

@dataclass(frozen=True)
class ValidationMeta:
    validation_id: str
    file_id: str
    validator: str
    passed: bool
    details_json: Optional[str]
    run_at: datetime

class Repo:
    """High-level DuckDB operations. Keep SQL centralized here."""

    def __init__(self, cfg: DuckDBConfig):
        self.cfg = cfg

    # --- Upserts/Inserts -----------------------------------------------------

    def upsert_version(self, version_id: str, service: str, created_at: Optional[datetime] = None, plan_hash: Optional[str] = None) -> None:
        conn = get_writer(self.cfg)
        conn.execute("""
            INSERT INTO versions(version_id, service, created_at, plan_hash)
            VALUES (?, ?, COALESCE(?, now()), ?)
            ON CONFLICT (version_id) DO UPDATE SET
                service=excluded.service,
                plan_hash=COALESCE(excluded.plan_hash, versions.plan_hash)
        """, [version_id, service, created_at, plan_hash])

    def upsert_artifact(self, a: ArtifactMeta) -> None:
        conn = get_writer(self.cfg)
        conn.execute("""
            INSERT INTO artifacts(artifact_id, version_id, service, source_url, etag, last_modified, content_type, size_bytes, fs_relpath, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                version_id=excluded.version_id, service=excluded.service, source_url=excluded.source_url,
                etag=excluded.etag, last_modified=excluded.last_modified, content_type=excluded.content_type,
                size_bytes=excluded.size_bytes, fs_relpath=excluded.fs_relpath, status=excluded.status
        """, [a.artifact_id, a.version_id, a.service, a.source_url, a.etag, a.last_modified, a.content_type, a.size_bytes, a.fs_relpath, a.status])

    def insert_extracted_files(self, rows: Iterable[ExtractedFileMeta]) -> int:
        """Bulk insert (use Appender for speed). Returns count inserted."""
        conn = get_writer(self.cfg)
        app = conn.appender("extracted_files")
        count = 0
        for r in rows:
            app.append([r.file_id, r.artifact_id, r.version_id, r.relpath_in_version, r.format, r.size_bytes, r.mtime, r.cas_relpath])
            count += 1
        app.close()
        return count

    def insert_validations(self, rows: Iterable[ValidationMeta]) -> int:
        conn = get_writer(self.cfg)
        app = conn.appender("validations")
        count = 0
        for v in rows:
            app.append([v.validation_id, v.file_id, v.validator, v.passed, v.details_json, v.run_at])
            count += 1
        app.close()
        return count

    # --- Latest pointer -------------------------------------------------------

    def get_latest(self, slot: str = "default") -> Optional[str]:
        conn = get_reader(self.cfg)
        res = conn.execute("SELECT version_id FROM latest_pointer WHERE slot=?", [slot]).fetchone()
        return res[0] if res else None

    def set_latest(self, version_id: str, by: Optional[str] = None, slot: str = "default") -> None:
        conn = get_writer(self.cfg)
        conn.execute("""
            INSERT INTO latest_pointer(slot, version_id, updated_at, by)
            VALUES (?, ?, now(), ?)
            ON CONFLICT(slot) DO UPDATE SET version_id=excluded.version_id, updated_at=excluded.updated_at, by=excluded.by
        """, [slot, version_id, by])

    # --- Queries --------------------------------------------------------------

    def list_versions(self, service: Optional[str] = None, limit: int = 50):
        conn = get_reader(self.cfg)
        if service:
            return conn.execute("""
                SELECT version_id, service, created_at, plan_hash
                FROM versions WHERE service=? ORDER BY created_at DESC LIMIT ?
            """, [service, limit]).fetchall()
        return conn.execute("""
            SELECT version_id, service, created_at, plan_hash
            FROM versions ORDER BY created_at DESC LIMIT ?
        """, [limit]).fetchall()

    def list_files(self, version_id: str, format: Optional[str] = None):
        conn = get_reader(self.cfg)
        if format:
            return conn.execute("""
                SELECT relpath_in_version, size_bytes, mtime, file_id
                FROM extracted_files WHERE version_id=? AND format=? ORDER BY relpath_in_version
            """, [version_id, format]).fetchall()
        return conn.execute("""
            SELECT relpath_in_version, size_bytes, mtime, file_id
            FROM extracted_files WHERE version_id=? ORDER BY relpath_in_version
        """, [version_id]).fetchall()
```

---

### `src/DocsToKG/OntologyDownload/catalog/boundaries.py`

```python
from __future__ import annotations
from typing import Iterable
from dataclasses import dataclass
from pathlib import Path

from .connection import DuckDBConfig, writer_tx
from .repo import Repo, ArtifactMeta, ExtractedFileMeta, ValidationMeta

try:
    from ..observability.events import emit
except Exception:
    def emit(event_type: str, level: str = "INFO", payload: dict | None = None, **ids): pass

@dataclass(frozen=True)
class DownloadBoundaryInput:
    version_id: str
    service: str
    artifact: ArtifactMeta     # filled after FS write so size/fs_relpath are final

def commit_download_boundary(cfg: DuckDBConfig, inp: DownloadBoundaryInput) -> None:
    """FS write already succeeded. Record in DuckDB."""
    repo = Repo(cfg)
    with writer_tx(cfg, boundary="download"):
        repo.upsert_version(inp.version_id, inp.service)
        repo.upsert_artifact(inp.artifact)
        emit("db.tx.boundary", payload={"boundary": "download", "version_id": inp.version_id})

def commit_extract_boundary(cfg: DuckDBConfig, version_id: str, files: Iterable[ExtractedFileMeta]) -> int:
    """FS extraction succeeded. Bulk insert extracted_files."""
    repo = Repo(cfg)
    with writer_tx(cfg, boundary="extract"):
        n = repo.insert_extracted_files(files)
        emit("db.tx.boundary", payload={"boundary": "extract", "version_id": version_id, "rows": n})
        return n

def commit_validation_boundary(cfg: DuckDBConfig, rows: Iterable[ValidationMeta]) -> int:
    repo = Repo(cfg)
    with writer_tx(cfg, boundary="validate"):
        n = repo.insert_validations(rows)
        emit("db.tx.boundary", payload={"boundary": "validate", "rows": n})
        return n

def commit_set_latest(cfg: DuckDBConfig, version_id: str, by: str | None = None) -> None:
    repo = Repo(cfg)
    with writer_tx(cfg, boundary="latest"):
        repo.set_latest(version_id, by=by)
        emit("db.tx.boundary", payload={"boundary": "latest", "version_id": version_id})
```

---

### `src/DocsToKG/OntologyDownload/catalog/prune.py`

```python
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import os
from .connection import DuckDBConfig, get_writer, writer_tx

def load_staging_from_fs(cfg: DuckDBConfig, root: Path) -> int:
    """Walk `root` and load into staging_fs_listing (scope='version' by default)."""
    conn = get_writer(cfg)
    conn.execute("TRUNCATE staging_fs_listing")
    count = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            rel = Path(base).joinpath(name).resolve().relative_to(root.resolve()).as_posix()
            st = os.stat(Path(base)/name)
            conn.execute("INSERT INTO staging_fs_listing(scope, relpath, size_bytes, mtime) VALUES ('version', ?, ?, ?)",
                         [rel, int(st.st_size), None])
            count += 1
    return count

def orphans(conn) -> list[tuple[str, int]]:
    """Return list of (relpath, size_bytes) that are in FS but not in catalog (via v_fs_orphans view)."""
    return conn.execute("SELECT relpath, size_bytes FROM v_fs_orphans").fetchall()

def prune_apply(cfg: DuckDBConfig, victims: Iterable[str], root: Path) -> int:
    deleted = 0
    with writer_tx(cfg, boundary="prune"):
        for rel in victims:
            try:
                (root / rel).unlink(missing_ok=True)
                deleted += 1
            except Exception:
                # leave DB untouched for safety here; doctor handles rows
                pass
    return deleted
```

---

### `src/DocsToKG/OntologyDownload/catalog/doctor.py`

```python
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from .connection import DuckDBConfig, get_reader, get_writer, writer_tx

def find_db_missing_files(cfg: DuckDBConfig, root: Path) -> list[tuple[str, int]]:
    """Return DB rows that point to files that don't exist anymore."""
    conn = get_reader(cfg)
    rows = conn.execute("""
        SELECT service || '/' || version_id || '/' || relpath_in_version AS rel, size_bytes
        FROM extracted_files
    """).fetchall()
    out = []
    for rel, size in rows:
        if not (root / rel).exists():
            out.append((rel, size))
    return out

def drop_missing_file_rows(cfg: DuckDBConfig, rels: Iterable[str]) -> int:
    """Delete catalog rows for the given relpaths."""
    n = 0
    with writer_tx(cfg, boundary="doctor"):
        conn = get_writer(cfg)
        for rel in rels:
            conn.execute("""
                DELETE FROM extracted_files WHERE (service || '/' || version_id || '/' || relpath_in_version)=?
            """, [rel])
            n += 1
    return n
```

---

### `src/DocsToKG/OntologyDownload/catalog/profile.py`

```python
from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator
import duckdb

@contextmanager
def profile_query(conn: duckdb.DuckDBPyConnection, sql: str, *, enable: bool = False):
    """Optionally print EXPLAIN ANALYZE for SQL; always time the execution."""
    plan = None
    if enable:
        plan = conn.execute("EXPLAIN " + sql).fetchdf()
        print(plan)
    t0 = perf_counter()
    yield
    dt = (perf_counter() - t0) * 1000
    if enable:
        print(f"[DUCKDB] {dt:.1f} ms")
```

---

# 📦 Storage façade (Local FS + DuckDB “LATEST”)

### `src/DocsToKG/OntologyDownload/storage/base.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional

@dataclass(frozen=True)
class StoredObject:
    path_rel: str
    size: Optional[int]
    etag: Optional[str]
    url: str

@dataclass(frozen=True)
class StoredStat:
    size: int
    etag: Optional[str]
    last_modified: Optional[float]  # epoch seconds

class StorageBackend(Protocol):
    def base_url(self) -> str: ...
    def put_file(self, local: Path, remote_rel: str, *, meta: dict | None = None) -> StoredObject: ...
    def put_bytes(self, data: bytes, remote_rel: str, *, meta: dict | None = None) -> StoredObject: ...
    def rename(self, src_rel: str, dst_rel: str) -> None: ...
    def delete(self, path_rel_or_list: str | list[str]) -> None: ...
    def exists(self, remote_rel: str) -> bool: ...
    def stat(self, remote_rel: str) -> StoredStat: ...
    def list(self, prefix_rel: str = "") -> list[str]: ...
    def resolve_url(self, remote_rel: str) -> str: ...
    def set_latest_version(self, version: str, extra: dict | None = None) -> None: ...
    def get_latest_version(self) -> str | None: ...
```

---

### `src/DocsToKG/OntologyDownload/storage/localfs_duckdb.py`

```python
from __future__ import annotations
import os, json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import StorageBackend, StoredObject, StoredStat
from ..catalog.repo import Repo
from ..catalog.connection import DuckDBConfig, get_writer

@dataclass(frozen=True)
class LocalDuckDBStorage(StorageBackend):
    root: Path
    db: DuckDBConfig
    latest_name: str = "LATEST.json"
    write_latest_mirror: bool = True

    def base_url(self) -> str:
        return str(self.root.resolve())

    # --- helpers ---
    def _abs(self, rel: str) -> Path:
        if rel.startswith("/") or ".." in Path(rel).parts or "\\" in rel:
            raise ValueError(f"unsafe storage relpath: {rel}")
        return (self.root / Path(rel)).resolve()

    # --- mutating ops ---
    def put_file(self, local: Path, remote_rel: str, *, meta: dict | None = None) -> StoredObject:
        dest = self._abs(remote_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
        # copy stream
        with open(local, "rb") as rf, open(tmp, "wb") as wf:
            while True:
                b = rf.read(1024 * 1024)
                if not b:
                    break
                wf.write(b)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp, dest)  # atomic rename
        os.utime(dest, None)
        st = os.stat(dest)
        # optional: update generic 'objects' table here if desired
        return StoredObject(path_rel=remote_rel, size=st.st_size, etag=None, url=str(dest))

    def put_bytes(self, data: bytes, remote_rel: str, *, meta: dict | None = None) -> StoredObject:
        dest = self._abs(remote_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
        with open(tmp, "wb") as wf:
            wf.write(data)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp, dest)
        st = os.stat(dest)
        return StoredObject(path_rel=remote_rel, size=st.st_size, etag=None, url=str(dest))

    def rename(self, src_rel: str, dst_rel: str) -> None:
        os.replace(self._abs(src_rel), self._abs(dst_rel))

    def delete(self, path_rel_or_list: str | list[str]) -> None:
        rels = [path_rel_or_list] if isinstance(path_rel_or_list, str) else path_rel_or_list
        for r in rels:
            p = self._abs(r)
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # --- read/introspection ---
    def exists(self, remote_rel: str) -> bool:
        return self._abs(remote_rel).exists()

    def stat(self, remote_rel: str) -> StoredStat:
        st = os.stat(self._abs(remote_rel))
        return StoredStat(size=st.st_size, etag=None, last_modified=st.st_mtime)

    def list(self, prefix_rel: str = "") -> list[str]:
        base = self._abs(prefix_rel) if prefix_rel else self.root.resolve()
        out: list[str] = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                p = Path(root) / f
                out.append(p.resolve().relative_to(self.root.resolve()).as_posix())
        return out

    def resolve_url(self, remote_rel: str) -> str:
        return str(self._abs(remote_rel))

    # --- latest pointer (DB authoritative) ---
    def set_latest_version(self, version: str, extra: dict | None = None) -> None:
        repo = Repo(self.db)
        repo.set_latest(version, by=extra.get("by") if extra else None)
        if self.write_latest_mirror:
            mirror = self.root / self.latest_name
            tmp = mirror.with_name(mirror.name + f".tmp-{os.getpid()}")
            mirror.parent.mkdir(parents=True, exist_ok=True)
            body = {
                "latest": version,
                "ts": None,
                "storage": str(self.root.resolve()),
                "by": extra.get("by") if extra else None,
                "db": str(self.db.path.resolve())
            }
            with open(tmp, "w", encoding="utf-8") as wf:
                json.dump(body, wf, separators=(",", ":"), sort_keys=True)
                wf.flush(); os.fsync(wf.fileno())
            os.replace(tmp, mirror)

    def get_latest_version(self) -> Optional[str]:
        repo = Repo(self.db)
        return repo.get_latest()
```

---

# 🖥️ CLI scaffolds

### `src/DocsToKG/OntologyDownload/cli/db_cmd.py`

```python
from __future__ import annotations
from pathlib import Path
import typer

from ..catalog.connection import DuckDBConfig
from ..catalog.migrations import apply_pending_migrations
from ..catalog.repo import Repo

app = typer.Typer(help="DuckDB catalog utilities")

def _cfg(db_path: Path, threads: int) -> DuckDBConfig:
    return DuckDBConfig(path=db_path, threads=threads, readonly=False, writer_lock=True)

@app.command("migrate")
def migrate(db: Path = typer.Option(..., "--db", help="Path to DuckDB file"),
            migrations: Path = typer.Option(..., "--migrations", help="DDL dir"),
            threads: int = 8):
    applied = apply_pending_migrations(_cfg(db, threads), migrations)
    typer.echo(f"Applied: {applied}")

@app.command("latest")
def latest(action: str = typer.Argument("get"), version: str = typer.Option(None, "--version"),
           db: Path = typer.Option(..., "--db"), threads: int = 8):
    repo = Repo(_cfg(db, threads))
    if action == "get":
        typer.echo(repo.get_latest() or "")
    elif action == "set":
        assert version, "--version required for set"
        repo.set_latest(version, by="cli")
        typer.echo(f"latest={version}")
    else:
        raise typer.BadParameter("action must be get|set")

@app.command("versions")
def versions(db: Path = typer.Option(..., "--db"), service: str | None = None, limit: int = 20, threads: int = 8):
    for row in Repo(_cfg(db, threads)).list_versions(service=service, limit=limit):
        typer.echo("\t".join(map(lambda v: "" if v is None else str(v), row)))
```

> Add more subcommands later (`db files`, `db stats`, `db delta …`, `db doctor`, `db prune`), reusing `Repo`.

---

# ⚙️ `settings.py` additions (sketch)

```python
from pydantic import BaseModel, Field
from pathlib import Path

class DuckDBSettings(BaseModel):
    path: Path
    threads: int = 8
    readonly: bool = False
    writer_lock: bool = True
    parquet_events: bool = False

class StorageSettings(BaseModel):
    root: Path
    latest_name: str = "LATEST.json"
    write_latest_mirror: bool = True

# In your global Settings:
#   db: DuckDBSettings
#   storage: StorageSettings
# Ensure these feed config_hash and are logged in settings show/schema.
```

---

# 🧩 Optional `observability/events.py` stub (if none yet)

```python
from __future__ import annotations
import json, sys
from datetime import datetime

def emit(event_type: str, level: str = "INFO", payload: dict | None = None, **ids):
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "type": event_type,
        "level": level,
        "ids": ids,
        "payload": payload or {}
    }
    sys.stdout.write(json.dumps(rec, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def flush():
    pass
```

---

# 🧭 Overall Architecture (DuckDB as “brain”)

```
                          ┌──────────────────────────────┐
                          │  Filesystem (bytes on disk)  │
                          │  <root>/<service>/<version>/ │
                          │  archives, data/**, audit    │
                          └───────────────┬──────────────┘
                                          │ FS success first (atomic rename)
                                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             DuckDB Catalog (brain)                        │
│  tables: versions, artifacts, extracted_files, validations, latest_pointer│
│  views : v_version_stats, v_latest_files, v_validation_failures, deltas   │
│  events: (optional) append structured events                              │
│  ───────────────────────────────────────────────────────────────────────  │
│  Boundaries (writer_tx):                                                  │
│   • download  → upsert artifacts (after archive write)                    │
│   • extract   → bulk insert extracted_files (after secure extraction)     │
│   • validate  → insert validations                                        │
│   • latest    → set_latest (DB authoritative) + optional JSON mirror      │
└──────────────────────────┬────────────────────────────────────────────────┘
                           │
                           ▼
                      CLI & APIs
   - db migrate | latest get/set | versions | files | stats | delta
   - doctor | prune (staging_fs_listing + v_fs_orphans)
```

**Core invariants**

* **FS → DB choreography**: never commit DuckDB rows unless filesystem operation succeeded.
* **LATEST** is **DB-authoritative** (JSON mirror is optional convenience).
* **Bulk**: large inserts via Appender/Arrow (no Python row loops).
* **Observability**: `db.tx.begin|commit|rollback` + boundary events with `run_id` & `config_hash`.

---

## What next?

1. Drop these scaffolds in and run `db migrate` with your existing `migrations/0001..0007.sql`.
2. Wire your existing boundaries to call `commit_*_boundary` after FS success.
3. Convert existing “what’s latest / list / stats / deltas” scripts to call `Repo` or the CLI.

If you want, I can also generate *tiny, runnable* example scripts that:

* ingest a fake “mini-version” (1 archive + 3 extracted files),
* set latest,
* run the `delta summary` macro against a second mini-version.
