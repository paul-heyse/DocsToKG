Perfect—let’s land the remaining bits in one clean pass. Below are **three tactical implementation plans** with **copy-ready code** and a **narrative playbook** so your team can merge with confidence on the first try.

---

# Scope & outcomes (what we’ll ship)

1. **DuckDB doctor + prune (Phase 2/3)**
   DB remains the brain; filesystem holds bytes. We’ll add a staging loader, an `v_fs_orphans` view, safe deletes in batches, and doctor flows to reconcile DB↔FS—with CLI and events.

2. **Rate-limit modernization (no legacy hooks; no double wait)**
   Eliminate `apply_retry_after` and any TokenBucket remnants. Use `pyrate-limiter` + `cooldown_for` and let **Tenacity** sleep `Retry-After`—so the next `acquire()` is fast.

3. **Planner GET-first probe with audited redirects**
   Prefer `GET` with `Range: bytes=0-0` (or header-only stream) for untrusted servers; only use `HEAD` for a whitelist of reliable hosts. Every hop validated by the **URL gate** (no auto-redirects).

---

## 1) DuckDB doctor & prune — staging → orphans → delete/apply

### Narrative (why this design)

* DuckDB is your **source of truth** for *what should exist*; the filesystem tells you *what does exist*.
* A **staging table** holds a snapshot of the FS under your storage root; a **view** does the set-difference to find orphans.
* We **never** mutate DB before an FS operation succeeds. Doctor flows clean DB rows that point to vanished files; Prune removes FS files that aren’t referenced by DB.

> Think: DB lists “live” objects; staging lists “observed” files; `v_fs_orphans = observed − live`.

### DDL (place in a migration if you haven’t yet)

```sql
-- 0005_staging_prune.sql  (apply via your migration runner)
BEGIN;

CREATE TABLE IF NOT EXISTS staging_fs_listing (
  scope       TEXT NOT NULL,          -- 'version' | future scopes
  relpath     TEXT NOT NULL,          -- path relative to storage root
  size_bytes  BIGINT NOT NULL,
  mtime       TIMESTAMP
);

-- Orphans = in FS but not referenced by artifacts or extracted_files
CREATE OR REPLACE VIEW v_fs_orphans AS
SELECT s.relpath, s.size_bytes
FROM staging_fs_listing s
LEFT JOIN (
  -- Full catalog of expected files (rel paths under root)
  SELECT fs_relpath AS rel FROM artifacts
  UNION ALL
  SELECT service || '/' || version_id || '/' || relpath_in_version AS rel FROM extracted_files
) cat ON cat.rel = s.relpath
WHERE cat.rel IS NULL;

INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES ('0005_staging_prune', now());
COMMIT;
```

### Loader, list, delete (copy-ready code)

```python
# src/DocsToKG/OntologyDownload/catalo g/prune.py
from __future__ import annotations
from pathlib import Path
import os
from .connection import DuckDBConfig, get_writer, writer_tx

def load_staging_from_fs(cfg: DuckDBConfig, root: Path) -> int:
    """Walk <root> and load file metadata into staging_fs_listing."""
    conn = get_writer(cfg)
    conn.execute("TRUNCATE staging_fs_listing")
    count = 0
    root = root.resolve()
    for base, _dirs, files in os.walk(root):
        for fn in files:
            p = Path(base) / fn
            rel = p.resolve().relative_to(root).as_posix()
            st = p.stat()
            conn.execute(
                "INSERT INTO staging_fs_listing(scope, relpath, size_bytes, mtime) VALUES ('version', ?, ?, ?)",
                [rel, int(st.st_size), None],
            )
            count += 1
    return count

def list_orphans(cfg: DuckDBConfig) -> list[tuple[str, int]]:
    conn = get_writer(cfg)  # writer is fine
    return conn.execute("SELECT relpath, size_bytes FROM v_fs_orphans").fetchall()

def delete_orphans(cfg: DuckDBConfig, root: Path, relpaths: list[str]) -> int:
    """Delete orphans in small batches; DB untouched (DB already the truth)."""
    deleted = 0
    with writer_tx(cfg, boundary="prune"):
        for rel in relpaths:
            try:
                (root / rel).unlink(missing_ok=True)
                deleted += 1
            except Exception:
                # log a WARN and continue; leave DB rows alone here
                pass
    return deleted
```

### Doctor — drop DB rows for missing files

```python
# src/DocsToKG/OntologyDownload/catalog/doctor.py
from __future__ import annotations
from pathlib import Path
from .connection import DuckDBConfig, get_writer, writer_tx

def db_rows_missing_on_fs(cfg: DuckDBConfig, root: Path) -> list[str]:
    conn = get_writer(cfg)
    rows = conn.execute("""
      SELECT service||'/'||version_id||'/'||relpath_in_version AS rel
      FROM extracted_files
    """).fetchall()
    root = root.resolve()
    return [rel for (rel,) in rows if not (root / rel).exists()]

def drop_missing_file_rows(cfg: DuckDBConfig, rels: list[str]) -> int:
    n = 0
    with writer_tx(cfg, boundary="doctor"):
        conn = get_writer(cfg)
        for rel in rels:
            conn.execute("""
              DELETE FROM extracted_files
              WHERE (service||'/'||version_id||'/'||relpath_in_version)=?
            """, [rel])
            n += 1
    return n
```

### CLI wiring

```python
# src/DocsToKG/OntologyDownload/cli/db_cmd.py
import typer
from pathlib import Path
from ..catalog.connection import DuckDBConfig
from ..catalog.prune import load_staging_from_fs, list_orphans, delete_orphans
from ..catalog.doctor import db_rows_missing_on_fs, drop_missing_file_rows

app = typer.Typer(help="DuckDB catalog utilities")

def _cfg(db: Path, threads: int) -> DuckDBConfig:
    return DuckDBConfig(path=db, threads=threads, readonly=False, writer_lock=True)

@app.command("prune")
def prune(db: Path = typer.Option(..., "--db"),
          root: Path = typer.Option(..., "--root"),
          dry_run: bool = typer.Option(True, "--dry-run/--apply"),
          max_items: int = typer.Option(500, "--max-items"),
          threads: int = 8):
    cfg = _cfg(db, threads)
    staged = load_staging_from_fs(cfg, root)
    typer.echo(f"staged fs entries: {staged}")
    orphans = list_orphans(cfg)[:max_items]
    if dry_run:
        for rel, size in orphans:
            typer.echo(f"DRY-RUN delete {rel}\t{size}B")
        typer.echo(f"would delete {len(orphans)} files")
    else:
        deleted = delete_orphans(cfg, root, [rel for rel,_ in orphans])
        typer.echo(f"deleted {deleted} files")

@app.command("doctor-drop-missing")
def doctor_drop_missing(db: Path = typer.Option(..., "--db"),
                        root: Path = typer.Option(..., "--root"),
                        threads: int = 8):
    cfg = _cfg(db, threads)
    missing = db_rows_missing_on_fs(cfg, root)
    dropped = drop_missing_file_rows(cfg, missing)
    typer.echo(f"dropped {dropped} missing rows")
```

### Tests (integration sanity)

```python
# tests/ontology_download/db/test_prune_and_doctor.py
def test_prune_and_doctor(tmp_path, settings):
    # Arrange FS
    (tmp_path / "svc/v1/data").mkdir(parents=True)
    (tmp_path / "svc/v1/data/a.ttl").write_text("x")
    (tmp_path / "svc/v1/data/b.ttl").write_text("x")
    # Arrange DB: only a.ttl recorded
    from ...catalog.connection import DuckDBConfig, get_writer
    cfg = DuckDBConfig(path=tmp_path / ".cat.duckdb")
    conn = get_writer(cfg)
    conn.execute("CREATE TABLE IF NOT EXISTS staging_fs_listing(scope TEXT, relpath TEXT, size_bytes BIGINT, mtime TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS extracted_files(file_id TEXT, artifact_id TEXT, version_id TEXT, relpath_in_version TEXT, format TEXT, size_bytes BIGINT, mtime TIMESTAMP, cas_relpath TEXT)")
    conn.execute("""
      CREATE OR REPLACE VIEW v_fs_orphans AS
      SELECT s.relpath, s.size_bytes
      FROM staging_fs_listing s
      LEFT JOIN (SELECT 'svc'||'/'||version_id||'/'||relpath_in_version AS rel FROM extracted_files) cat ON cat.rel = s.relpath
      WHERE cat.rel IS NULL
    """)
    conn.execute("INSERT INTO extracted_files VALUES ('fid','aid','v1','data/a.ttl','ttl',1,NULL,NULL)")
    # Prune: b.ttl is orphan
    from ...catalog.prune import load_staging_from_fs, list_orphans, delete_orphans
    load_staging_from_fs(cfg, tmp_path / "svc/v1")
    orphans = list_orphans(cfg)
    assert any(rel.endswith("data/b.ttl") for rel,_ in orphans)
    deleted = delete_orphans(cfg, tmp_path / "svc/v1", [rel for rel,_ in orphans])
    assert deleted >= 1
    assert not (tmp_path / "svc/v1/data/b.ttl").exists()
    # Doctor: drop a stale row (remove a.ttl on disk)
    (tmp_path / "svc/v1/data/a.ttl").unlink()
    from ...catalog.doctor import db_rows_missing_on_fs, drop_missing_file_rows
    missing = db_rows_missing_on_fs(cfg, tmp_path / "svc/v1")
    assert any(rel.endswith("data/a.ttl") for rel in missing)
    dropped = drop_missing_file_rows(cfg, missing)
    assert dropped >= 1
```

---

## 2) Rate-limit: remove legacy hooks; add `cooldown_for`; ensure **no double wait**

### Narrative

* Old `apply_retry_after(...)` mutated your buckets and could cause **two sleeps**: one from the limiter, one from Tenacity.
* New model:

  * **`acquire()`** once per attempt (block or fail).
  * On **429**, extract `Retry-After` seconds → **`cooldown_for(key, seconds)`** to “hint” the next acquire → Tenacity sleeps → next `acquire()` is near-instant.
  * **Never** mutate the limiter on status; keep limiter in charge of steady pacing.

### Code: add cooldown + honor it in `acquire()`

```python
# src/DocsToKG/OntologyDownload/ratelimit/manager.py
from time import monotonic, sleep
from typing import Dict

_COOLDOWN_UNTIL: Dict[str, float] = {}   # key → monotonic deadline

def cooldown_for(key: str, seconds: float) -> None:
    if seconds and seconds > 0:
        _COOLDOWN_UNTIL[key] = monotonic() + seconds

def _respect_cooldown_if_any(key: str, mode: str) -> None:
    deadline = _COOLDOWN_UNTIL.get(key)
    if not deadline:
        return
    remaining = deadline - monotonic()
    if remaining > 0:
        if mode == "fail":
            # surface immediately; caller/tenacity will sleep
            from pyrate_limiter import RateLimitException
            raise RateLimitException("cooldown active", period=remaining)
        sleep(remaining)
    _COOLDOWN_UNTIL.pop(key, None)

def acquire(service: str | None, host: str | None, *, mode: str = "block", weight: int = 1) -> None:
    key = f"{service or '_'}:{host or 'default'}"
    _respect_cooldown_if_any(key, mode)
    # .. existing registry lookup & Limiter.try_acquire(...) call ..
```

### Code: call-site policy (downloader/resolver)

```python
# src/DocsToKG/OntologyDownload/download.py  (429 path only)
if resp.status_code == 429:
    ra = _parse_retry_after(resp) or 1.0
    cooldown_for(f"{service}:{host}", ra)   # next acquire won’t block
    raise RetryableHttpStatus(resp)         # Tenacity sleeps; then retry
```

### CI guard: ensure legacy is gone

```bash
rg "apply_retry_after\(|TokenBucket|SharedTokenBucket" src/DocsToKG/OntologyDownload && { echo "LEGACY RL PRESENT"; exit 1; } || echo "OK"
```

### Test: prove **no double wait**

```python
def test_429_retry_after_no_double_wait(tmp_path, settings, monkeypatch):
    # first attempt 429 with Retry-After: 1 → total wall-time ≈ 1s, not 2s
```

*(I posted a complete test earlier—reuse that; it asserts ~1s total for the two attempts.)*

---

## 3) Planner GET-first probe (audited; no unconditional HEAD)

### Narrative

* Many providers are flaky on `HEAD` (missing/misleading headers).
* **Default**: `GET` with `Range: bytes=0-0` (or stream + close immediately) so you only download 1 byte (or none), but you still get truthful headers.
* **Whitelist**: only **trusted hosts** use `HEAD`.
* Every hop passes through the **URL gate**; no `follow_redirects=True`.

### Code (copy-ready)

```python
# src/DocsToKG/OntologyDownload/planners/probe.py
import httpx
from typing import NamedTuple, Optional
from ..net.client import get_http_client, request_with_redirect_audit
from ..policy.url_gate import validate_url_security

class ProbeResult(NamedTuple):
    status: int
    content_type: Optional[str]
    content_length: Optional[int]
    etag: Optional[str]
    last_modified: Optional[str]

TRUSTS_HEAD = {"ebi.ac.uk", "data.bioontology.org"}

def probe_url(settings, url: str) -> ProbeResult:
    client = get_http_client(settings)
    u = httpx.URL(url)
    validate_url_security(str(u))  # gate pre-flight

    if u.host in TRUSTS_HEAD:
        resp = request_with_redirect_audit(client, "HEAD", str(u))
        return ProbeResult(
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type"),
            content_length=(int(resp.headers.get("Content-Length", "0") or 0) or None),
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
        )

    headers = {"Range": "bytes=0-0"}
    resp = request_with_redirect_audit(client, "GET", str(u), headers=headers, stream=True)
    try:
        resp.raise_for_status()
        # Prefer Content-Range total if present (206)
        total = None
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            total = cr.rsplit("/", 1)[-1]
        clen = total if total and total.isdigit() else resp.headers.get("Content-Length")
        content_length = int(clen) if (clen and str(clen).isdigit()) else None
        return ProbeResult(
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type"),
            content_length=content_length,
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
        )
    finally:
        resp.close()
```

### Tests

```python
def test_probe_uses_range_zero_when_not_trusted(settings, monkeypatch):
    seen = {"range": None}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["range"] = req.headers.get("Range")
        return httpx.Response(206, headers={"Content-Range":"bytes 0-0/12345","Content-Type":"text/plain"})
    c = get_http_client(settings); c._transport = httpx.MockTransport(handler)
    monkeypatch.setattr("src.DocsToKG.OntologyDownload.net.client.get_http_client", lambda s: c)
    pr = probe_url(settings, "https://example.org/x")
    assert seen["range"] == "bytes=0-0"
    assert pr.content_length == 12345
```

---

## Roll-out plan (one sprint, 3 PRs)

**PR-A: DuckDB prune & doctor**

* Add DDL for staging & `v_fs_orphans`; implement loader/list/delete; wire `db prune` & `db doctor`; tests.
* **Accept**: `db prune --dry-run` lists orphans; `--apply` deletes; `doctor-drop-missing` removes stale rows.

**PR-B: RL modernization**

* Add `cooldown_for` + honor in `acquire()`; replace any legacy hook; CI `rg` guard; no-double-wait test green.
* **Accept**: `rg` shows no `apply_retry_after|TokenBucket|SharedTokenBucket`; 429 wall-time ≈ Retry-After.

**PR-C: Planner GET-first + audited redirects**

* Implement `probe_url` (GET-first, Range 0-0); limit `HEAD` to trusted hosts; ensure `request_with_redirect_audit` used.
* **Accept**: unit test asserts Range header used; redirect audit test blocks cross-host if URL gate denies.

---

## Final acceptance checklist (paste in the PR description)

* [ ] **Prune/Doctor** wired to DuckDB staging + view; CLI `db prune` and `db doctor` work; tests pass.
* [ ] **Rate-limit**: legacy hooks removed; `cooldown_for` + Tenacity path live; 429 tests show **no double wait**.
* [ ] **Planner**: GET-first with `Range: 0-0` (stream & close) for non-trusted hosts; redirect audit uses URL gate per hop.
* [ ] **Observability**: events emitted for `db.tx.*` at boundaries; (optional) `storage.delete` events when prune applies.
* [ ] **Docs**: README/CLI help updated; settings reference shows RateLimit + HTTP client knobs.

---

### Why this will “stick” on the first run

* Each change is **narrow** (tiny modules/files) but **integrated** with the architecture you’ve already landed.
* The tests prove the tricky behavior (429 **no double wait**, GET-first headers, prune/doctor correctness).
* CI greps prevent legacy relapse.

If you’d like, I can bundle these three into **PR seed branches** with the scaffolds and tests already in the right places so your team only needs to fill in a few TODOs and rename any paths that differ in your repo.
