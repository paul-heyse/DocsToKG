Amazing—let’s make this as “one-click” as we can. Below are **three PR seed branches**, each with:

* a **branch name**,
* a **commit message**,
* **ready-to-apply patch** (unified diff) that adds scaffolds + tests,
* a **how-to-apply** block, and
* a **mini acceptance checklist** per PR.

> These patches assume your repo layout:
> `src/DocsToKG/OntologyDownload/…` and `tests/ontology_download/…`.
> If a path differs, tweak the patch headers accordingly before `git apply`.

---

## PR-A — DuckDB **Prune & Doctor** (Phase 2/3)

**Branch:** `feature/catalog-prune-doctor`
**Commit:** `catalog(prune,doctor): staging → v_fs_orphans → prune + doctor CLIs + tests`

### How to apply

```bash
git checkout -b feature/catalog-prune-doctor
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: src/DocsToKG/OntologyDownload/catalog/prune.py
+from __future__ import annotations
+from pathlib import Path
+import os
+from .connection import DuckDBConfig, get_writer, writer_tx
+
+def load_staging_from_fs(cfg: DuckDBConfig, root: Path) -> int:
+    """
+    Walk <root> and load file metadata into staging_fs_listing.
+    Requires DDL for:
+      CREATE TABLE IF NOT EXISTS staging_fs_listing(scope TEXT, relpath TEXT, size_bytes BIGINT, mtime TIMESTAMP);
+    """
+    conn = get_writer(cfg)
+    conn.execute("TRUNCATE staging_fs_listing")
+    count = 0
+    root = root.resolve()
+    for base, _dirs, files in os.walk(root):
+        for fn in files:
+            p = Path(base) / fn
+            rel = p.resolve().relative_to(root).as_posix()
+            st = p.stat()
+            conn.execute(
+                "INSERT INTO staging_fs_listing(scope, relpath, size_bytes, mtime) VALUES ('version', ?, ?, ?)",
+                [rel, int(st.st_size), None],
+            )
+            count += 1
+    return count
+
+def list_orphans(cfg: DuckDBConfig) -> list[tuple[str, int]]:
+    """
+    Returns (relpath, size_bytes) using v_fs_orphans view.
+    Requires DDL for:
+      CREATE OR REPLACE VIEW v_fs_orphans AS
+        SELECT s.relpath, s.size_bytes
+        FROM staging_fs_listing s
+        LEFT JOIN (
+          SELECT fs_relpath AS rel FROM artifacts
+          UNION ALL
+          SELECT service||'/'||version_id||'/'||relpath_in_version AS rel FROM extracted_files
+        ) cat ON cat.rel = s.relpath
+        WHERE cat.rel IS NULL;
+    """
+    conn = get_writer(cfg)
+    return conn.execute("SELECT relpath, size_bytes FROM v_fs_orphans").fetchall()
+
+def delete_orphans(cfg: DuckDBConfig, root: Path, relpaths: list[str]) -> int:
+    """
+    Deletes FS orphans in small batches. DB is not mutated here—DuckDB is the source of truth already.
+    """
+    deleted = 0
+    with writer_tx(cfg, boundary="prune"):
+        for rel in relpaths:
+            try:
+                (root / rel).unlink(missing_ok=True)
+                deleted += 1
+            except Exception:
+                # Log WARN upstream; keep pruning resilient
+                pass
+    return deleted
+
*** End Patch
PATCH
```

```bash
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: src/DocsToKG/OntologyDownload/catalog/doctor.py
+from __future__ import annotations
+from pathlib import Path
+from .connection import DuckDBConfig, get_writer, writer_tx
+
+def db_rows_missing_on_fs(cfg: DuckDBConfig, root: Path) -> list[str]:
+    """
+    Returns relpaths recorded in extracted_files that do not exist on disk.
+    """
+    conn = get_writer(cfg)
+    rows = conn.execute("""
+      SELECT service||'/'||version_id||'/'||relpath_in_version AS rel
+      FROM extracted_files
+    """).fetchall()
+    root = root.resolve()
+    return [rel for (rel,) in rows if not (root / rel).exists()]
+
+def drop_missing_file_rows(cfg: DuckDBConfig, rels: list[str]) -> int:
+    """
+    Deletes extracted_files rows that point to missing files.
+    """
+    n = 0
+    with writer_tx(cfg, boundary="doctor"):
+        conn = get_writer(cfg)
+        for rel in rels:
+            conn.execute("""
+              DELETE FROM extracted_files
+              WHERE (service||'/'||version_id||'/'||relpath_in_version)=?
+            """, [rel])
+            n += 1
+    return n
+
*** End Patch
PATCH
```

```bash
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/DocsToKG/OntologyDownload/cli/db_cmd.py
@@
 import typer
 from pathlib import Path
-from ..catalog.connection import DuckDBConfig
+from ..catalog.connection import DuckDBConfig
+from ..catalog.prune import load_staging_from_fs, list_orphans, delete_orphans
+from ..catalog.doctor import db_rows_missing_on_fs, drop_missing_file_rows

 app = typer.Typer(help="DuckDB catalog utilities")

 def _cfg(db: Path, threads: int) -> DuckDBConfig:
     return DuckDBConfig(path=db, threads=threads, readonly=False, writer_lock=True)

+# --- PRUNE ---
+@app.command("prune")
+def prune(db: Path = typer.Option(..., "--db"),
+          root: Path = typer.Option(..., "--root"),
+          dry_run: bool = typer.Option(True, "--dry-run/--apply"),
+          max_items: int = typer.Option(1000, "--max-items"),
+          threads: int = 8):
+    cfg = _cfg(db, threads)
+    staged = load_staging_from_fs(cfg, root)
+    typer.echo(f"staged fs entries: {staged}")
+    orphans = list_orphans(cfg)[:max_items]
+    if dry_run:
+        for rel, size in orphans:
+            typer.echo(f"DRY-RUN delete {rel}\t{size}B")
+        typer.echo(f"would delete {len(orphans)} files")
+    else:
+        deleted = delete_orphans(cfg, root, [rel for rel, _ in orphans])
+        typer.echo(f"deleted {deleted} files")
+
+# --- DOCTOR: drop missing rows ---
+@app.command("doctor-drop-missing")
+def doctor_drop_missing(db: Path = typer.Option(..., "--db"),
+                        root: Path = typer.Option(..., "--root"),
+                        threads: int = 8):
+    cfg = _cfg(db, threads)
+    missing = db_rows_missing_on_fs(cfg, root)
+    dropped = drop_missing_file_rows(cfg, missing)
+    typer.echo(f"dropped {dropped} missing rows")
+
*** End Patch
PATCH
```

```bash
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: tests/ontology_download/db/test_prune_and_doctor.py
+import os
+from pathlib import Path
+from src.DocsToKG.OntologyDownload.catalog.connection import DuckDBConfig, get_writer
+from src.DocsToKG.OntologyDownload.catalog.prune import load_staging_from_fs, list_orphans, delete_orphans
+from src.DocsToKG.OntologyDownload.catalog.doctor import db_rows_missing_on_fs, drop_missing_file_rows
+
+def _bootstrap_min_schema(cfg: DuckDBConfig):
+    conn = get_writer(cfg)
+    conn.execute("CREATE TABLE IF NOT EXISTS staging_fs_listing(scope TEXT, relpath TEXT, size_bytes BIGINT, mtime TIMESTAMP)")
+    conn.execute("CREATE TABLE IF NOT EXISTS artifacts(artifact_id TEXT, version_id TEXT, service TEXT, source_url TEXT, etag TEXT, last_modified TIMESTAMP, content_type TEXT, size_bytes BIGINT, fs_relpath TEXT, status TEXT)")
+    conn.execute("CREATE TABLE IF NOT EXISTS extracted_files(file_id TEXT, artifact_id TEXT, version_id TEXT, relpath_in_version TEXT, format TEXT, size_bytes BIGINT, mtime TIMESTAMP, cas_relpath TEXT)")
+    conn.execute("""
+      CREATE OR REPLACE VIEW v_fs_orphans AS
+      SELECT s.relpath, s.size_bytes
+      FROM staging_fs_listing s
+      LEFT JOIN (
+        SELECT fs_relpath AS rel FROM artifacts
+        UNION ALL
+        SELECT service||'/'||version_id||'/'||relpath_in_version AS rel FROM extracted_files
+      ) cat ON cat.rel = s.relpath
+      WHERE cat.rel IS NULL
+    """)
+
+def test_prune_and_doctor(tmp_path: Path):
+    root = tmp_path / "ontologies"
+    (root / "svc/v1/data").mkdir(parents=True)
+    (root / "svc/v1/data/a.ttl").write_text("x")
+    (root / "svc/v1/data/b.ttl").write_text("x")
+
+    cfg = DuckDBConfig(path=tmp_path / "catalog.duckdb")
+    _bootstrap_min_schema(cfg)
+    conn = get_writer(cfg)
+    conn.execute("INSERT INTO extracted_files VALUES ('fid','aid','v1','data/a.ttl','ttl',1,NULL,NULL)")
+
+    # PRUNE: b.ttl should be orphan
+    staged = load_staging_from_fs(cfg, root / "svc/v1")
+    assert staged >= 2
+    orphans = list_orphans(cfg)
+    assert any(rel.endswith("data/b.ttl") for rel, _ in orphans)
+    deleted = delete_orphans(cfg, root / "svc/v1", [rel for rel, _ in orphans])
+    assert deleted >= 1
+    assert not (root / "svc/v1/data/b.ttl").exists()
+
+    # DOCTOR: now delete a.ttl so DB row is stale
+    (root / "svc/v1/data/a.ttl").unlink()
+    missing = db_rows_missing_on_fs(cfg, root / "svc/v1")
+    assert any(rel.endswith("data/a.ttl") for rel in missing)
+    dropped = drop_missing_file_rows(cfg, missing)
+    assert dropped >= 1
+
*** End Patch
PATCH
```

```bash
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: src/DocsToKG/OntologyDownload/catalog/migrations/0005_staging_prune.sql
+-- DuckDB prune/doctor staging + orphans view
+BEGIN TRANSACTION;
+
+CREATE TABLE IF NOT EXISTS schema_version(
+  version TEXT PRIMARY KEY,
+  applied_at TIMESTAMP NOT NULL DEFAULT now()
+);
+
+CREATE TABLE IF NOT EXISTS staging_fs_listing (
+  scope       TEXT NOT NULL,
+  relpath     TEXT NOT NULL,
+  size_bytes  BIGINT NOT NULL,
+  mtime       TIMESTAMP
+);
+
+CREATE OR REPLACE VIEW v_fs_orphans AS
+SELECT s.relpath, s.size_bytes
+FROM staging_fs_listing s
+LEFT JOIN (
+  SELECT fs_relpath AS rel FROM artifacts
+  UNION ALL
+  SELECT service||'/'||version_id||'/'||relpath_in_version AS rel FROM extracted_files
+) cat ON cat.rel = s.relpath
+WHERE cat.rel IS NULL;
+
+INSERT OR IGNORE INTO schema_version(version) VALUES('0005_staging_prune');
+COMMIT;
+
*** End Patch
PATCH
```

```bash
git commit -m "catalog(prune,doctor): add staging loader, orphans view usage, CLI commands and tests"
```

**Accept when**

* `ontofetch db prune --db … --root … --dry-run` lists realistic orphans.
* `--apply` deletes files and prints counts.
* `ontofetch db doctor-drop-missing` drops stale rows when files are missing.
* Tests pass: `pytest tests/ontology_download/db/test_prune_and_doctor.py -q`.

---

## PR-B — **Rate-limit modernization** (no legacy hooks; `cooldown_for`)

**Branch:** `feature/ratelimit-cooldown-no-legacy`
**Commit:** `ratelimit: add cooldown_for + honor in acquire(); remove legacy apply_retry_after`

### How to apply

```bash
git checkout -b feature/ratelimit-cooldown-no-legacy
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/DocsToKG/OntologyDownload/ratelimit/manager.py
@@
-from time import monotonic
+from time import monotonic, sleep
 from typing import Dict
@@
 _COOLDOWN_UNTIL: Dict[str, float] = {}  # key -> deadline (monotonic seconds)

 def cooldown_for(key: str, seconds: float) -> None:
     """Record a provider-suggested cooldown so the next acquire() returns fast."""
     if seconds <= 0:
         return
     _COOLDOWN_UNTIL[key] = monotonic() + seconds

+def _respect_cooldown_if_any(key: str, mode: str) -> None:
+    deadline = _COOLDOWN_UNTIL.get(key)
+    if deadline is None:
+        return
+    remaining = deadline - monotonic()
+    if remaining > 0:
+        if mode == "fail":
+            # surface immediately; caller decides to sleep (Tenacity)
+            from pyrate_limiter import RateLimitException
+            raise RateLimitException("cooldown active", period=remaining)
+        sleep(remaining)
+    _COOLDOWN_UNTIL.pop(key, None)
+
 def acquire(service: str | None, host: str | None, *, mode: str = "block", weight: int = 1) -> None:
     key = f"{service or '_'}:{host or 'default'}"
+    _respect_cooldown_if_any(key, mode)
     # ... existing Limiter lookup + try_acquire(name=key, weight=weight, ...) ...
*** End Patch
PATCH
```

> **Remove legacy references** manually if any exist (often in older helpers):
> `grep -R "apply_retry_after\(|TokenBucket|SharedTokenBucket" src/DocsToKG/OntologyDownload` → should be **0** hits after this PR.

### Add “no double wait” test

```bash
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: tests/ontology_download/ratelimit/test_retry_after_no_double_wait.py
+import time, httpx, os, pytest
+from pathlib import Path
+from src.DocsToKG.OntologyDownload.ratelimit.manager import cooldown_for
+
+def test_no_double_wait(monkeypatch, tmp_path, settings):
+    calls = {"n": 0}
+    def handler(req: httpx.Request) -> httpx.Response:
+        calls["n"] += 1
+        if calls["n"] == 1:
+            return httpx.Response(429, headers={"Retry-After": "1"})
+        return httpx.Response(200, content=b"x"*1024)
+
+    from src.DocsToKG.OntologyDownload.net import client as net_client
+    c = net_client.get_http_client(settings)
+    c._transport = httpx.MockTransport(handler)
+
+    # wrap fetch_to_path if present; else simulate call-site logic
+    from src.DocsToKG.OntologyDownload.download import fetch_to_path
+    t0 = time.perf_counter()
+    out = tmp_path / "file.bin"
+    fetch_to_path(settings, "https://example.org/file.bin", out, service="ols")
+    dt = time.perf_counter() - t0
+    assert dt == pytest.approx(1.0, rel=0.4)  # ~ Retry-After, not double
+    assert out.stat().st_size == 1024
*** End Patch
PATCH
```

```bash
git commit -m "ratelimit: add cooldown_for + honor in acquire(); test no double wait on 429"
```

**Accept when**

* `rg "apply_retry_after\(|TokenBucket|SharedTokenBucket" src/DocsToKG/OntologyDownload` → **0** matches.
* The retry test shows ~Retry-After total wall-time, not 2×.

---

## PR-C — Planner **GET-first probe** + **audited redirects**

**Branch:** `feature/planner-get-first-redirect-audit`
**Commit:** `planner(probe): GET-first with Range 0-0; audited redirects via URL gate; tests`

### How to apply

```bash
git checkout -b feature/planner-get-first-redirect-audit
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: src/DocsToKG/OntologyDownload/planners/probe.py
+from __future__ import annotations
+import httpx
+from typing import NamedTuple, Optional
+from ..net.client import get_http_client, request_with_redirect_audit
+from ..policy.url_gate import validate_url_security
+
+class ProbeResult(NamedTuple):
+    status: int
+    content_type: Optional[str]
+    content_length: Optional[int]
+    etag: Optional[str]
+    last_modified: Optional[str]
+
+TRUSTS_HEAD = {"ebi.ac.uk", "data.bioontology.org"}
+
+def probe_url(settings, url: str) -> ProbeResult:
+    client = get_http_client(settings)
+    u = httpx.URL(url)
+    validate_url_security(str(u))  # gate pre-flight
+
+    if u.host in TRUSTS_HEAD:
+        resp = request_with_redirect_audit(client, "HEAD", str(u))
+        return ProbeResult(
+            status=resp.status_code,
+            content_type=resp.headers.get("Content-Type"),
+            content_length=(int(resp.headers.get("Content-Length", "0") or 0) or None),
+            etag=resp.headers.get("ETag"),
+            last_modified=resp.headers.get("Last-Modified"),
+        )
+
+    headers = {"Range": "bytes=0-0"}
+    resp = request_with_redirect_audit(client, "GET", str(u), headers=headers, stream=True)
+    try:
+        resp.raise_for_status()
+        total = None
+        cr = resp.headers.get("Content-Range")
+        if cr and "/" in cr:
+            total = cr.rsplit("/", 1)[-1]
+        clen = total if (total and total.isdigit()) else resp.headers.get("Content-Length")
+        content_length = int(clen) if (clen and str(clen).isdigit()) else None
+        return ProbeResult(
+            status=resp.status_code,
+            content_type=resp.headers.get("Content-Type"),
+            content_length=content_length,
+            etag=resp.headers.get("ETag"),
+            last_modified=resp.headers.get("Last-Modified"),
+        )
+    finally:
+        resp.close()
+
*** End Patch
PATCH
```

```bash
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: tests/ontology_download/net/test_planner_probe_get_first.py
+import httpx
+import pytest
+from src.DocsToKG.OntologyDownload.planners.probe import probe_url
+from src.DocsToKG.OntologyDownload.net.client import get_http_client
+
+def test_probe_get_first_uses_range_zero(settings, monkeypatch):
+    seen = {"range": None}
+    def handler(req: httpx.Request) -> httpx.Response:
+        seen["range"] = req.headers.get("Range")
+        return httpx.Response(206, headers={"Content-Range": "bytes 0-0/12345", "Content-Type":"text/plain"})
+    c = get_http_client(settings)
+    c._transport = httpx.MockTransport(handler)
+    monkeypatch.setattr("src.DocsToKG.OntologyDownload.net.client.get_http_client", lambda s: c)
+    pr = probe_url(settings, "https://example.org/file.rdf")
+    assert seen["range"] == "bytes=0-0"
+    assert pr.content_length == 12345
+
*** End Patch
PATCH
```

```bash
git commit -m "planner(probe): GET-first with Range 0-0 for untrusted HEAD; audited redirects; tests"
```

**Accept when**

* The GET-first test passes and devtools show `Range: bytes=0-0` being set for non-trusted hosts.
* Any planner code that previously did unconditional `HEAD` now routes through this helper (or uses `HEAD` only for the `TRUSTS_HEAD` set).

---

## One-shot review checklist (copy into your sprint ticket)

* [ ] PR-A merged: `db prune` & `db doctor` wired, tests pass.
* [ ] PR-B merged: **no** `apply_retry_after|TokenBucket|SharedTokenBucket` survives; cooldown + Tenacity show **no double wait**.
* [ ] PR-C merged: planner uses **`probe_url()`** with GET-first (Range 0-0), audited redirects.
* [ ] `net.request` & `db.tx.*` events visible around these boundaries; (optional) `storage.delete` during prune.
* [ ] README/CLI help updated to reflect new commands and retry behavior.

---

### Final notes

* All three PRs are **intentionally small** and self-contained. You can merge in any order, but the best flow is **PR-A → PR-B → PR-C**.
* If your repository already has some of these files, `git apply` will merge hunks; otherwise it will create them.
* If you want I can also provide a GitHub Actions step for the **legacy RL grep guard** to prevent regressions.
