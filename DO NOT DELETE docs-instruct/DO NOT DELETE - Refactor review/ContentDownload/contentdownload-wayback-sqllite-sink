Absolutely—here’s a drop-in **SQLite sink** for your Wayback telemetry that:

* creates all tables & indexes the first time it opens the DB,
* routes each `TelemetryWayback` event to the right table,
* supports WAL, busy timeouts, and file locking,
* is junior-dev friendly (explicit, strongly typed, and thoroughly commented).

You can paste this into a new file:

`src/DocsToKG/ContentDownload/telemetry_wayback_sqlite.py`

---

```python
"""
SQLite sink for Wayback telemetry.

Pairs with TelemetryWayback (see telemetry_wayback.py). This sink:
- Creates the entire schema on first use (DDL below)
- Writes one row per event with safe parameterized SQL
- Supports WAL, busy timeout, and optional file-based locking
- Keeps cardinality low (enums stored as TEXT with indexes on key fields)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# If you implemented a central lock helper, import it here.
# It should be a context manager that takes the DB path.
# Fallback to a no-op if it's not available (useful for unit tests).
try:
    from DocsToKG.ContentDownload.locks import sqlite_lock as default_sqlite_lock
except Exception:  # pragma: no cover
    @contextmanager
    def default_sqlite_lock(_: Path):
        yield


@dataclass
class SQLiteTuning:
    """Pragmas & tuning you can tweak from the resolver/runner."""
    busy_timeout_ms: int = 4000          # retry on "database is locked"
    journal_mode: str = "WAL"            # WAL improves concurrency
    synchronous: str = "NORMAL"          # durability vs throughput
    foreign_keys: bool = True            # keep referential integrity
    cache_size_mb: int = 64              # negative means size in KiB
    page_size: int = 4096                # page size; default 4096 is fine


class SQLiteSink:
    """
    Implements the TelemetrySink protocol: emit(event: Mapping[str, Any]) -> None

    Usage:
        sink = SQLiteSink(Path("run/telemetry/wayback.sqlite"))
        tele = TelemetryWayback(run_id, sinks=[sink])
    """

    def __init__(
        self,
        db_path: Path,
        *,
        tuning: SQLiteTuning | None = None,
        lock_ctx=default_sqlite_lock,     # pass your locks.sqlite_lock here
        auto_commit_every: int = 1,       # commit after N events (>=1)
        schema_version: str = "1",        # bump if you change shapes
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tuning = tuning or SQLiteTuning()
        self.lock_ctx = lock_ctx
        self.auto_commit_every = max(1, int(auto_commit_every))
        self.schema_version = schema_version

        self._conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._ensure_schema()
        self._pending = 0

    # ────────────────────────────────────────────────────────────────────────────
    # Public API expected by TelemetryWayback
    # ────────────────────────────────────────────────────────────────────────────

    def emit(self, event: Mapping[str, Any]) -> None:
        """Dispatch an event dict (already envelope-enriched) to the right table."""
        et = event.get("event_type")
        if not et:
            return  # ignore malformed events silently

        with self.lock_ctx(self.db_path):
            cur = self._conn.cursor()
            if et == "wayback_attempt":
                self._emit_attempt(cur, event)
            elif et == "wayback_discovery":
                self._emit_discovery(cur, event)
            elif et == "wayback_candidate":
                self._emit_candidate(cur, event)
            elif et == "wayback_html_parse":
                self._emit_html_parse(cur, event)
            elif et == "wayback_pdf_check":
                self._emit_pdf_check(cur, event)
            elif et == "wayback_emit":
                self._emit_emit(cur, event)
            elif et == "wayback_skip":
                self._emit_skip(cur, event)
            else:
                # Unknown event_type; ignore to be forward-compatible
                return

            self._pending += 1
            if self._pending >= self.auto_commit_every:
                self._conn.commit()
                self._pending = 0

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    # ────────────────────────────────────────────────────────────────────────────
    # DDL & PRAGMAs
    # ────────────────────────────────────────────────────────────────────────────

    def _apply_pragmas(self) -> None:
        t = self.tuning
        c = self._conn.cursor()
        if t.foreign_keys:
            c.execute("PRAGMA foreign_keys = ON;")
        c.execute("PRAGMA journal_mode = %s;" % t.journal_mode)
        c.execute("PRAGMA synchronous = %s;" % t.synchronous)
        c.execute("PRAGMA busy_timeout = ?;", (int(t.busy_timeout_ms),))
        # cache_size: negative means KB
        c.execute("PRAGMA cache_size = ?;", (-int(t.cache_size_mb) * 1024,))
        # page_size applies on new DBs before first table; safe to set
        c.execute("PRAGMA page_size = ?;", (int(t.page_size),))

    def _ensure_schema(self) -> None:
        c = self._conn.cursor()

        # Meta schema version (simple flag)
        c.execute("""
        CREATE TABLE IF NOT EXISTS _meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        c.execute("INSERT OR IGNORE INTO _meta(key, value) VALUES ('wayback_schema_version', ?);", (self.schema_version,))

        # Attempts (start & end merge here via upsert)
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_attempts (
            attempt_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            work_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            resolver TEXT NOT NULL,
            schema TEXT NOT NULL,
            original_url TEXT,
            canonical_url TEXT,
            publication_year INTEGER,
            start_ts TEXT,   -- RFC3339
            end_ts   TEXT,   -- RFC3339
            result TEXT,     -- AttemptResult enum
            mode_selected TEXT, -- ModeSelected enum
            total_duration_ms INTEGER,
            candidates_scanned INTEGER
        );
        """)

        # Discovery (Availability & CDX)
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            stage TEXT NOT NULL,      -- DiscoveryStage
            query_url TEXT NOT NULL,
            year_window TEXT,
            "limit" INTEGER,
            returned INTEGER,
            first_ts TEXT,
            last_ts TEXT,
            http_status INTEGER,
            from_cache INTEGER,
            revalidated INTEGER,
            rate_delay_ms INTEGER,
            retry_after_s INTEGER,
            retry_count INTEGER,
            error TEXT,
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        # Candidates evaluated
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            archive_url TEXT NOT NULL,
            memento_ts TEXT NOT NULL,
            statuscode INTEGER,
            mimetype TEXT,
            source_stage TEXT NOT NULL,   -- DiscoveryStage
            decision TEXT NOT NULL,       -- CandidateDecision
            distance_to_pub_year INTEGER,
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        # HTML parse events
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_html_parses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            archive_html_url TEXT NOT NULL,
            html_http_status INTEGER,
            from_cache INTEGER,
            revalidated INTEGER,
            html_bytes INTEGER,
            pdf_link_found INTEGER NOT NULL,
            pdf_discovery_method TEXT,
            discovered_pdf_url TEXT,
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        # Archived PDF verification
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_pdf_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            archive_pdf_url TEXT NOT NULL,
            head_status INTEGER,
            content_type TEXT,
            content_length INTEGER,
            is_pdf_signature INTEGER,
            min_bytes_pass INTEGER,
            decision TEXT NOT NULL,     -- CandidateDecision
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        # Emits (success)
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_emits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            emitted_url TEXT NOT NULL,
            memento_ts TEXT NOT NULL,
            source_mode TEXT NOT NULL,      -- ModeSelected
            http_ct_expected TEXT NOT NULL,
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        # Skips (terminal non-success)
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_skips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            reason TEXT NOT NULL,           -- SkipReason
            details TEXT,
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        # Helpful indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_attempts_run   ON wayback_attempts(run_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_attempts_art   ON wayback_attempts(artifact_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_emits_mode     ON wayback_emits(source_mode);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_disc_attempt   ON wayback_discoveries(attempt_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_cand_attempt   ON wayback_candidates(attempt_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_html_attempt   ON wayback_html_parses(attempt_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_pdf_attempt    ON wayback_pdf_checks(attempt_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_wayback_skip_attempt   ON wayback_skips(attempt_id);")

    # ────────────────────────────────────────────────────────────────────────────
    # Per-event writers
    # ────────────────────────────────────────────────────────────────────────────

    def _emit_attempt(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        # Start or end of an attempt; use upsert (insert-or-update)
        ev = e.get("event")
        if ev == "start":
            cur.execute(
                """
                INSERT INTO wayback_attempts (
                    attempt_id, run_id, work_id, artifact_id, resolver, schema,
                    original_url, canonical_url, publication_year, start_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    work_id=excluded.work_id,
                    artifact_id=excluded.artifact_id,
                    resolver=excluded.resolver,
                    schema=excluded.schema,
                    original_url=excluded.original_url,
                    canonical_url=excluded.canonical_url,
                    publication_year=excluded.publication_year,
                    start_ts=excluded.start_ts
                """,
                (
                    e["attempt_id"], e["run_id"], e["work_id"], e["artifact_id"],
                    e.get("resolver", "wayback"), e.get("schema", "1"),
                    e.get("original_url"), e.get("canonical_url"), e.get("publication_year"),
                    e.get("ts"),
                ),
            )
        elif ev == "end":
            cur.execute(
                """
                UPDATE wayback_attempts
                SET end_ts = ?,
                    result = ?,
                    mode_selected = ?,
                    total_duration_ms = ?,
                    candidates_scanned = ?
                WHERE attempt_id = ?
                """,
                (
                    e.get("ts"), e.get("result"), e.get("mode_selected"),
                    e.get("total_duration_ms"), e.get("candidates_scanned"),
                    e["attempt_id"],
                ),
            )

    def _emit_discovery(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO wayback_discoveries (
                attempt_id, ts, monotonic_ms, stage, query_url, year_window,
                "limit", returned, first_ts, last_ts, http_status,
                from_cache, revalidated, rate_delay_ms, retry_after_s, retry_count, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"], e["ts"], e["monotonic_ms"],
                e.get("stage"), e.get("query_url"), e.get("year_window"),
                e.get("limit"), e.get("returned"), e.get("first_ts"), e.get("last_ts"),
                e.get("http_status"),
                _b(e.get("from_cache")), _b(e.get("revalidated")),
                e.get("rate_delay_ms"), e.get("retry_after_s"), e.get("retry_count"),
                e.get("error"),
            ),
        )

    def _emit_candidate(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO wayback_candidates (
                attempt_id, ts, monotonic_ms, archive_url, memento_ts,
                statuscode, mimetype, source_stage, decision, distance_to_pub_year
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"], e["ts"], e["monotonic_ms"],
                e.get("archive_url"), e.get("memento_ts"),
                e.get("statuscode"), e.get("mimetype"),
                e.get("source_stage"), e.get("decision"),
                e.get("distance_to_pub_year"),
            ),
        )

    def _emit_html_parse(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO wayback_html_parses (
                attempt_id, ts, monotonic_ms, archive_html_url,
                html_http_status, from_cache, revalidated, html_bytes,
                pdf_link_found, pdf_discovery_method, discovered_pdf_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"], e["ts"], e["monotonic_ms"],
                e.get("archive_html_url"),
                e.get("html_http_status"), _b(e.get("from_cache")), _b(e.get("revalidated")),
                e.get("html_bytes"),
                _b(e.get("pdf_link_found")), e.get("pdf_discovery_method"),
                e.get("discovered_pdf_url"),
            ),
        )

    def _emit_pdf_check(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO wayback_pdf_checks (
                attempt_id, ts, monotonic_ms, archive_pdf_url, head_status,
                content_type, content_length, is_pdf_signature, min_bytes_pass, decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"], e["ts"], e["monotonic_ms"],
                e.get("archive_pdf_url"), e.get("head_status"),
                e.get("content_type"), e.get("content_length"),
                _b(e.get("is_pdf_signature")), _b(e.get("min_bytes_pass")),
                e.get("decision"),
            ),
        )

    def _emit_emit(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO wayback_emits (
                attempt_id, ts, monotonic_ms, emitted_url, memento_ts,
                source_mode, http_ct_expected
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"], e["ts"], e["monotonic_ms"],
                e.get("emitted_url"), e.get("memento_ts"),
                e.get("source_mode"), e.get("http_ct_expected", "application/pdf"),
            ),
        )

    def _emit_skip(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO wayback_skips (
                attempt_id, ts, monotonic_ms, reason, details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"], e["ts"], e["monotonic_ms"],
                e.get("reason"), e.get("details"),
            ),
        )

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _b(v: Optional[bool]) -> Optional[int]:
    """Convert bool/None to 1/0/None for SQLite."""
    if v is None:
        return None
    return 1 if bool(v) else 0
```

---

## How to wire it (quick steps)

1. **Create the file** above at `src/DocsToKG/ContentDownload/telemetry_wayback_sqlite.py`.
2. In your resolver (or runner), instantiate the sink and pass it to `TelemetryWayback`:

```python
from pathlib import Path
from DocsToKG.ContentDownload.telemetry_wayback_sqlite import SQLiteSink

db_path = Path(run_dir) / "telemetry/wayback.sqlite"
sqlite_sink = SQLiteSink(db_path)
tele = TelemetryWayback(run_id, sinks=[sqlite_sink])
```

3. Keep your **JSONL sink** if you want “human tail logs” alongside SQLite; TelemetryWayback accepts multiple sinks:

```python
from DocsToKG.ContentDownload.telemetry_wayback import JsonlSink
tele = TelemetryWayback(run_id, sinks=[sqlite_sink, JsonlSink(Path(run_dir)/"telemetry/wayback.jsonl")])
```

4. **Locking:** If you implemented `locks.sqlite_lock(db_path)`, the sink uses it automatically (see `default_sqlite_lock`). This prevents “database is locked” storms in multi-process runs.

5. **Tuning:** You can pass a custom `SQLiteTuning()` to adjust WAL/busy timeout in weird environments.

---

## DDL recap (what gets created)

* `wayback_attempts` (1 row per attempt; “start” upserts basics, “end” fills result/duration)
* `wayback_discoveries` (Availability & CDX)
* `wayback_candidates` (few snapshots you actually evaluated)
* `wayback_html_parses` (when you take the HTML path)
* `wayback_pdf_checks` (HEAD/sniff)
* `wayback_emits` (successes)
* `wayback_skips` (explicit reasons)
* Indexes on `(run_id)`, `(artifact_id)`, and per-table `attempt_id` for fast joins.

This matches the one-pager spec we drafted, so your dashboards/queries can be copy-pasted.

---

## Testing checklist

* Emit a **start** then **end** attempt; assert one row in `wayback_attempts` with `start_ts` and `end_ts`.
* Emit **availability** and **cdx** rows; assert both present with `stage` different.
* Emit a **candidate** and an **emit**; assert both present and joinable on `attempt_id`.
* Emit an **html_parse** for the same attempt; assert `pdf_link_found=1`.
* Emit a **skip**; assert it inserts and that attempts table has still been updated independently by your “end” row.
* Concurrency: from two processes, write to the same DB (with locks); assert no “database is locked” exceptions and WAL mode is set.

---

## Nice-to-haves (easy add-ons)

* **Vacuum/maintenance** API on sink (e.g., `sink.vacuum()` and `sink.analyze()`) to run post-run.
* **Batching**: set `auto_commit_every=100` for better throughput in heavy crawls.
* **Retention**: one SQL to delete rows by `run_id` or older than X days (if the file grows too large).
* **Foreign key cascades** are already enabled, so deleting an attempt row cleans up its children.

If you want me to also provide a tiny set of SQLite queries (Python helpers) to compute the KPIs (yield, path mix, p95 latency), I can add that as a companion util—just say the word.
