# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.telemetry_wayback_sqlite",
#   "purpose": "SQLite sink for Wayback telemetry",
#   "sections": [
#     {
#       "id": "sqlitetuning",
#       "name": "SQLiteTuning",
#       "anchor": "class-sqlitetuning",
#       "kind": "class"
#     },
#     {
#       "id": "sqlitesink",
#       "name": "SQLiteSink",
#       "anchor": "class-sqlitesink",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===
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
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

# If you implemented a central lock helper, import it here.
# It should be a context manager that takes the DB path.
# Fallback to a no-op if it's not available (useful for unit tests).
from DocsToKG.ContentDownload.locks import sqlite_lock as default_sqlite_lock


@dataclass
class SQLiteTuning:
    """Pragmas & tuning you can tweak from the resolver/runner."""

    busy_timeout_ms: int = 4000  # retry on "database is locked"
    journal_mode: str = "WAL"  # WAL improves concurrency
    synchronous: str = "NORMAL"  # durability vs throughput
    foreign_keys: bool = True  # keep referential integrity
    cache_size_mb: int = 64  # negative means size in KiB
    page_size: int = 4096  # page size; default 4096 is fine
    wal_autocheckpoint: int = 1000  # WAL checkpoint every N pages
    mmap_size_mb: int = 256  # mmap size for read-heavy workloads


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
        lock_ctx=default_sqlite_lock,  # pass your locks.sqlite_lock here
        auto_commit_every: int = 1,  # commit after N events (>=1)
        schema_version: str = "2",  # bump if you change shapes
        enable_metrics: bool = True,  # track performance metrics
        backpressure_threshold_ms: float = 50.0,  # warn if emit takes > X ms
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Override with environment variables if set
        import os

        auto_commit_every = int(
            os.environ.get("WAYBACK_SQLITE_AUTOCOMMIT_EVERY", str(auto_commit_every))
        )
        backpressure_threshold_ms = float(
            os.environ.get(
                "WAYBACK_SQLITE_BACKPRESSURE_THRESHOLD_MS", str(backpressure_threshold_ms)
            )
        )

        # Override tuning with environment variables
        tuning_obj = tuning or SQLiteTuning()
        tuning_obj.busy_timeout_ms = int(
            os.environ.get("WAYBACK_SQLITE_BUSY_TIMEOUT_MS", str(tuning_obj.busy_timeout_ms))
        )
        tuning_obj.wal_autocheckpoint = int(
            os.environ.get("WAYBACK_SQLITE_WAL_AUTOCHECKPOINT", str(tuning_obj.wal_autocheckpoint))
        )
        tuning_obj.page_size = int(
            os.environ.get("WAYBACK_SQLITE_PAGE_SIZE", str(tuning_obj.page_size))
        )
        tuning_obj.cache_size_mb = int(
            os.environ.get("WAYBACK_SQLITE_CACHE_SIZE_MB", str(tuning_obj.cache_size_mb))
        )
        tuning_obj.mmap_size_mb = int(
            os.environ.get("WAYBACK_SQLITE_MMAP_SIZE_MB", str(tuning_obj.mmap_size_mb))
        )

        self.tuning = tuning_obj
        self.lock_ctx = lock_ctx
        self.auto_commit_every = max(1, int(auto_commit_every))
        self.schema_version = schema_version
        self.enable_metrics = enable_metrics
        self.backpressure_threshold_ms = backpressure_threshold_ms

        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, detect_types=0, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._ensure_schema()
        self._pending = 0
        self._transaction_open = False

        # Apply schema migrations
        from DocsToKG.ContentDownload.telemetry_wayback_migrations import migrate_schema

        migrate_schema(self._conn, target_version="2")

        # Performance metrics
        self._metrics: Dict[str, Any] = {
            "events_total": 0,
            "commits_total": 0,
            "emit_times": [],  # last 100 emit durations
            "db_locked_retries": 0,
            "dead_letters_total": 0,
        }

        # Prepared statements cache
        self._prepared_stmts: Dict[str, sqlite3.Cursor] = {}

        # Register cleanup handlers
        import atexit

        atexit.register(self._cleanup_on_exit)

    # ────────────────────────────────────────────────────────────────────────────
    # Public API expected by TelemetryWayback
    # ────────────────────────────────────────────────────────────────────────────

    def emit(self, event: Mapping[str, Any]) -> None:
        """Dispatch an event dict (already envelope-enriched) to the right table."""
        import time
        import logging

        start_time = time.perf_counter()
        et = event.get("event_type")
        if not et:
            return  # ignore malformed events silently

        handler = None
        if et == "wayback_attempt":
            handler = self._emit_attempt
        elif et == "wayback_discovery":
            handler = self._emit_discovery
        elif et == "wayback_candidate":
            handler = self._emit_candidate
        elif et == "wayback_html_parse":
            handler = self._emit_html_parse
        elif et == "wayback_pdf_check":
            handler = self._emit_pdf_check
        elif et == "wayback_emit":
            handler = self._emit_emit
        elif et == "wayback_skip":
            handler = self._emit_skip
        else:
            # Unknown event_type; ignore to be forward-compatible
            return

        # Retry logic for database locked errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.lock_ctx(self.db_path):
                    self._ensure_transaction()
                    cur = self._conn.cursor()
                    handler(cur, event)

                    self._pending += 1
                    if self._pending >= self.auto_commit_every:
                        self._commit_transaction()

                # Success - break out of retry loop
                break

            except sqlite3.OperationalError as e:
                self._rollback_transaction()
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Retry with jitter
                    import random

                    delay = random.uniform(0.01, 0.05)  # 10-50ms jitter
                    time.sleep(delay)
                    if self.enable_metrics:
                        self._metrics["db_locked_retries"] = (
                            int(self._metrics["db_locked_retries"]) + 1
                        )
                    continue
                else:
                    # Max retries exceeded or other error - send to dead letter queue
                    self._send_to_dead_letter_queue(event, str(e))
                    raise

        # Track performance metrics
        if self.enable_metrics:
            emit_duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics["events_total"] = int(self._metrics["events_total"]) + 1
            emit_times = self._metrics["emit_times"]
            assert isinstance(emit_times, list)
            emit_times.append(emit_duration_ms)

            # Keep only last 100 emit times for rolling average
            if len(emit_times) > 100:
                self._metrics["emit_times"] = emit_times[-100:]

            # Check for backpressure
            if len(emit_times) >= 10:
                avg_emit_ms = sum(emit_times) / len(emit_times)
                if avg_emit_ms > self.backpressure_threshold_ms:
                    logging.warning(
                        f"Wayback SQLite sink backpressure: avg emit time {avg_emit_ms:.1f}ms "
                        f"(threshold: {self.backpressure_threshold_ms}ms). "
                        f"Consider increasing auto_commit_every from {self.auto_commit_every}."
                    )

    def _ensure_transaction(self) -> None:
        """Start a transaction if one is not already active."""

        if self._transaction_open:
            return

        # Explicit BEGIN ensures deterministic transaction boundaries even when
        # sqlite3 would otherwise lazily create them for us.
        self._conn.execute("BEGIN DEFERRED")
        self._transaction_open = True
        self._pending = 0

    def _commit_transaction(self) -> None:
        """Commit the current transaction and update metrics."""

        if not self._conn:
            return

        try:
            in_tx = self._conn.in_transaction
        except sqlite3.ProgrammingError:
            return

        was_open = self._transaction_open or in_tx
        if not was_open:
            return

        pending_before_commit = self._pending
        try:
            self._conn.commit()
        except sqlite3.ProgrammingError:
            self._transaction_open = False
            self._pending = 0
            return

        if self.enable_metrics and was_open and pending_before_commit > 0:
            self._metrics["commits_total"] = int(self._metrics["commits_total"]) + 1

        self._transaction_open = False
        self._pending = 0

    def _rollback_transaction(self) -> None:
        """Rollback the active transaction if present."""

        if not self._conn:
            return

        try:
            in_tx = self._conn.in_transaction
        except sqlite3.ProgrammingError:
            in_tx = False

        if not (self._transaction_open or in_tx):
            return

        try:
            self._conn.rollback()
        except sqlite3.ProgrammingError:
            pass
        finally:
            self._transaction_open = False
            self._pending = 0

    def close(self) -> None:
        """Close the sink and perform cleanup operations."""
        if self._conn:
            try:
                # Run optimization and WAL checkpoint
                self._cleanup_on_exit()
            finally:
                self._conn.close()

    def _send_to_dead_letter_queue(self, event: Mapping[str, Any], error: str) -> None:
        """Send failed events to a dead letter queue JSONL file."""
        import json
        import logging

        dlq_path = self.db_path.parent / f"{self.db_path.stem}.dlq.jsonl"
        try:
            with dlq_path.open("a", encoding="utf-8") as f:
                dlq_entry = {"timestamp": time.time(), "error": error, "event": dict(event)}
                f.write(json.dumps(dlq_entry, separators=(",", ":")) + "\n")

            if self.enable_metrics:
                self._metrics["dead_letters_total"] = int(self._metrics["dead_letters_total"]) + 1

        except Exception as e:
            logging.error(f"Failed to write to dead letter queue: {e}")

    def _cleanup_on_exit(self) -> None:
        """Run cleanup operations on exit (atexit handler)."""
        if not self._conn:
            return

        try:
            c = self._conn.cursor()
            # Run optimization
            c.execute("PRAGMA optimize;")

            # WAL checkpoint with truncation
            if self.tuning.journal_mode == "WAL":
                c.execute("PRAGMA wal_checkpoint(TRUNCATE);")

        except Exception as e:
            import logging

            logging.warning(f"Cleanup operations failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics collected during operation."""
        if not self.enable_metrics:
            return {}

        metrics = self._metrics.copy()
        emit_times = metrics["emit_times"]
        assert isinstance(emit_times, list)

        if emit_times:
            metrics["avg_emit_ms"] = sum(emit_times) / len(emit_times)
            sorted_times = sorted(emit_times)
            p95_index = int(len(sorted_times) * 0.95)
            metrics["p95_emit_ms"] = float(sorted_times[p95_index])
        else:
            metrics["avg_emit_ms"] = 0.0
            metrics["p95_emit_ms"] = 0.0

        return metrics

    def vacuum(self, incremental: bool = True) -> None:
        """Run database maintenance (VACUUM or incremental_vacuum).

        Args:
            incremental: If True, use PRAGMA incremental_vacuum(2000) for online cleanup.
                        If False, use VACUUM (offline, blocks all access).
        """
        try:
            c = self._conn.cursor()
            if incremental:
                c.execute("PRAGMA incremental_vacuum(2000);")
            else:
                c.execute("VACUUM;")
        except Exception as e:
            import logging

            logging.error(f"Vacuum operation failed: {e}")

    def delete_run(self, run_id: str) -> int:
        """Delete all telemetry for a specific run (retention policy).

        Args:
            run_id: Run identifier to delete.

        Returns:
            Total number of rows deleted.
        """
        try:
            c = self._conn.cursor()

            # Delete from child tables first (due to foreign key constraints)
            tables_to_delete = [
                "wayback_discoveries",
                "wayback_candidates",
                "wayback_html_parses",
                "wayback_pdf_checks",
                "wayback_emits",
                "wayback_skips",
                "wayback_attempts",
            ]

            total_deleted = 0
            for table in tables_to_delete:
                c.execute(
                    f"DELETE FROM {table} WHERE attempt_id IN (SELECT attempt_id FROM wayback_attempts WHERE run_id = ?);",
                    (run_id,),
                )
                total_deleted += c.rowcount

            self._conn.commit()
            return total_deleted
        except Exception as e:
            import logging

            logging.error(f"Failed to delete run {run_id}: {e}")
            return 0

    def analyze_schema(self) -> None:
        """Run SQLite ANALYZE to refresh query optimizer statistics."""
        try:
            c = self._conn.cursor()
            c.execute("ANALYZE;")
            self._conn.commit()
        except Exception as e:
            import logging

            logging.error(f"ANALYZE failed: {e}")

    def finalize_run_metrics(self, run_id: str) -> None:
        """Populate wayback_run_metrics roll-up table with end-of-run aggregations.

        Call this at the end of a run to compute and store metrics for fast dashboards.

        Args:
            run_id: Run identifier to finalize.
        """
        import datetime

        try:
            c = self._conn.cursor()
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Compute aggregates
            c.execute(
                """
                SELECT
                    COUNT(*) as attempts,
                    SUM(CASE WHEN result LIKE 'emitted%' THEN 1 ELSE 0 END) as emits
                FROM wayback_attempts
                WHERE run_id = ?
            """,
                (run_id,),
            )

            row = c.fetchone()
            if not row:
                return

            attempts = row[0] or 0
            emits = row[1] or 0
            yield_pct = (emits / attempts * 100.0) if attempts > 0 else 0.0

            # P95 latency
            c.execute(
                """
                SELECT total_duration_ms FROM wayback_attempts
                WHERE run_id = ? AND total_duration_ms IS NOT NULL
                ORDER BY total_duration_ms
            """,
                (run_id,),
            )

            durations = [r[0] for r in c.fetchall()]
            p95_latency_ms = None
            if durations:
                p95_index = int(len(durations) * 0.95)
                p95_latency_ms = float(durations[p95_index])

            # Cache hit rate
            c.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN from_cache = 1 THEN 1 ELSE 0 END) as cached
                FROM wayback_discoveries
                WHERE attempt_id IN (SELECT attempt_id FROM wayback_attempts WHERE run_id = ?)
            """,
                (run_id,),
            )

            row = c.fetchone()
            cache_hit_pct = 0.0
            if row and row[0] > 0:
                cache_hit_pct = (row[1] or 0) / row[0] * 100.0

            # Non-PDF and below-min-size rates
            c.execute(
                """
                SELECT
                    SUM(CASE WHEN reason = 'non_pdf' THEN 1 ELSE 0 END) as non_pdf,
                    SUM(CASE WHEN reason = 'below_min_size' THEN 1 ELSE 0 END) as below_min
                FROM wayback_skips
                WHERE attempt_id IN (SELECT attempt_id FROM wayback_attempts WHERE run_id = ?)
            """,
                (run_id,),
            )

            row = c.fetchone()
            total_skips = emits  # Rough approximation
            non_pdf_rate = ((row[0] or 0) / total_skips * 100.0) if total_skips > 0 else 0.0
            below_min_size_rate = ((row[1] or 0) / total_skips * 100.0) if total_skips > 0 else 0.0

            # Upsert into roll-up table
            c.execute(
                """
                INSERT INTO wayback_run_metrics
                (run_id, attempts, emits, yield_pct, p95_latency_ms, cache_hit_pct, non_pdf_rate, below_min_size_rate, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    attempts = excluded.attempts,
                    emits = excluded.emits,
                    yield_pct = excluded.yield_pct,
                    p95_latency_ms = excluded.p95_latency_ms,
                    cache_hit_pct = excluded.cache_hit_pct,
                    non_pdf_rate = excluded.non_pdf_rate,
                    below_min_size_rate = excluded.below_min_size_rate,
                    updated_at = excluded.updated_at
            """,
                (
                    run_id,
                    attempts,
                    emits,
                    yield_pct,
                    p95_latency_ms,
                    cache_hit_pct,
                    non_pdf_rate,
                    below_min_size_rate,
                    now,
                    now,
                ),
            )

            self._conn.commit()
        except Exception as e:
            import logging

            logging.error(f"Failed to finalize run metrics for {run_id}: {e}")

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
        c.execute("PRAGMA busy_timeout = %d;" % int(t.busy_timeout_ms))
        # cache_size: negative means KB
        c.execute("PRAGMA cache_size = %d;" % (-int(t.cache_size_mb) * 1024))
        # page_size applies on new DBs before first table; safe to set
        c.execute("PRAGMA page_size = %d;" % int(t.page_size))

        # WAL tuning
        if t.journal_mode == "WAL":
            c.execute("PRAGMA wal_autocheckpoint = %d;" % int(t.wal_autocheckpoint))

        # mmap for read-heavy workloads (64-bit systems)
        try:
            c.execute("PRAGMA mmap_size = %d;" % (int(t.mmap_size_mb) * 1024 * 1024))
        except sqlite3.OperationalError:
            # mmap not supported on this system
            pass

    def _ensure_schema(self) -> None:
        c = self._conn.cursor()

        # Meta schema version (simple flag)
        c.execute("""
        CREATE TABLE IF NOT EXISTS _meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        c.execute(
            "INSERT OR IGNORE INTO _meta(key, value) VALUES ('wayback_schema_version', ?);",
            (self.schema_version,),
        )

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
            rate_limiter_role TEXT,
            FOREIGN KEY(attempt_id) REFERENCES wayback_attempts(attempt_id) ON DELETE CASCADE
        );
        """)

        self._ensure_column_exists(
            c,
            table="wayback_discoveries",
            column="rate_limiter_role",
            ddl="ALTER TABLE wayback_discoveries ADD COLUMN rate_limiter_role TEXT",
        )

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

        # Basic indexes
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_attempts_run   ON wayback_attempts(run_id);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_attempts_art   ON wayback_attempts(artifact_id);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_emits_mode     ON wayback_emits(source_mode);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_disc_attempt   ON wayback_discoveries(attempt_id);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_cand_attempt   ON wayback_candidates(attempt_id);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_html_attempt   ON wayback_html_parses(attempt_id);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_pdf_attempt    ON wayback_pdf_checks(attempt_id);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_skip_attempt   ON wayback_skips(attempt_id);"
        )

        # Composite/covering indexes for common queries
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_run_result ON wayback_attempts(run_id, result);"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_emits_mode ON wayback_emits(source_mode, memento_ts);"
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_discovery_stage ON wayback_discoveries(stage);")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wayback_disc_role ON wayback_discoveries(rate_limiter_role);"
        )

        # Partial index for successful attempts (SQLite >= 3.8.0)
        try:
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempts_success ON wayback_attempts(result) WHERE result LIKE 'emitted%';"
            )
        except sqlite3.OperationalError:
            # Partial indexes not supported
            pass

        # Roll-up table for fast dashboards (fast query of run-level metrics)
        c.execute("""
        CREATE TABLE IF NOT EXISTS wayback_run_metrics (
            run_id TEXT PRIMARY KEY,
            attempts INTEGER DEFAULT 0,
            emits INTEGER DEFAULT 0,
            yield_pct REAL DEFAULT 0.0,
            p95_latency_ms REAL,
            cache_hit_pct REAL DEFAULT 0.0,
            non_pdf_rate REAL DEFAULT 0.0,
            below_min_size_rate REAL DEFAULT 0.0,
            created_at TEXT,
            updated_at TEXT
        );
        """)

    def _ensure_column_exists(
        self, cur: sqlite3.Cursor, *, table: str, column: str, ddl: str
    ) -> None:
        """Add a column if it is missing from an existing database."""

        cur.execute(f"PRAGMA table_info({table});")
        columns = {row[1] for row in cur.fetchall()}
        if column not in columns:
            cur.execute(ddl)

    # ────────────────────────────────────────────────────────────────────────────
    # Per-event writers
    # ────────────────────────────────────────────────────────────────────────────

    def _ensure_attempt_stub(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        attempt_id = e.get("attempt_id")
        if not attempt_id:
            return
        cur.execute(
            """
            INSERT OR IGNORE INTO wayback_attempts (
                attempt_id, run_id, work_id, artifact_id, resolver, schema, start_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                e.get("run_id") or "unknown-run",
                e.get("work_id") or "unknown-work",
                e.get("artifact_id") or "unknown-artifact",
                e.get("resolver", "wayback"),
                e.get("schema", "1"),
                e.get("ts"),
            ),
        )

    def _emit_attempt(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        # Start or end of an attempt; use upsert (insert-or-update)
        ev = e.get("event")
        self._ensure_attempt_stub(cur, e)
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
                    e["attempt_id"],
                    e["run_id"],
                    e["work_id"],
                    e["artifact_id"],
                    e.get("resolver", "wayback"),
                    e.get("schema", "1"),
                    e.get("original_url"),
                    e.get("canonical_url"),
                    e.get("publication_year"),
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
                    e.get("ts"),
                    e.get("result"),
                    e.get("mode_selected"),
                    e.get("total_duration_ms"),
                    e.get("candidates_scanned"),
                    e["attempt_id"],
                ),
            )

    def _emit_discovery(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        self._ensure_attempt_stub(cur, e)
        cur.execute(
            """
            INSERT INTO wayback_discoveries (
                attempt_id, ts, monotonic_ms, stage, query_url, year_window,
                "limit", returned, first_ts, last_ts, http_status,
                from_cache, revalidated, rate_delay_ms, retry_after_s, retry_count, error,
                rate_limiter_role
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"],
                e["ts"],
                e["monotonic_ms"],
                e.get("stage"),
                e.get("query_url"),
                e.get("year_window"),
                e.get("limit"),
                e.get("returned"),
                e.get("first_ts"),
                e.get("last_ts"),
                e.get("http_status"),
                _b(e.get("from_cache")),
                _b(e.get("revalidated")),
                e.get("rate_delay_ms"),
                e.get("retry_after_s"),
                e.get("retry_count"),
                e.get("error"),
                _normalize_role(e.get("rate_limiter_role") or e.get("role")),
            ),
        )

    def _emit_candidate(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        self._ensure_attempt_stub(cur, e)
        cur.execute(
            """
            INSERT INTO wayback_candidates (
                attempt_id, ts, monotonic_ms, archive_url, memento_ts,
                statuscode, mimetype, source_stage, decision, distance_to_pub_year
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"],
                e["ts"],
                e["monotonic_ms"],
                e.get("archive_url"),
                e.get("memento_ts"),
                e.get("statuscode"),
                e.get("mimetype"),
                e.get("source_stage"),
                e.get("decision"),
                e.get("distance_to_pub_year"),
            ),
        )

    def _emit_html_parse(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        self._ensure_attempt_stub(cur, e)
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
                e["attempt_id"],
                e["ts"],
                e["monotonic_ms"],
                e.get("archive_html_url"),
                e.get("html_http_status"),
                _b(e.get("from_cache")),
                _b(e.get("revalidated")),
                e.get("html_bytes"),
                _b(e.get("pdf_link_found")),
                e.get("pdf_discovery_method"),
                e.get("discovered_pdf_url"),
            ),
        )

    def _emit_pdf_check(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        self._ensure_attempt_stub(cur, e)
        cur.execute(
            """
            INSERT INTO wayback_pdf_checks (
                attempt_id, ts, monotonic_ms, archive_pdf_url, head_status,
                content_type, content_length, is_pdf_signature, min_bytes_pass, decision
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"],
                e["ts"],
                e["monotonic_ms"],
                e.get("archive_pdf_url"),
                e.get("head_status"),
                e.get("content_type"),
                e.get("content_length"),
                _b(e.get("is_pdf_signature")),
                _b(e.get("min_bytes_pass")),
                e.get("decision"),
            ),
        )

    def _emit_emit(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        self._ensure_attempt_stub(cur, e)
        cur.execute(
            """
            INSERT INTO wayback_emits (
                attempt_id, ts, monotonic_ms, emitted_url, memento_ts,
                source_mode, http_ct_expected
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"],
                e["ts"],
                e["monotonic_ms"],
                e.get("emitted_url"),
                e.get("memento_ts"),
                e.get("source_mode"),
                e.get("http_ct_expected", "application/pdf"),
            ),
        )

    def _emit_skip(self, cur: sqlite3.Cursor, e: Mapping[str, Any]) -> None:
        self._ensure_attempt_stub(cur, e)
        cur.execute(
            """
            INSERT INTO wayback_skips (
                attempt_id, ts, monotonic_ms, reason, details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                e["attempt_id"],
                e["ts"],
                e["monotonic_ms"],
                e.get("reason"),
                e.get("details"),
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


def _normalize_role(role: Optional[Any]) -> Optional[str]:
    """Normalize limiter role strings for consistent storage."""

    if isinstance(role, str):
        return role.strip().lower() or None
    return None


__all__ = [
    "SQLiteSink",
    "SQLiteTuning",
]
