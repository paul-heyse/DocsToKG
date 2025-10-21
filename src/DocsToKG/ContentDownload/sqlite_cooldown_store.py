# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.sqlite_cooldown_store",
#   "purpose": "Cross-process cooldown store for circuit breaker overrides (Retry-After, rolling-window)",
#   "sections": [
#     {
#       "id": "sqlitecooldownstore",
#       "name": "SQLiteCooldownStore",
#       "anchor": "class-sqlitecooldownstore",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Cross-process cooldown store backed by SQLite for circuit breaker overrides.

This module provides a thread-safe, multi-process cooldown store that persists
host cooldown deadlines in a SQLite database. It uses wall-clock time for storage
and converts to monotonic time at runtime to prevent issues with clock drift
across processes.

Key Design:
- Stores wall-clock deadlines (UTC epoch) for durability across restarts
- Converts to monotonic time (time.monotonic) at runtime for steady-state timeouts
- Uses PRAGMA journal_mode=WAL for concurrent access safety
- Uses file-level locking (via locks.sqlite_lock) for cross-process write serialization
- Automatic expiry detection and pruning

Typical Usage:
    from pathlib import Path
    from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore

    store = SQLiteCooldownStore(Path("tmp/breakers.sqlite"))

    # Set a cooldown for host api.crossref.org (until monotonic time + 60s)
    store.set_until("api.crossref.org", time.monotonic() + 60, reason="retry-after")

    # Check if cooldown is still active
    deadline = store.get_until("api.crossref.org")
    if deadline and deadline > time.monotonic():
        print("Still in cooldown")

    # Clear the cooldown
    store.clear("api.crossref.org")

    # Maintenance: prune expired entries
    pruned = store.prune_expired()
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Optional

# Optional import: uses project's existing lock utilities if available
try:
    from DocsToKG.ContentDownload.locks import sqlite_lock as default_lock
except Exception:  # pragma: no cover
    # Fallback: no-op context manager for testing or when locks not available
    @contextmanager  # type: ignore[misc]
    def default_lock(_: Path):  # type: ignore[misc]
        yield  # type: ignore[misc]


# ────────────────────────────────────────────────────────────────────────────────
# Database Schema (DDL)
# ────────────────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=4000;
CREATE TABLE IF NOT EXISTS breaker_cooldowns (
    host TEXT PRIMARY KEY,
    until_wall REAL NOT NULL,    -- UTC epoch seconds (wall-clock)
    reason TEXT,                  -- "retry-after" or "rolling-window"
    updated_at REAL NOT NULL      -- Timestamp of last update
);
CREATE INDEX IF NOT EXISTS idx_cd_until ON breaker_cooldowns(until_wall);
"""


# ────────────────────────────────────────────────────────────────────────────────
# SQLiteCooldownStore
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class SQLiteCooldownStore:
    """
    Cross-process cooldown store backed by SQLite.

    Stores host cooldown deadlines as wall-clock times (UTC epoch seconds).
    At runtime, converts to monotonic time (time.monotonic) to ensure deadlines
    are immune to clock drift across processes.

    All times are stored as wall-clock (float, UTC epoch) for durability.
    Conversions happen transparently at read/write.

    Parameters
    ----------
    db_path : Path
        Path to SQLite database file. Directories are created if missing.
    lock_ctx : Callable[[Path], ContextManager]
        Context manager for file-level locking. Defaults to project's
        locks.sqlite_lock() if available, else no-op.
    now_wall : Callable[[], float]
        Function returning current wall-clock time (default: time.time).
    now_mono : Callable[[], float]
        Function returning current monotonic time (default: time.monotonic).

    Attributes
    ----------
    db_path : Path
        Database file location.
    lock_ctx : Callable
        Lock context manager factory.
    now_wall : Callable
        Wall-clock time provider.
    now_mono : Callable
        Monotonic time provider.
    _conn : sqlite3.Connection
        Persistent connection to database (autocommit mode).

    Raises
    ------
    RuntimeError
        If database initialization fails.
    """

    db_path: Path
    lock_ctx: Callable[[Path], ContextManager] = default_lock  # type: ignore[assignment]
    now_wall: Callable[[], float] = time.time
    now_mono: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        """Initialize database connection and create schema if needed."""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection in autocommit mode (no isolation level)
        self._conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # autocommit mode
            check_same_thread=False,  # allow use across threads
            detect_types=0,
        )

        # Initialize schema
        cursor = self._conn.cursor()
        for stmt in _DDL.strip().split(";\n"):
            if stmt.strip():
                cursor.execute(stmt)
        self._conn.commit()

    # ── CooldownStore API (Protocol) ───────────────────────────────────────

    def get_until(self, host: str) -> Optional[float]:
        """
        Retrieve cooldown deadline for a host, converted to monotonic time.

        Queries the database for an active cooldown. If found and still valid,
        converts the stored wall-clock deadline to monotonic time.
        If expired, deletes the entry and returns None.

        Parameters
        ----------
        host : str
            Hostname (normalized, lowercase).

        Returns
        -------
        Optional[float]
            Monotonic deadline if cooldown is active and in the future, else None.

        Notes
        -----
        - Expired entries are automatically cleaned up
        - Conversion: monotonic_deadline = now_mono + max(0, until_wall - now_wall)
        """
        now_w = self.now_wall()
        now_m = self.now_mono()

        with self.lock_ctx(self.db_path):
            row = self._conn.execute(
                "SELECT until_wall FROM breaker_cooldowns WHERE host=?",
                (host,),
            ).fetchone()

        if not row:
            return None

        until_wall = float(row[0])
        if until_wall <= now_w:
            # Expired — clean it up (best-effort, no lock needed)
            self.clear(host)
            return None

        # Convert wall → monotonic
        return now_m + max(0.0, until_wall - now_w)

    def set_until(self, host: str, until_monotonic: float, reason: str) -> None:
        """
        Write or update a cooldown deadline for a host.

        Converts the monotonic deadline to wall-clock time, then stores it.
        Uses INSERT OR REPLACE for atomicity.

        Parameters
        ----------
        host : str
            Hostname (normalized, lowercase).
        until_monotonic : float
            Monotonic deadline (time.monotonic() + delta).
        reason : str
            Reason for cooldown ("retry-after", "rolling-window", etc.).
            Truncated to 128 chars for safety.

        Notes
        -----
        - Conversion: until_wall = now_wall + max(0, until_monotonic - now_mono)
        - Timestamp (updated_at) is always set to current wall time
        """
        now_w = self.now_wall()
        now_m = self.now_mono()

        # Convert monotonic → wall
        until_wall = now_w + max(0.0, until_monotonic - now_m)

        with self.lock_ctx(self.db_path):
            self._conn.execute(
                """
                INSERT INTO breaker_cooldowns(host, until_wall, reason, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(host) DO UPDATE SET
                    until_wall=excluded.until_wall,
                    reason=excluded.reason,
                    updated_at=excluded.updated_at
                """,
                (host, float(until_wall), str(reason)[:128], now_w),
            )

    def clear(self, host: str) -> None:
        """
        Delete a cooldown entry for a host.

        Parameters
        ----------
        host : str
            Hostname (normalized, lowercase).
        """
        with self.lock_ctx(self.db_path):
            self._conn.execute("DELETE FROM breaker_cooldowns WHERE host=?", (host,))

    # ── Maintenance ───────────────────────────────────────────────────────────

    def prune_expired(self) -> int:
        """
        Delete all expired cooldown entries (older than 1 second).

        Returns
        -------
        int
            Number of rows deleted.

        Notes
        -----
        - Called periodically to reclaim disk space
        - Expired entries are also cleaned on-read via get_until()
        """
        now_w = self.now_wall()
        with self.lock_ctx(self.db_path):
            cur = self._conn.execute(
                "DELETE FROM breaker_cooldowns WHERE until_wall < ?",
                (now_w - 1.0,),
            )
            return cur.rowcount or 0

    def close(self) -> None:
        """Close the database connection. (Best-effort cleanup.)"""
        try:  # pragma: no cover
            self._conn.commit()
        finally:  # pragma: no cover
            self._conn.close()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_all_until(self) -> dict[str, tuple[float, str]]:
        """
        Get all current cooldown entries (for debugging).

        Returns
        -------
        dict[str, tuple[float, str]]
            Map of host → (until_wall_timestamp, reason).

        Notes
        -----
        - Times are wall-clock (UTC epoch), not monotonic
        - Useful for observability dashboards or CLI status
        """
        with self.lock_ctx(self.db_path):
            rows = self._conn.execute(
                "SELECT host, until_wall, reason FROM breaker_cooldowns"
            ).fetchall()
        return {host: (until_wall, reason) for host, until_wall, reason in rows}
