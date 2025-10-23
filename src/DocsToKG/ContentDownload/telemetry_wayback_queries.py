"""Wayback telemetry query helpers.

This module intentionally keeps the SQL required for querying the
Wayback-specific telemetry tables close to the Python helpers so that the
tests can exercise both the schema definition and the analytical helpers in a
single place.

Only a very small subset of the original production module is reproduced in
this kata-sized repository: just enough structure to exercise
``rate_smoothing_p95``.  The helper relies on the ``wayback_rate_events``
table, which records rate limiter interactions per host/role pair.  The tests
assert that the percentile calculation only includes entries for the requested
role, which means the schema must persist the role information explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sqlite3
from typing import Optional, Sequence


@dataclass(frozen=True)
class WaybackTelemetrySchema:
    """In-memory representation of the telemetry tables used in tests.

    The real project maintains a richer schema (multiple tables and migration
    handling).  For the purposes of the exercises in this repository we only
    need the ``wayback_rate_events`` table.  Capturing the schema as data makes
    it easy for tests to assert that we persist the ``role`` column which is
    required for filtering in :func:`rate_smoothing_p95`.
    """

    rate_events: Sequence[str]


WAYBACK_SCHEMA = WaybackTelemetrySchema(
    rate_events=(
        """
        CREATE TABLE IF NOT EXISTS wayback_rate_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            host TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            delay_ms INTEGER,
            max_delay_ms INTEGER
        );
        """.strip(),
        "CREATE INDEX IF NOT EXISTS idx_wayback_rate_run_role ON wayback_rate_events(run_id, role);",
        "CREATE INDEX IF NOT EXISTS idx_wayback_rate_host_role ON wayback_rate_events(host, role);",
    ),
)


def ensure_schema(conn: sqlite3.Connection, schema: WaybackTelemetrySchema = WAYBACK_SCHEMA) -> None:
    """Ensure the minimal Wayback telemetry schema exists for ``conn``.

    Parameters
    ----------
    conn:
        The SQLite connection that should contain the Wayback telemetry tables.
    schema:
        Schema description to materialise.  Tests can provide a tailored
        schema, but by default the module-level :data:`WAYBACK_SCHEMA` is used.
    """

    for statement in schema.rate_events:
        conn.execute(statement)


def _percentile(sorted_values: Sequence[int], percentile: float) -> Optional[int]:
    """Return the percentile value using the nearest-rank method."""

    if not sorted_values:
        return None
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 1:
        return sorted_values[-1]

    index = max(int(math.ceil(percentile * len(sorted_values))) - 1, 0)
    return sorted_values[index]


def rate_smoothing_p95(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    role: str,
) -> Optional[int]:
    """Compute the P95 rate limiter delay for ``run_id`` and ``role``.

    Only ``acquire`` actions that include a non-null ``delay_ms`` field are
    considered.  The underlying query explicitly filters by ``role`` so that
    percentile calculations remain isolated per limiter role (metadata,
    landing, or artifact).
    """

    role_normalised = role.strip().lower()

    cursor = conn.execute(
        """
        SELECT delay_ms
        FROM wayback_rate_events
        WHERE run_id = :run_id
          AND role = :role
          AND action = 'acquire'
          AND delay_ms IS NOT NULL
        ORDER BY delay_ms ASC
        """,
        {"run_id": run_id, "role": role_normalised},
    )
    delays = [int(row[0]) for row in cursor.fetchall()]
    return _percentile(delays, 0.95)

