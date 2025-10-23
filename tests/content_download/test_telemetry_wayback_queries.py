"""Tests for the Wayback telemetry query helpers."""

from __future__ import annotations

import sqlite3

import pytest

from DocsToKG.ContentDownload.telemetry_wayback_queries import (
    WAYBACK_SCHEMA,
    ensure_schema,
    rate_smoothing_p95,
)


@pytest.fixture()
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    ensure_schema(connection)
    return connection


def _insert_rate_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    host: str,
    role: str,
    action: str,
    delay_ms: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO wayback_rate_events (run_id, host, role, action, delay_ms)
        VALUES (:run_id, :host, :role, :action, :delay_ms)
        """,
        {
            "run_id": run_id,
            "host": host,
            "role": role,
            "action": action,
            "delay_ms": delay_ms,
        },
    )


def test_schema_includes_role_column(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info('wayback_rate_events')")
    columns = {row[1] for row in cursor.fetchall()}
    assert "role" in columns  # regression guard


def test_rate_smoothing_p95_filters_by_role(conn: sqlite3.Connection) -> None:
    # Metadata role events for run-1
    for delay in [50, 120, 180, 200, 300]:
        _insert_rate_event(
            conn,
            run_id="run-1",
            host="api.example.com",
            role="metadata",
            action="acquire",
            delay_ms=delay,
        )

    # Artifact role events for the same run - significantly higher delays
    for delay in [750, 900]:
        _insert_rate_event(
            conn,
            run_id="run-1",
            host="api.example.com",
            role="artifact",
            action="acquire",
            delay_ms=delay,
        )

    # Non-acquire actions should be ignored even if they match the role
    _insert_rate_event(
        conn,
        run_id="run-1",
        host="api.example.com",
        role="metadata",
        action="block",
        delay_ms=400,
    )

    # A different run with the same role should not bleed into the percentile
    _insert_rate_event(
        conn,
        run_id="run-2",
        host="api.example.com",
        role="metadata",
        action="acquire",
        delay_ms=1000,
    )

    metadata_p95 = rate_smoothing_p95(conn, "run-1", role="metadata")
    artifact_p95 = rate_smoothing_p95(conn, "run-1", role="artifact")

    # Metadata percentile should ignore artifact delays
    assert metadata_p95 == 300
    # Artifact percentile should only consider artifact rows
    assert artifact_p95 == 900


def test_rate_smoothing_p95_returns_none_when_no_rows(conn: sqlite3.Connection) -> None:
    assert rate_smoothing_p95(conn, "missing-run", role="metadata") is None

