# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.telemetry_wayback_queries",
#   "purpose": "Query helpers for Wayback telemetry analysis",
#   "sections": [
#     {
#       "id": "queryhelpers",
#       "name": "Query Helpers",
#       "anchor": "module-functions",
#       "kind": "functions"
#     }
#   ]
# }
# === END NAVMAP ===

"""
Query helpers for Wayback telemetry analysis.

This module provides ready-to-use functions for analyzing Wayback resolver
performance and effectiveness from SQLite telemetry databases.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def yield_by_path(db_path: Path, run_id: str) -> Dict[str, int]:
    """
    Get yield counts by discovery path (pdf_direct vs html_parse).

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        Dictionary with 'pdf_direct' and 'html_parse' counts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.source_mode, COUNT(*) as count
            FROM wayback_emits e
            JOIN wayback_attempts a ON e.attempt_id = a.attempt_id
            WHERE a.run_id = ?
            GROUP BY e.source_mode
        """,
            (run_id,),
        )

        results = {row["source_mode"]: row["count"] for row in cur.fetchall()}

        # Ensure both keys exist
        return {
            "pdf_direct": results.get("pdf_direct", 0),
            "html_parse": results.get("html_parse", 0),
        }
    finally:
        conn.close()


def p95_selection_latency(db_path: Path, run_id: str) -> Optional[int]:
    """
    Calculate P95 selection latency in milliseconds.

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        P95 latency in milliseconds, or None if no data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT total_duration_ms
            FROM wayback_attempts
            WHERE run_id = ? AND total_duration_ms IS NOT NULL
            ORDER BY total_duration_ms
        """,
            (run_id,),
        )

        durations = [row["total_duration_ms"] for row in cur.fetchall()]
        if not durations:
            return None

        # Calculate P95
        p95_index = int(len(durations) * 0.95)
        return durations[p95_index]
    finally:
        conn.close()


def skip_reasons(db_path: Path, run_id: str) -> List[Tuple[str, int]]:
    """
    Get skip reasons with counts.

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        List of (reason, count) tuples sorted by count descending
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.reason, COUNT(*) as count
            FROM wayback_skips s
            JOIN wayback_attempts a ON s.attempt_id = a.attempt_id
            WHERE a.run_id = ?
            GROUP BY s.reason
            ORDER BY count DESC
        """,
            (run_id,),
        )

        return [(row["reason"], row["count"]) for row in cur.fetchall()]
    finally:
        conn.close()


def cache_assist_rate(db_path: Path, run_id: str) -> float:
    """
    Calculate cache hit rate for discovery calls.

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        Cache hit rate as a float between 0.0 and 1.0
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN d.from_cache = 1 THEN 1 ELSE 0 END) as cached
            FROM wayback_discoveries d
            JOIN wayback_attempts a ON d.attempt_id = a.attempt_id
            WHERE a.run_id = ?
        """,
            (run_id,),
        )

        row = cur.fetchone()
        if not row or row["total"] == 0:
            return 0.0

        return row["cached"] / row["total"]
    finally:
        conn.close()


def rate_smoothing_p95(db_path: Path, run_id: str, role: str) -> Optional[int]:
    """
    Calculate P95 rate delay for a specific role.

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by
        role: Role to filter by (metadata, landing, artifact)

    Returns:
        P95 rate delay in milliseconds, or None if no data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        normalized_role = role.strip().lower()
        cur.execute(
            """
            SELECT d.rate_delay_ms
            FROM wayback_discoveries d
            JOIN wayback_attempts a ON d.attempt_id = a.attempt_id
            WHERE a.run_id = ?
              AND d.rate_delay_ms IS NOT NULL
              AND LOWER(COALESCE(d.rate_limiter_role, '')) = LOWER(?)
            ORDER BY d.rate_delay_ms
        """,
            (run_id, normalized_role),
        )

        delays = [row["rate_delay_ms"] for row in cur.fetchall()]
        if not delays:
            return None

        # Calculate P95
        p95_index = min(len(delays) - 1, max(0, int(len(delays) * 0.95)))
        return delays[p95_index]
    finally:
        conn.close()


def backoff_mean(db_path: Path, run_id: str) -> Optional[float]:
    """
    Calculate mean retry-after delay for 429/503 responses.

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        Mean retry-after delay in seconds, or None if no data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT AVG(d.retry_after_s) as mean_delay
            FROM wayback_discoveries d
            JOIN wayback_attempts a ON d.attempt_id = a.attempt_id
            WHERE a.run_id = ? AND d.retry_after_s IS NOT NULL
        """,
            (run_id,),
        )

        row = cur.fetchone()
        return row["mean_delay"] if row and row["mean_delay"] is not None else None
    finally:
        conn.close()


def wayback_yield(db_path: Path, run_id: str) -> float:
    """
    Calculate overall Wayback yield (emits / attempts).

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        Yield rate as a float between 0.0 and 1.0
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) as attempts,
                SUM(CASE WHEN result LIKE 'emitted%' THEN 1 ELSE 0 END) as emits
            FROM wayback_attempts
            WHERE run_id = ? AND end_ts IS NOT NULL
        """,
            (run_id,),
        )

        row = cur.fetchone()
        if not row or row["attempts"] == 0:
            return 0.0

        return row["emits"] / row["attempts"]
    finally:
        conn.close()


def run_summary(db_path: Path, run_id: str) -> Dict[str, Any]:
    """
    Generate a comprehensive summary for a run.

    Args:
        db_path: Path to SQLite database
        run_id: Run identifier to filter by

    Returns:
        Dictionary with all key metrics
    """
    return {
        "run_id": run_id,
        "yield": wayback_yield(db_path, run_id),
        "yield_by_path": yield_by_path(db_path, run_id),
        "p95_latency_ms": p95_selection_latency(db_path, run_id),
        "skip_reasons": skip_reasons(db_path, run_id),
        "cache_hit_rate": cache_assist_rate(db_path, run_id),
        "p95_rate_delay_ms": rate_smoothing_p95(db_path, run_id, "metadata"),
        "mean_backoff_s": backoff_mean(db_path, run_id),
    }
