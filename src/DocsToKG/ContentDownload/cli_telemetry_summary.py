# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.cli_telemetry_summary",
#   "purpose": "CLI for telemetry summary and SLO evaluation.",
#   "sections": [
#     {
#       "id": "load-one",
#       "name": "load_one",
#       "anchor": "function-load-one",
#       "kind": "function"
#     },
#     {
#       "id": "summarize",
#       "name": "summarize",
#       "anchor": "function-summarize",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI for telemetry summary and SLO evaluation.

Computes SLIs from telemetry database and evaluates against SLO targets.
Exits with code 1 if any SLO fails, 0 if all pass.

Usage:
    python -m DocsToKG.ContentDownload.cli_telemetry_summary --db telemetry.sqlite --run <run_id>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

# SLO targets (tune to your environment)
SLO = {
    "yield_pct_min": 85.0,
    "ttfp_p50_ms_max": 3000,
    "ttfp_p95_ms_max": 20000,
    "cache_hit_pct_min": 60.0,
    "rate_delay_p95_ms_max": 250,  # for metadata
    "http429_pct_max": 2.0,
    "corruption_max": 0,
}


def load_one(conn: sqlite3.Connection, sql: str, params: tuple) -> float | None:
    """Load single value from query result."""
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def summarize(db_path: str, run_id: str) -> int:
    """Compute SLIs and evaluate SLOs.

    Parameters
    ----------
    db_path : str
        Path to telemetry SQLite database
    run_id : str
        Run identifier

    Returns
    -------
    int
        0 if all SLOs pass, 1 if any fail
    """
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row

    # Yield
    y = cx.execute(
        """
        SELECT COUNT(*) AS tot,
               SUM(CASE WHEN sha256 IS NOT NULL AND final_path IS NOT NULL THEN 1 ELSE 0 END) AS ok
        FROM downloads WHERE run_id=?
        """,
        (run_id,),
    ).fetchone()
    yield_pct = 100.0 * (y["ok"] or 0) / max(1, (y["tot"] or 1))

    # TTFP p50/p95 (ms)
    ttfp = (
        cx.execute(
            """
        WITH s AS (
          SELECT artifact_id, MIN(ts) ts_s FROM fallback_attempts
          WHERE run_id=? AND outcome='success' GROUP BY artifact_id
        ),
        f AS (
          SELECT artifact_id, MIN(ts) ts_f FROM fallback_attempts
          WHERE run_id=? GROUP BY artifact_id
        ),
        d AS (
          SELECT (s.ts_s - f.ts_f)*1000.0 ms FROM s JOIN f USING(artifact_id)
        )
        SELECT
          (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.50 AS INT) FROM d)) p50,
          (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.95 AS INT) FROM d)) p95
        """,
            (run_id, run_id),
        ).fetchone()
        or {"p50": None, "p95": None}
    )

    # Cache hit (metadata)
    cache_hit_pct = (
        load_one(
            cx,
            """
            SELECT 100.0*SUM(CASE WHEN from_cache=1 THEN 1 ELSE 0 END)/COUNT(*)
            FROM http_events WHERE run_id=? AND role='metadata'
            """,
            (run_id,),
        )
        or 0.0
    )

    # Rate delay p95 (metadata)
    rate_p95 = (
        load_one(
            cx,
            """
            WITH x AS (
              SELECT rate_delay_ms FROM http_events
              WHERE run_id=? AND role='metadata' AND rate_delay_ms IS NOT NULL
            )
            SELECT (SELECT rate_delay_ms FROM x ORDER BY rate_delay_ms LIMIT 1
                    OFFSET (SELECT CAST(COUNT(*)*0.95 AS INT) FROM x))
            """,
            (run_id,),
        )
        or 0
    )

    # HTTP 429 ratio (net only)
    http429 = (
        load_one(
            cx,
            """
            SELECT 100.0*SUM(CASE WHEN status=429 THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN from_cache!=1 THEN 1 ELSE 0 END),0)
            FROM http_events WHERE run_id=?
            """,
            (run_id,),
        )
        or 0.0
    )

    # Dedupe saved (MB)
    saved_mb = (
        load_one(
            cx,
            """
            SELECT SUM(CASE dedupe_action
                       WHEN 'hardlink' THEN content_length
                       WHEN 'copy'     THEN content_length
                       WHEN 'skipped'  THEN content_length
                       ELSE 0 END)/(1024.0*1024.0)
            FROM downloads WHERE run_id=? AND content_length IS NOT NULL
            """,
            (run_id,),
        )
        or 0.0
    )

    # Corruption (should be zero)
    corruption = (
        load_one(
            cx,
            "SELECT COUNT(*) FROM downloads WHERE run_id=? AND (final_path IS NULL OR sha256 IS NULL)",
            (run_id,),
        )
        or 0
    )

    summary = {
        "run_id": run_id,
        "yield_pct": round(yield_pct, 2),
        "ttfp_p50_ms": int(ttfp["p50"] or 0),
        "ttfp_p95_ms": int(ttfp["p95"] or 0),
        "cache_hit_pct": round(cache_hit_pct, 1),
        "rate_delay_p95_ms": int(rate_p95),
        "http_429_pct": round(http429, 2),
        "dedupe_saved_mb": round(saved_mb, 2),
        "corruption_count": int(corruption),
        "finished_at": time.time(),
    }
    print(json.dumps(summary, indent=2))

    # SLO evaluation → nonzero exit on fail
    fail = (
        summary["yield_pct"] < SLO["yield_pct_min"]
        or summary["ttfp_p50_ms"] > SLO["ttfp_p50_ms_max"]
        or summary["ttfp_p95_ms"] > SLO["ttfp_p95_ms_max"]
        or summary["cache_hit_pct"] < SLO["cache_hit_pct_min"]
        or summary["rate_delay_p95_ms"] > SLO["rate_delay_p95_ms_max"]
        or summary["http_429_pct"] > SLO["http429_pct_max"]
        or summary["corruption_count"] > SLO["corruption_max"]
    )

    # Print SLO pass/fail
    print("\n" + "=" * 80)
    print("SLO EVALUATION")
    print("=" * 80)
    print(
        f"Yield:              {summary['yield_pct']:.1f}% (min {SLO['yield_pct_min']:.1f}%) - {'✅ PASS' if summary['yield_pct'] >= SLO['yield_pct_min'] else '❌ FAIL'}"
    )
    print(
        f"TTFP p50:           {summary['ttfp_p50_ms']} ms (max {SLO['ttfp_p50_ms_max']}) - {'✅ PASS' if summary['ttfp_p50_ms'] <= SLO['ttfp_p50_ms_max'] else '❌ FAIL'}"
    )
    print(
        f"TTFP p95:           {summary['ttfp_p95_ms']} ms (max {SLO['ttfp_p95_ms_max']}) - {'✅ PASS' if summary['ttfp_p95_ms'] <= SLO['ttfp_p95_ms_max'] else '❌ FAIL'}"
    )
    print(
        f"Cache hit:          {summary['cache_hit_pct']:.1f}% (min {SLO['cache_hit_pct_min']:.1f}%) - {'✅ PASS' if summary['cache_hit_pct'] >= SLO['cache_hit_pct_min'] else '❌ FAIL'}"
    )
    print(
        f"Rate delay p95:     {summary['rate_delay_p95_ms']} ms (max {SLO['rate_delay_p95_ms_max']}) - {'✅ PASS' if summary['rate_delay_p95_ms'] <= SLO['rate_delay_p95_ms_max'] else '❌ FAIL'}"
    )
    print(
        f"HTTP 429 ratio:     {summary['http_429_pct']:.2f}% (max {SLO['http429_pct_max']:.2f}%) - {'✅ PASS' if summary['http_429_pct'] <= SLO['http429_pct_max'] else '❌ FAIL'}"
    )
    print(
        f"Corruption count:   {summary['corruption_count']} (max {SLO['corruption_max']}) - {'✅ PASS' if summary['corruption_count'] <= SLO['corruption_max'] else '❌ FAIL'}"
    )
    print("=" * 80)

    return 1 if fail else 0


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser("telemetry-summary")
    ap.add_argument("--db", required=True, help="Path to telemetry SQLite")
    ap.add_argument("--run", required=True, help="Run ID")
    args = ap.parse_args()
    return summarize(args.db, args.run)


if __name__ == "__main__":
    sys.exit(main())
