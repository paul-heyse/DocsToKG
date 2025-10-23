# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.cli_telemetry",
#   "purpose": "CLI subcommands for telemetry inspection, SLO evaluation, and export",
#   "sections": [
#     {
#       "id": "install-telemetry-cli",
#       "name": "install_telemetry_cli",
#       "anchor": "function-install-telemetry-cli",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-one",
#       "name": "_fetch_one",
#       "anchor": "function-fetch-one",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-summary",
#       "name": "_cmd_summary",
#       "anchor": "function-cmd-summary",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-export",
#       "name": "_cmd_export",
#       "anchor": "function-cmd-export",
#       "kind": "function"
#     },
#     {
#       "id": "cmd-query",
#       "name": "_cmd_query",
#       "anchor": "function-cmd-query",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI subcommands for telemetry inspection, SLO evaluation, and export.

This module provides operator-friendly commands to:
- Evaluate SLOs from a telemetry database
- Export telemetry to Parquet for long-term trending
- Query telemetry tables for debugging and analysis

Typical Usage:
    # Install into argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=False)
    install_telemetry_cli(subparsers)

    # Use from CLI
    python -m DocsToKG.ContentDownload.cli telemetry summary --db manifest.sqlite3 --run <run_id>
    python -m DocsToKG.ContentDownload.cli telemetry export --db manifest.sqlite3 --out parquet/
    python -m DocsToKG.ContentDownload.cli telemetry query --db manifest.sqlite3 --query "SELECT COUNT(*) FROM http_events"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

__all__ = ["install_telemetry_cli"]


# SLO targets (configurable in future)
_DEFAULT_SLO = {
    "yield_pct_min": 85.0,
    "ttfp_p50_ms_max": 3000,
    "ttfp_p95_ms_max": 20000,
    "cache_hit_pct_min": 60.0,
    "rate_delay_p95_ms_max": 250,
    "http429_pct_max": 2.0,
    "corruption_max": 0,
}


def install_telemetry_cli(subparsers: argparse._SubParsersAction) -> None:
    """Install telemetry CLI subcommands into argument parser.

    Adds:
      - telemetry summary --db DB --run RUN_ID - evaluate SLOs
      - telemetry export --db DB --out OUT_DIR - export to Parquet
      - telemetry query --db DB --query SQL - run SQL query

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Parent subparsers action from ArgumentParser.add_subparsers()
    """
    p = subparsers.add_parser("telemetry", help="Telemetry inspection and export commands")
    sp = p.add_subparsers(dest="telemetry_cmd", required=True)

    # summary
    ps = sp.add_parser("summary", help="Evaluate SLOs and compute SLIs from telemetry database")
    ps.add_argument("--db", type=Path, required=True, help="Path to telemetry SQLite database")
    ps.add_argument("--run", type=str, required=True, help="Run ID to summarize")
    ps.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    ps.set_defaults(func=_cmd_summary)

    # export
    pe = sp.add_parser("export", help="Export telemetry tables to Parquet for trending")
    pe.add_argument("--db", type=Path, required=True, help="Path to telemetry SQLite database")
    pe.add_argument("--out", type=Path, required=True, help="Output directory for Parquet files")
    pe.set_defaults(func=_cmd_export)

    # query
    pq = sp.add_parser("query", help="Execute SQL query against telemetry database")
    pq.add_argument("--db", type=Path, required=True, help="Path to telemetry SQLite database")
    pq.add_argument("--query", type=str, required=True, help="SQL query to execute")
    pq.add_argument(
        "--format",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table)",
    )
    pq.set_defaults(func=_cmd_query)


def _fetch_one(cx: sqlite3.Connection, sql: str, params: tuple = ()) -> float | None:
    """Fetch single value from query result."""
    row = cx.execute(sql, params).fetchone()
    return row[0] if row and row[0] is not None else None


def _cmd_summary(args: argparse.Namespace) -> int:
    """Evaluate SLOs and compute SLIs from telemetry database.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: db (Path), run (str), output_format (str)

    Returns
    -------
    int
        0 if all SLOs pass, 1 if any fail
    """
    db_path = args.db
    run_id = args.run

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        return 1

    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row

    try:
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
            _fetch_one(
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
            _fetch_one(
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
            _fetch_one(
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
            _fetch_one(
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
            _fetch_one(
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

        # Output
        if args.output_format == "json":
            print(json.dumps(summary, indent=2))
        else:
            # Text output
            print(json.dumps(summary, indent=2))

        # SLO evaluation
        fail = (
            summary["yield_pct"] < _DEFAULT_SLO["yield_pct_min"]
            or summary["ttfp_p50_ms"] > _DEFAULT_SLO["ttfp_p50_ms_max"]
            or summary["ttfp_p95_ms"] > _DEFAULT_SLO["ttfp_p95_ms_max"]
            or summary["cache_hit_pct"] < _DEFAULT_SLO["cache_hit_pct_min"]
            or summary["rate_delay_p95_ms"] > _DEFAULT_SLO["rate_delay_p95_ms_max"]
            or summary["http_429_pct"] > _DEFAULT_SLO["http429_pct_max"]
            or summary["corruption_count"] > _DEFAULT_SLO["corruption_max"]
        )

        # Print SLO evaluation
        print("\n" + "=" * 80)
        print("SLO EVALUATION")
        print("=" * 80)
        print(
            f"Yield:              {summary['yield_pct']:.1f}% (min {_DEFAULT_SLO['yield_pct_min']:.1f}%) - {'✅ PASS' if summary['yield_pct'] >= _DEFAULT_SLO['yield_pct_min'] else '❌ FAIL'}"
        )
        print(
            f"TTFP p50:           {summary['ttfp_p50_ms']} ms (max {_DEFAULT_SLO['ttfp_p50_ms_max']}) - {'✅ PASS' if summary['ttfp_p50_ms'] <= _DEFAULT_SLO['ttfp_p50_ms_max'] else '❌ FAIL'}"
        )
        print(
            f"TTFP p95:           {summary['ttfp_p95_ms']} ms (max {_DEFAULT_SLO['ttfp_p95_ms_max']}) - {'✅ PASS' if summary['ttfp_p95_ms'] <= _DEFAULT_SLO['ttfp_p95_ms_max'] else '❌ FAIL'}"
        )
        print(
            f"Cache hit:          {summary['cache_hit_pct']:.1f}% (min {_DEFAULT_SLO['cache_hit_pct_min']:.1f}%) - {'✅ PASS' if summary['cache_hit_pct'] >= _DEFAULT_SLO['cache_hit_pct_min'] else '❌ FAIL'}"
        )
        print(
            f"Rate delay p95:     {summary['rate_delay_p95_ms']} ms (max {_DEFAULT_SLO['rate_delay_p95_ms_max']}) - {'✅ PASS' if summary['rate_delay_p95_ms'] <= _DEFAULT_SLO['rate_delay_p95_ms_max'] else '❌ FAIL'}"
        )
        print(
            f"HTTP 429 ratio:     {summary['http_429_pct']:.2f}% (max {_DEFAULT_SLO['http429_pct_max']:.2f}%) - {'✅ PASS' if summary['http_429_pct'] <= _DEFAULT_SLO['http429_pct_max'] else '❌ FAIL'}"
        )
        print(
            f"Corruption count:   {summary['corruption_count']} (max {_DEFAULT_SLO['corruption_max']}) - {'✅ PASS' if summary['corruption_count'] <= _DEFAULT_SLO['corruption_max'] else '❌ FAIL'}"
        )
        print("=" * 80)

        return 1 if fail else 0

    finally:
        cx.close()


def _cmd_export(args: argparse.Namespace) -> int:
    """Export telemetry tables to Parquet for trending.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: db (Path), out (Path)

    Returns
    -------
    int
        0 on success, 1 on failure
    """
    db_path = args.db
    out_dir = args.out

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        return 1

    try:
        import duckdb
    except ImportError:
        print(
            "ERROR: duckdb is required for export. Install with: pip install duckdb",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    tables = [
        "http_events",
        "rate_events",
        "breaker_transitions",
        "fallback_attempts",
        "downloads",
        "run_summary",
    ]

    try:
        con = duckdb.connect()
        con.execute(f"ATTACH '{db_path.as_posix()}' AS t (TYPE sqlite)")

        for tbl in tables:
            # Check if table exists
            exists = con.execute(
                f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'"
            ).fetchone()[0]
            if not exists:
                print(f"[export] skip missing table: {tbl}")
                continue

            dst = out_dir / f"{tbl}.parquet"
            con.execute(
                f"COPY (SELECT * FROM t.{tbl}) TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            print(f"[export] wrote {dst}")

        con.execute("DETACH t")
        con.close()
        print(f"[export] complete: {out_dir}")
        return 0

    except Exception as e:
        print(f"ERROR during export: {e}", file=sys.stderr)
        return 1


def _cmd_query(args: argparse.Namespace) -> int:
    """Execute SQL query against telemetry database.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: db (Path), query (str), format (str)

    Returns
    -------
    int
        0 on success, 1 on failure
    """
    db_path = args.db

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        return 1

    try:
        cx = sqlite3.connect(db_path)
        cx.row_factory = sqlite3.Row
        cur = cx.execute(args.query)
        rows = cur.fetchall()

        if args.format == "json":
            output = [dict(row) for row in rows]
            print(json.dumps(output, indent=2, default=str))
        else:
            # Table format
            if not rows:
                print("No results.")
            else:
                cols = [desc[0] for desc in cur.description]
                print(" | ".join(cols))
                print("-" * (sum(len(c) for c in cols) + 3 * len(cols)))
                for row in rows:
                    print(" | ".join(str(row[c]) for c in cols))

        cx.close()
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
