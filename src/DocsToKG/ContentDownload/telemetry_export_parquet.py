# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.telemetry_export_parquet",
#   "purpose": "DuckDB-based Parquet exporter for long-term trend analysis.",
#   "sections": [
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DuckDB-based Parquet exporter for long-term trend analysis.

Exports SQLite telemetry tables to compressed Parquet format for efficient
long-term storage and analysis with DuckDB, Polars, or Pandas.

Usage:
    python -m DocsToKG.ContentDownload.telemetry_export_parquet \\
        --sqlite /path/to/telemetry.sqlite \\
        --out /path/to/parquet/
"""

from __future__ import annotations

import argparse
import pathlib

try:
    import duckdb
except ImportError as e:
    raise RuntimeError(
        "duckdb is required for telemetry_export_parquet; install with: pip install duckdb"
    ) from e


TABLES = [
    "http_events",
    "rate_events",
    "breaker_transitions",
    "fallback_attempts",
    "downloads",
    "run_summary",
]


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser("export-parquet")
    ap.add_argument("--sqlite", required=True, help="Path to telemetry sqlite")
    ap.add_argument("--out", required=True, help="Output directory for parquet files")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"ATTACH '{args.sqlite}' AS t (TYPE sqlite)")

    for tbl in TABLES:
        # Skip if table missing
        exists = con.execute(
            f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'"
        ).fetchone()[0]
        if not exists:
            print(f"[export] skip missing table: {tbl}")
            continue

        dst = out / f"{tbl}.parquet"
        con.execute(
            f"COPY (SELECT * FROM t.{tbl}) TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        print(f"[export] wrote {dst}")

    con.execute("DETACH t")
    con.close()
    print(f"[export] complete: {out}")


if __name__ == "__main__":
    main()
