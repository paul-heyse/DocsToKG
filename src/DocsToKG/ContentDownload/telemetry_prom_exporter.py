# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.telemetry_prom_exporter",
#   "purpose": "SQLite \u2192 Prometheus exporter for telemetry metrics (Grafana-ready).",
#   "sections": [
#     {
#       "id": "fetchone",
#       "name": "_fetchone",
#       "anchor": "function-fetchone",
#       "kind": "function"
#     },
#     {
#       "id": "latest-run-id",
#       "name": "_latest_run_id",
#       "anchor": "function-latest-run-id",
#       "kind": "function"
#     },
#     {
#       "id": "compute-run-summary",
#       "name": "_compute_run_summary",
#       "anchor": "function-compute-run-summary",
#       "kind": "function"
#     },
#     {
#       "id": "emit-host-429",
#       "name": "_emit_host_429",
#       "anchor": "function-emit-host-429",
#       "kind": "function"
#     },
#     {
#       "id": "emit-breaker-opens",
#       "name": "_emit_breaker_opens",
#       "anchor": "function-emit-breaker-opens",
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

"""SQLite → Prometheus exporter for telemetry metrics (Grafana-ready).

Polls the run database every N seconds and exposes metrics on /metrics endpoint.

Usage:
    python -m DocsToKG.ContentDownload.telemetry_prom_exporter \\
        --db /path/to/telemetry.sqlite \\
        --port 9108 \\
        --poll 10
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from typing import Any, Dict, Optional

try:
    from prometheus_client import Counter, Gauge, start_http_server
except ImportError as e:
    raise RuntimeError(
        "prometheus_client is required for telemetry_prom_exporter; "
        "install with: pip install prometheus_client"
    ) from e

# -------- Prom metrics (static registry) --------
RUN_YIELD_PCT = Gauge("docstokg_run_yield_pct", "Yield percent for a run", ["run_id"])
RUN_TTFP_MS = Gauge(
    "docstokg_run_ttfp_ms",
    "Time-to-first-PDF (ms) quantiles",
    ["run_id", "quantile"],
)
RUN_CACHE_HIT_PCT = Gauge("docstokg_run_cache_hit_pct", "Metadata cache hit percent", ["run_id"])
RUN_RATE_P95_MS = Gauge("docstokg_run_rate_delay_p95_ms", "Rate delay p95 (ms)", ["run_id", "role"])
HOST_HTTP429_RATIO = Gauge(
    "docstokg_host_http429_ratio", "HTTP 429 ratio (%) by host", ["run_id", "host"]
)
BREAKER_OPENS_TOTAL = Counter(
    "docstokg_breaker_open_events_total",
    "Breaker state changes to OPEN",
    ["run_id", "host"],
)
RUN_DEDUPE_SAVED_MB = Gauge(
    "docstokg_run_dedupe_saved_mb", "Dedupe savings in MB for a run", ["run_id"]
)
RUN_CORRUPTION_COUNT = Gauge(
    "docstokg_run_corruption_count",
    "Rows missing final/hash (should be 0)",
    ["run_id"],
)


def _fetchone(cx: sqlite3.Connection, sql: str, args: tuple = ()) -> Optional[Any]:
    """Fetch single value from query."""
    row = cx.execute(sql, args).fetchone()
    return row[0] if row and row[0] is not None else None


def _latest_run_id(cx: sqlite3.Connection) -> Optional[str]:
    """Get latest run_id from database."""
    # Prefer explicit run_summary ordering; fallback to max ts in http_events
    row = cx.execute("SELECT run_id FROM run_summary ORDER BY finished_at DESC LIMIT 1").fetchone()
    if row:
        return row[0]
    row = cx.execute("SELECT run_id FROM http_events ORDER BY ts DESC LIMIT 1").fetchone()
    return row[0] if row else None


def _compute_run_summary(cx: sqlite3.Connection, run_id: str) -> Dict[str, float]:
    """Compute SLIs for a run."""
    out: Dict[str, float] = {}

    # Yield
    tot, ok = cx.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN sha256 IS NOT NULL AND final_path IS NOT NULL THEN 1 ELSE 0 END)
        FROM downloads WHERE run_id=?
        """,
        (run_id,),
    ).fetchone()
    tot = tot or 0
    ok = ok or 0
    out["yield_pct"] = 100.0 * ok / max(1, tot)

    # TTFP p50/p95 from fallback_attempts
    row = (
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
          (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.50 AS INT) FROM d)) AS p50,
          (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.95 AS INT) FROM d)) AS p95
        """,
            (run_id, run_id),
        ).fetchone()
        or (0, 0)
    )
    out["ttfp_p50_ms"] = int(row[0] or 0)
    out["ttfp_p95_ms"] = int(row[1] or 0)

    # Cache hit (metadata)
    out["cache_hit_pct"] = (
        _fetchone(
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
    out["rate_p95_ms_meta"] = (
        _fetchone(
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

    # Dedupe saved MB
    out["saved_mb"] = (
        _fetchone(
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

    # Corruption count
    out["corruption"] = (
        _fetchone(
            cx,
            "SELECT COUNT(*) FROM downloads WHERE run_id=? AND (final_path IS NULL OR sha256 IS NULL)",
            (run_id,),
        )
        or 0
    )

    return out


def _emit_host_429(cx: sqlite3.Connection, run_id: str) -> None:
    """Emit per-host 429 ratios."""
    cur = cx.execute(
        """
        SELECT host,
               100.0*SUM(CASE WHEN status=429 THEN 1 ELSE 0 END)
               / NULLIF(SUM(CASE WHEN from_cache!=1 THEN 1 ELSE 0 END),0) AS pct
        FROM http_events WHERE run_id=? GROUP BY host
        """,
        (run_id,),
    )
    for host, pct in cur.fetchall():
        HOST_HTTP429_RATIO.labels(run_id=run_id, host=host).set(float(pct or 0.0))


def _emit_breaker_opens(cx: sqlite3.Connection, run_id: str, seen: Dict[tuple, int]) -> None:
    """Emit breaker open counters."""
    cur = cx.execute(
        """
        SELECT host, COUNT(*) FROM breaker_transitions
        WHERE run_id=? AND old_state LIKE '%CLOSED%' AND new_state LIKE '%OPEN%'
        GROUP BY host
        """,
        (run_id,),
    )
    for host, cnt in cur.fetchall():
        key = (run_id, host)
        prev = seen.get(key, 0)
        inc = int(cnt or 0) - prev
        if inc > 0:
            BREAKER_OPENS_TOTAL.labels(run_id=run_id, host=host).inc(inc)
            seen[key] = prev + inc


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser("docstokg-prom-exporter")
    ap.add_argument("--db", required=True, help="Path to telemetry sqlite")
    ap.add_argument("--port", type=int, default=9108, help="HTTP port for /metrics")
    ap.add_argument("--poll", type=float, default=10.0, help="Poll interval (s)")
    ap.add_argument("--run-id", help="If omitted, exporter uses latest run")
    args = ap.parse_args()

    start_http_server(args.port)
    print(f"[prom-exporter] listening on http://0.0.0.0:{args.port}/metrics")

    seen_breaker: Dict[tuple, int] = {}

    while True:
        try:
            cx = sqlite3.connect(args.db)
            cx.row_factory = sqlite3.Row

            run_id = args.run_id or _latest_run_id(cx)
            if run_id:
                # summary (from table if filled; otherwise compute)
                row = cx.execute("SELECT * FROM run_summary WHERE run_id=?", (run_id,)).fetchone()
                if row:
                    RUN_YIELD_PCT.labels(run_id).set(float(row["yield_pct"] or 0))
                    RUN_TTFP_MS.labels(run_id, "p50").set(int(row["ttfp_p50_ms"] or 0))
                    RUN_TTFP_MS.labels(run_id, "p95").set(int(row["ttfp_p95_ms"] or 0))
                    RUN_CACHE_HIT_PCT.labels(run_id).set(float(row["cache_hit_pct"] or 0))
                    RUN_RATE_P95_MS.labels(run_id, "metadata").set(
                        int(row["rate_delay_p95_ms"] or 0)
                    )
                    RUN_DEDUPE_SAVED_MB.labels(run_id).set(float(row["dedupe_saved_mb"] or 0))
                    RUN_CORRUPTION_COUNT.labels(run_id).set(int(row["corruption_count"] or 0))
                else:
                    s = _compute_run_summary(cx, run_id)
                    RUN_YIELD_PCT.labels(run_id).set(s["yield_pct"])
                    RUN_TTFP_MS.labels(run_id, "p50").set(s["ttfp_p50_ms"])
                    RUN_TTFP_MS.labels(run_id, "p95").set(s["ttfp_p95_ms"])
                    RUN_CACHE_HIT_PCT.labels(run_id).set(s["cache_hit_pct"])
                    RUN_RATE_P95_MS.labels(run_id, "metadata").set(s["rate_p95_ms_meta"])
                    RUN_DEDUPE_SAVED_MB.labels(run_id).set(s["saved_mb"])
                    RUN_CORRUPTION_COUNT.labels(run_id).set(s["corruption"])

                _emit_host_429(cx, run_id)
                _emit_breaker_opens(cx, run_id, seen_breaker)

            cx.close()
        except Exception as e:  # pragma: no cover
            # Keep exporter up; surface errors via logs
            print(f"[prom-exporter] poll error: {e}", flush=True)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
