"""Run summary builders and console reporting helpers.

Responsibilities
----------------
- Provide the :class:`RunResult` dataclass that packages the canonical metrics
  emitted by :mod:`DocsToKG.ContentDownload.runner` for downstream consumers
  (CLI, agents, tests).
- Assemble structured summary payloads via :func:`build_summary_record`, ready
  to be written into JSON/CSV manifest metrics files or emitted to telemetry
  sinks.
- Expose :func:`emit_console_summary` to render human-friendly progress reports
  that mirror the metrics stored on disk, making smoke tests easy to interpret.

Design Notes
------------
- Functions here accept plain primitives so they can be imported without
  pulling in heavyweight dependencies; the CLI reuses them directly.
- The console renderer intentionally mirrors the JSON payload layout so that
  parsing logs or inspecting structured data yields the same information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = [
    "RunResult",
    "build_summary_record",
    "emit_console_summary",
]


@dataclass
class RunResult:
    """Aggregated metrics captured at the end of a content download run."""

    run_id: str
    processed: int
    saved: int
    html_only: int
    xml_only: int
    skipped: int
    worker_failures: int
    bytes_downloaded: int
    summary: Dict[str, Any]
    summary_record: Dict[str, Any]


def build_summary_record(
    *,
    run_id: str,
    processed: int,
    saved: int,
    html_only: int,
    xml_only: int,
    skipped: int,
    worker_failures: int,
    bytes_downloaded: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the structured run summary record persisted to metrics sinks."""

    reason_totals = summary.get("reason_totals", {})
    classification_totals = summary.get("classification_totals", {})
    return {
        "run_id": run_id,
        "processed": processed,
        "saved": saved,
        "html_only": html_only,
        "xml_only": xml_only,
        "skipped": skipped,
        "worker_failures": worker_failures,
        "bytes_downloaded": bytes_downloaded,
        "classification_totals": dict(classification_totals),
        "reason_totals": dict(reason_totals),
        "resolvers": summary,
    }


def emit_console_summary(result: RunResult, *, dry_run: bool) -> None:
    """Pretty-print the run summary to stdout."""

    print(
        f"\nDone. Processed {result.processed} works, saved {result.saved} PDFs, "
        f"HTML-only {result.html_only}, XML-only {result.xml_only}, skipped {result.skipped}."
    )
    print(f"Total bytes downloaded {result.bytes_downloaded}.")
    if dry_run:
        print("DRY RUN: no files written, resolver coverage only.")
    if result.worker_failures:
        print(f"Worker exceptions encountered: {result.worker_failures}")

    summary = result.summary
    print("Resolver summary:")
    for key in ("attempts", "successes", "html", "xml", "skips", "failures"):
        values = summary.get(key, {})
        if values:
            print(f"  {key}: {values}")

    latency_summary = summary.get("latency_ms", {})
    if latency_summary:
        print("  latency_ms:")
        for resolver_name, stats in latency_summary.items():
            mean_ms = stats.get("mean_ms", 0.0)
            p95_ms = stats.get("p95_ms", 0.0)
            max_ms = stats.get("max_ms", 0.0)
            count = stats.get("count", 0)
            print(
                f"    {resolver_name}: count={count} mean={mean_ms:.1f}ms "
                f"p95={p95_ms:.1f}ms max={max_ms:.1f}ms"
            )

    status_counts = summary.get("status_counts", {})
    if status_counts:
        print("  status_counts:")
        for resolver_name, counts in status_counts.items():
            print(f"    {resolver_name}: {counts}")

    error_reasons = summary.get("error_reasons", {})
    if error_reasons:
        print("  top_error_reasons:")
        for resolver_name, items in error_reasons.items():
            formatted = ", ".join(f"{entry['reason']} ({entry['count']})" for entry in items)
            print(f"    {resolver_name}: {formatted}")

    limiter_summary = summary.get("rate_limiter", {})
    if limiter_summary:
        backend = limiter_summary.get("backend")
        print("Rate limiter:")
        if backend:
            print(f"  backend: {backend}")
        metrics = limiter_summary.get("metrics", {})
        for host, roles in metrics.items():
            for role, stats in roles.items():
                acquire = stats.get("acquire_total", 0)
                blocked = stats.get("blocked_total", 0)
                avg_wait = stats.get("wait_ms_avg", 0.0)
                max_wait = stats.get("wait_ms_max", 0.0)
                print(
                    f"  {host}.{role}: acquire={acquire} blocked={blocked} "
                    f"avg_wait={avg_wait:.1f}ms max_wait={max_wait:.1f}ms"
                )
