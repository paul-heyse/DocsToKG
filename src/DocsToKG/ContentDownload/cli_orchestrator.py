# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.cli_orchestrator",
#   "purpose": "CLI commands for work orchestration (queue management)",
#   "sections": [
#     {
#       "id": "queue-enqueue",
#       "name": "queue_enqueue",
#       "anchor": "function-queue-enqueue",
#       "kind": "function"
#     },
#     {
#       "id": "queue-import",
#       "name": "queue_import",
#       "anchor": "function-queue-import",
#       "kind": "function"
#     },
#     {
#       "id": "queue-run",
#       "name": "queue_run",
#       "anchor": "function-queue-run",
#       "kind": "function"
#     },
#     {
#       "id": "queue-stats",
#       "name": "queue_stats",
#       "anchor": "function-queue-stats",
#       "kind": "function"
#     },
#     {
#       "id": "queue-retry-failed",
#       "name": "queue_retry_failed",
#       "anchor": "function-queue-retry-failed",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI commands for work orchestration and queue management.

This module provides Typer commands for:
- **enqueue**: Add artifacts to the work queue
- **import**: Bulk import from JSONL file
- **run**: Start orchestrator with worker pool
- **stats**: Display queue statistics
- **retry-failed**: Requeue failed jobs

**Usage:**

    # Add single artifact
    contentdownload queue enqueue doi:10.1234/example '{"doi":"10.1234/example"}'

    # Bulk import from file
    contentdownload queue import artifacts.jsonl

    # Start orchestrator (runs until queue empty with --drain)
    contentdownload queue run --workers 8 --drain

    # Display queue stats
    contentdownload queue stats

    # Retry all failed jobs
    contentdownload queue retry-failed
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC
from pathlib import Path

import typer

from DocsToKG.ContentDownload.orchestrator import (
    OrchestratorConfig,
    WorkQueue,
)

__all__ = [
    "app",
    "queue_enqueue",
    "queue_import",
    "queue_run",
    "queue_stats",
    "queue_retry_failed",
]

logger = logging.getLogger(__name__)

app = typer.Typer(help="Persistent work queue for artifact downloads")


@app.command()
def queue_enqueue(
    artifact_id: str = typer.Argument(..., help="Unique artifact ID (e.g., doi:10.1234/example)"),
    artifact_json: str = typer.Argument(
        "{}", help="Artifact payload as JSON string (default: empty dict)"
    ),
    queue_path: str = typer.Option(
        "state/workqueue.sqlite", "--queue", help="Path to work queue database"
    ),
) -> None:
    """Enqueue a single artifact for processing.

    Args:
        artifact_id: Unique identifier (e.g., "doi:10.1234/example")
        artifact_json: JSON payload for the artifact
        queue_path: Path to SQLite queue database
    """
    try:
        # Parse artifact JSON
        artifact = json.loads(artifact_json)

        # Create queue and enqueue
        queue = WorkQueue(queue_path, wal_mode=True)
        was_new = queue.enqueue(artifact_id, artifact)

        if was_new:
            typer.echo(f"✓ Enqueued: {artifact_id}", err=False)
        else:
            typer.echo(f"ⓘ Already queued (idempotent): {artifact_id}", err=False)

    except json.JSONDecodeError as e:
        typer.echo(f"✗ Invalid JSON: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def queue_import(
    file_path: Path = typer.Argument(..., help="JSONL file with artifacts (one per line)"),
    queue_path: str = typer.Option(
        "state/workqueue.sqlite", "--queue", help="Path to work queue database"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Max artifacts to import (default: all)"
    ),
) -> None:
    """Bulk import artifacts from JSONL file.

    Each line should be valid JSON with 'id' and optional artifact fields.

    Args:
        file_path: Path to JSONL file
        queue_path: Path to SQLite queue database
        limit: Optional limit on number of artifacts to import
    """
    if not file_path.exists():
        typer.echo(f"✗ File not found: {file_path}", err=True)
        raise typer.Exit(1)

    try:
        queue = WorkQueue(queue_path, wal_mode=True)
        enqueued = 0
        duplicate = 0
        errors = 0

        with open(file_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    artifact_id = record.get("id") or record.get("artifact_id")
                    if not artifact_id:
                        typer.echo(f"✗ Line {i + 1}: missing 'id' or 'artifact_id'", err=True)
                        errors += 1
                        continue

                    was_new = queue.enqueue(artifact_id, record)
                    if was_new:
                        enqueued += 1
                    else:
                        duplicate += 1

                except json.JSONDecodeError as e:
                    typer.echo(f"✗ Line {i + 1}: Invalid JSON: {e}", err=True)
                    errors += 1

        typer.echo(f"✓ Import complete: {enqueued} new, {duplicate} duplicate, {errors} errors")

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def queue_run(
    queue_path: str = typer.Option(
        "state/workqueue.sqlite", "--queue", help="Path to work queue database"
    ),
    workers: int = typer.Option(8, "--workers", help="Number of worker threads"),
    max_per_resolver: str | None = typer.Option(
        None, "--max-per-resolver", help="Per-resolver limits (e.g., unpaywall:2,crossref:3)"
    ),
    max_per_host: int = typer.Option(4, "--max-per-host", help="Per-host concurrency limit"),
    drain: bool = typer.Option(False, "--drain", help="Exit when queue is empty"),
) -> None:
    """Start the orchestrator and process queued artifacts.

    Args:
        queue_path: Path to SQLite queue database
        workers: Number of worker threads
        max_per_resolver: Per-resolver concurrency limits
        max_per_host: Per-host concurrency limit
        drain: Exit when queue empty
    """
    try:
        # Parse per-resolver limits
        per_resolver = {}
        if max_per_resolver:
            for pair in max_per_resolver.split(","):
                resolver, limit = pair.split(":")
                per_resolver[resolver.strip()] = int(limit.strip())

        # Create queue and config
        queue = WorkQueue(queue_path, wal_mode=True)
        _ = OrchestratorConfig(
            max_workers=workers,
            max_per_resolver=per_resolver,
            max_per_host=max_per_host,
        )

        # Create orchestrator (pipeline not wired for now)
        typer.echo(f"Starting orchestrator with {workers} workers...")

        # Note: In real integration, pipeline would be passed here
        typer.echo("⚠ Note: Pipeline integration requires full CLI bootstrap", err=True)
        typer.echo("✓ Orchestrator created (pipeline wiring needed for production)", err=False)

        if drain:
            typer.echo("Monitoring queue (drain mode - will exit when empty)")
            while True:
                stats = queue.stats()
                queued = stats.get("queued", 0)
                in_progress = stats.get("in_progress", 0)
                done = stats.get("done", 0)

                typer.echo(f"Queue: queued={queued}, in_progress={in_progress}, done={done}")

                if queued == 0 and in_progress == 0:
                    typer.echo("✓ Queue empty, exiting")
                    break

                time.sleep(2)

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def queue_stats(
    queue_path: str = typer.Option(
        "state/workqueue.sqlite", "--queue", help="Path to work queue database"
    ),
    format: str = typer.Option("table", "--format", help="Output format: table|json"),
) -> None:
    """Display work queue statistics.

    Args:
        queue_path: Path to SQLite queue database
        format: Output format (table or json)
    """
    try:
        queue = WorkQueue(queue_path, wal_mode=True)
        stats = queue.stats()

        if format == "json":
            typer.echo(json.dumps(stats, indent=2))
        else:
            typer.echo("Queue Statistics:")
            typer.echo("=" * 40)
            for key, value in stats.items():
                typer.echo(f"  {key:20} {value:10}")

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def queue_retry_failed(
    queue_path: str = typer.Option(
        "state/workqueue.sqlite", "--queue", help="Path to work queue database"
    ),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Max attempts before giving up"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be retried"),
) -> None:
    """Retry all failed jobs in the queue.

    Marks failed jobs as queued again for another attempt.

    Args:
        queue_path: Path to SQLite queue database
        max_attempts: Maximum attempts per job
        dry_run: Show what would be retried without making changes
    """
    try:
        queue = WorkQueue(queue_path, wal_mode=True)
        conn = queue._get_connection()

        try:
            # Find failed jobs
            cursor = conn.execute(
                """
                SELECT id, artifact_id, attempts
                FROM jobs
                WHERE state = 'error'
                ORDER BY updated_at DESC
                """
            )
            failed_jobs = cursor.fetchall()

            if not failed_jobs:
                typer.echo("No failed jobs to retry")
                return

            typer.echo(f"Found {len(failed_jobs)} failed jobs:")
            for job_id, artifact_id, attempts in failed_jobs:
                typer.echo(f"  - {artifact_id} (attempts: {attempts})")

            if dry_run:
                typer.echo("\n✓ Dry run complete (no changes made)")
                return

            # Retry failed jobs
            from datetime import datetime

            now_iso = datetime.now(UTC).isoformat()
            retried = 0

            for job_id, artifact_id, attempts in failed_jobs:
                if attempts >= max_attempts:
                    typer.echo(f"ⓘ Skipping {artifact_id} (max attempts reached)")
                    continue

                # Mark as queued
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = 'queued', worker_id = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, job_id),
                )
                retried += 1

            conn.commit()
            typer.echo(f"\n✓ Retried {retried} jobs")

        finally:
            conn.close()

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
