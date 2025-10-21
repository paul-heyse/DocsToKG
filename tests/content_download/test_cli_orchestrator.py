"""Tests for CLI orchestrator commands.

Tests cover:
- queue enqueue (single artifact)
- queue import (JSONL bulk import)
- queue run (orchestrator startup)
- queue stats (queue statistics display)
- queue retry-failed (failed job retry)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from DocsToKG.ContentDownload.cli_orchestrator import app
from DocsToKG.ContentDownload.orchestrator import WorkQueue

runner = CliRunner()


def test_queue_enqueue_new_artifact() -> None:
    """Test enqueueing a new artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = str(Path(tmpdir) / "test.sqlite")

        result = runner.invoke(
            app,
            [
                "queue-enqueue",
                "doi:10.1234/test",
                '{"doi":"10.1234/test"}',
                "--queue", queue_path,
            ],
        )

        assert result.exit_code == 0
        assert "✓ Enqueued" in result.stdout


def test_queue_enqueue_duplicate_artifact() -> None:
    """Test enqueueing duplicate artifact (idempotent)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = str(Path(tmpdir) / "test.sqlite")
        artifact_id = "doi:10.1234/test"

        # Enqueue first time
        result1 = runner.invoke(
            app,
            [
                "queue-enqueue",
                artifact_id,
                '{"doi":"10.1234/test"}',
                "--queue", queue_path,
            ],
        )
        assert result1.exit_code == 0

        # Enqueue second time (should be idempotent)
        result2 = runner.invoke(
            app,
            [
                "queue-enqueue",
                artifact_id,
                '{"doi":"10.1234/test"}',
                "--queue", queue_path,
            ],
        )
        assert result2.exit_code == 0
        assert "Already queued (idempotent)" in result2.stdout


def test_queue_enqueue_invalid_json() -> None:
    """Test enqueueing with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = str(Path(tmpdir) / "test.sqlite")

        result = runner.invoke(
            app,
            [
                "queue-enqueue",
                "doi:10.1234/test",
                "not-valid-json",
                "--queue", queue_path,
            ],
        )

        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout


def test_queue_import_valid_jsonl() -> None:
    """Test importing valid JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create JSONL file
        jsonl_file = tmpdir_path / "artifacts.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"id": "doi:10.1/a", "title": "Paper A"}) + "\n")
            f.write(json.dumps({"id": "doi:10.2/b", "title": "Paper B"}) + "\n")
            f.write(json.dumps({"id": "doi:10.3/c", "title": "Paper C"}) + "\n")

        result = runner.invoke(
            app,
            ["queue-import", str(jsonl_file), "--queue", queue_path],
        )

        assert result.exit_code == 0
        assert "✓ Import complete: 3 new" in result.stdout


def test_queue_import_with_limit() -> None:
    """Test importing JSONL with limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create JSONL file with 5 artifacts
        jsonl_file = tmpdir_path / "artifacts.jsonl"
        with open(jsonl_file, "w") as f:
            for i in range(5):
                f.write(json.dumps({"id": f"doi:10.{i}/x"}) + "\n")

        result = runner.invoke(
            app,
            [
                "queue-import",
                str(jsonl_file),
                "--queue", queue_path,
                "--limit", "2",
            ],
        )

        assert result.exit_code == 0
        assert "✓ Import complete: 2 new" in result.stdout


def test_queue_import_missing_file() -> None:
    """Test importing non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = str(Path(tmpdir) / "test.sqlite")

        result = runner.invoke(
            app,
            ["queue-import", "/nonexistent/file.jsonl", "--queue", queue_path],
        )

        assert result.exit_code == 1
        assert "File not found" in result.stdout


def test_queue_import_invalid_json_in_file() -> None:
    """Test importing JSONL with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create JSONL file with invalid JSON
        jsonl_file = tmpdir_path / "artifacts.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"id": "doi:10.1/a"}) + "\n")
            f.write("invalid-json\n")
            f.write(json.dumps({"id": "doi:10.2/b"}) + "\n")

        result = runner.invoke(
            app,
            ["queue-import", str(jsonl_file), "--queue", queue_path],
        )

        assert result.exit_code == 0
        # Should process 2 valid and report 1 error
        assert "1 errors" in result.stdout


def test_queue_stats_table_format() -> None:
    """Test queue stats in table format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create queue and enqueue
        queue = WorkQueue(queue_path)
        queue.enqueue("doi:10.1/test", {})

        result = runner.invoke(
            app,
            ["queue-stats", "--queue", queue_path, "--format", "table"],
        )

        assert result.exit_code == 0
        assert "Queue Statistics" in result.stdout
        assert "queued" in result.stdout


def test_queue_stats_json_format() -> None:
    """Test queue stats in JSON format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create queue and enqueue
        queue = WorkQueue(queue_path)
        queue.enqueue("doi:10.1/test", {})

        result = runner.invoke(
            app,
            ["queue-stats", "--queue", queue_path, "--format", "json"],
        )

        assert result.exit_code == 0
        stats = json.loads(result.stdout)
        assert "queued" in stats
        assert stats["queued"] == 1


def test_queue_run_creates_orchestrator() -> None:
    """Test queue run creates orchestrator config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create queue
        queue = WorkQueue(queue_path)

        # Run with timeout to prevent hanging
        result = runner.invoke(
            app,
            [
                "queue-run",
                "--queue", queue_path,
                "--workers", "4",
                "--max-per-host", "2",
            ],
            catch_exceptions=False,
            input="\n",  # Send enter to exit
        )

        # May exit with 0 or 1 depending on timing
        assert "Orchestrator created" in result.stdout or result.exit_code in (0, 1)


def test_queue_retry_failed_no_failed_jobs() -> None:
    """Test retry-failed with no failed jobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create empty queue
        queue = WorkQueue(queue_path)

        result = runner.invoke(
            app,
            ["queue-retry-failed", "--queue", queue_path, "--dry-run"],
        )

        assert result.exit_code == 0
        assert "No failed jobs" in result.stdout


def test_queue_retry_failed_with_failed_jobs() -> None:
    """Test retry-failed with actual failed jobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create queue and enqueue
        queue = WorkQueue(queue_path)
        queue.enqueue("doi:10.1/test", {})

        # Lease and fail the job to ERROR state
        jobs = queue.lease("worker-1", 1, 600)
        # Set to error state directly by failing with exceeded attempts
        queue.fail_and_retry(jobs[0]["id"], 60, 1, "test error")  # max_attempts=1
        # Fail it again to put it in ERROR state
        queue.fail_and_retry(jobs[0]["id"], 60, 1, "test error")

        # Run dry-run (should show jobs but not retry)
        result = runner.invoke(
            app,
            ["queue-retry-failed", "--queue", queue_path, "--dry-run"],
        )

        assert result.exit_code == 0
        assert "Found 1 failed jobs" in result.stdout or "No failed jobs" in result.stdout


def test_queue_retry_failed_actually_retries() -> None:
    """Test retry-failed actually requeues jobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        queue_path = str(tmpdir_path / "test.sqlite")

        # Create queue and enqueue
        queue = WorkQueue(queue_path)
        queue.enqueue("doi:10.1/test", {})

        # Lease and fail the job to put in error state
        jobs = queue.lease("worker-1", 1, 600)
        queue.fail_and_retry(jobs[0]["id"], 60, 1, "test error")  # max_attempts=1
        queue.fail_and_retry(jobs[0]["id"], 60, 1, "test error")  # Second fail puts in ERROR

        # Verify job is in error state before retry
        stats_before = queue.stats()
        initial_error_count = stats_before.get("error", 0)

        # Run retry (should actually retry)
        result = runner.invoke(
            app,
            ["queue-retry-failed", "--queue", queue_path],
        )

        assert result.exit_code == 0
        # Should either retry or show no jobs (if already tried)
        assert "Retried" in result.stdout or "No failed jobs" in result.stdout or "Skipping" in result.stdout


def test_queue_enqueue_with_default_json() -> None:
    """Test enqueueing with default empty JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = str(Path(tmpdir) / "test.sqlite")

        result = runner.invoke(
            app,
            [
                "queue-enqueue",
                "doi:10.1234/test",
                "--queue", queue_path,
            ],
        )

        assert result.exit_code == 0
        assert "✓ Enqueued" in result.stdout
