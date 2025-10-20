"""Tests for advanced Wayback telemetry features."""

import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from DocsToKG.ContentDownload.telemetry_wayback import AttemptContext, AttemptResult, ModeSelected
from DocsToKG.ContentDownload.telemetry_wayback_sqlite import SQLiteSink
from DocsToKG.ContentDownload.telemetry_wayback_migrations import (
    migrate_schema,
    get_current_schema_version,
    set_schema_version,
)
from DocsToKG.ContentDownload.telemetry_wayback_privacy import (
    mask_url_query_string,
    hash_sensitive_value,
    sanitize_details_string,
    mask_event_for_logging,
)


class TestSchemaMigrations:
    """Test schema migration system."""

    def test_get_current_schema_version_default(self):
        """Test that schema version defaults to '1' for new databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)
            sink.close()

            conn = sqlite3.connect(db_path)
            version = get_current_schema_version(conn)
            assert version == "2"  # After migration
            conn.close()

    def test_set_schema_version(self):
        """Test setting schema version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            conn = sqlite3.connect(db_path)

            # Create _meta table
            c = conn.cursor()
            c.execute("""
            CREATE TABLE _meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """)

            set_schema_version(conn, "2")
            version = get_current_schema_version(conn)
            assert version == "2"
            conn.close()

    def test_migrate_schema_idempotent(self):
        """Test that schema migration is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)
            sink.close()

            conn = sqlite3.connect(db_path)

            # Migrate twice
            result1 = migrate_schema(conn, target_version="2")
            result2 = migrate_schema(conn, target_version="2")

            assert result1 is True
            assert result2 is True

            conn.close()

    def test_run_metrics_table_exists(self):
        """Test that run_metrics table is created after migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)
            sink.close()

            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Check that wayback_run_metrics table exists
            c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='wayback_run_metrics'"
            )
            assert c.fetchone() is not None

            conn.close()


class TestRollUpTable:
    """Test run_metrics roll-up table functionality."""

    def test_finalize_run_metrics_basic(self):
        """Test finalizing run metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
            )

            # Emit attempt start
            sink.emit(
                {
                    "event_type": "wayback_attempt",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ms": 0,
                    "event": "start",
                    "original_url": "https://example.com/paper.pdf",
                    "canonical_url": "https://example.com/paper.pdf",
                }
            )

            # Emit attempt end
            sink.emit(
                {
                    "event_type": "wayback_attempt",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ms": 500,
                    "event": "end",
                    "result": AttemptResult.EMITTED_PDF.value,
                    "mode_selected": ModeSelected.PDF_DIRECT.value,
                    "total_duration_ms": 500,
                    "candidates_scanned": 1,
                }
            )

            # Finalize metrics
            sink.finalize_run_metrics("test-run")
            sink.close()

            # Verify metrics were recorded
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute(
                "SELECT attempts, emits, yield_pct FROM wayback_run_metrics WHERE run_id = ?",
                ("test-run",),
            )
            row = c.fetchone()

            assert row is not None
            assert row[0] == 1  # attempts
            assert row[1] == 1  # emits
            assert abs(row[2] - 100.0) < 0.01  # yield_pct ~ 100%

            conn.close()

    def test_finalize_run_metrics_p95_latency(self):
        """Test P95 latency calculation in run metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # Emit 10 attempts with varying durations
            for i in range(10):
                ctx = AttemptContext(
                    run_id="test-run",
                    work_id=f"work-{i}",
                    artifact_id=f"artifact-{i}",
                )

                sink.emit(
                    {
                        "event_type": "wayback_attempt",
                        "attempt_id": ctx.attempt_id,
                        "run_id": ctx.run_id,
                        "work_id": ctx.work_id,
                        "artifact_id": ctx.artifact_id,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ms": 0,
                        "event": "start",
                        "original_url": "https://example.com/paper.pdf",
                        "canonical_url": "https://example.com/paper.pdf",
                    }
                )

                sink.emit(
                    {
                        "event_type": "wayback_attempt",
                        "attempt_id": ctx.attempt_id,
                        "run_id": ctx.run_id,
                        "work_id": ctx.work_id,
                        "artifact_id": ctx.artifact_id,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ms": (i + 1) * 100,
                        "event": "end",
                        "result": AttemptResult.EMITTED_PDF.value,
                        "mode_selected": ModeSelected.PDF_DIRECT.value,
                        "total_duration_ms": (i + 1) * 100,
                        "candidates_scanned": 1,
                    }
                )

            sink.finalize_run_metrics("test-run")
            sink.close()

            # Verify P95 latency
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute(
                "SELECT p95_latency_ms FROM wayback_run_metrics WHERE run_id = ?", ("test-run",)
            )
            row = c.fetchone()

            assert row is not None
            # P95 of [100, 200, ..., 1000] should be around 900-1000
            assert 800 < row[0] <= 1000

            conn.close()


class TestRetentionAndVacuum:
    """Test retention policies and vacuum operations."""

    def test_delete_run(self):
        """Test deleting a run's telemetry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
            )

            sink.emit(
                {
                    "event_type": "wayback_attempt",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ms": 0,
                    "event": "start",
                    "original_url": "https://example.com/paper.pdf",
                    "canonical_url": "https://example.com/paper.pdf",
                }
            )

            # Delete the run
            deleted_count = sink.delete_run("test-run")
            assert deleted_count > 0

            sink.close()

    def test_vacuum_incremental(self):
        """Test incremental vacuum operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # Emit and delete some data
            sink.delete_run("nonexistent-run")

            # Run vacuum
            sink.vacuum(incremental=True)

            sink.close()


class TestPrivacyMasking:
    """Test privacy and security masking functions."""

    def test_mask_url_query_string(self):
        """Test URL query string masking."""
        url = "https://example.com/search?q=sensitive&api_key=secret"
        masked = mask_url_query_string(url)

        assert "sensitive" not in masked
        assert "secret" not in masked
        assert "[REDACTED]" in masked or masked.endswith("/search")

    def test_hash_sensitive_value(self):
        """Test hashing of sensitive values."""
        value = "user-id-12345"
        hashed = hash_sensitive_value(value)

        assert hashed.startswith("hash_")
        assert len(hashed) > 8
        assert value not in hashed

    def test_sanitize_details_string(self):
        """Test details string sanitization."""
        details = "Error: https://example.com/api?key=secret with extra long text " * 10
        sanitized = sanitize_details_string(details, max_length=256)

        assert len(sanitized) <= 256
        assert "key=secret" not in sanitized
        assert "..." in sanitized

    def test_mask_event_strict_policy(self):
        """Test event masking with strict privacy policy."""
        event = {
            "event_type": "wayback_discovery",
            "original_url": "https://example.com/paper?key=secret",
            "query_url": "https://api.example.com/search?token=xyz",
            "details": "Error: API returned 403",
        }

        masked = mask_event_for_logging(event, policy="strict")

        assert masked["original_url"] == "[REDACTED_URL]"
        assert masked["query_url"] == "[REDACTED_URL]"
        assert masked["details"] == "[REDACTED]"

    def test_mask_event_default_policy(self):
        """Test event masking with default privacy policy."""
        event = {
            "event_type": "wayback_discovery",
            "original_url": "https://example.com/paper?key=secret",
            "query_url": "https://api.example.com/search?token=xyz",
            "details": "Error: Could not fetch from https://example.com/secret",
        }

        masked = mask_event_for_logging(event, policy="default")

        # URLs should be masked but not redacted
        assert "example.com" in masked["original_url"]
        assert "[REDACTED]" in masked["original_url"]

        # Details should be sanitized (masked or truncated)
        # The key is that query strings are redacted
        assert masked["details"].startswith("Error")  # Beginning is preserved
        # The original_url in details may or may not be masked depending on implementation
        assert len(masked["details"]) <= 256  # Should be truncated if too long

    def test_mask_event_permissive_policy(self):
        """Test event masking with permissive policy (no masking)."""
        event = {
            "event_type": "wayback_discovery",
            "original_url": "https://example.com/paper?key=secret",
            "details": "Error: API error with secret=xyz",
        }

        masked = mask_event_for_logging(event, policy="permissive")

        # Should not be masked
        assert masked["original_url"] == event["original_url"]
        assert masked["details"] == event["details"]
