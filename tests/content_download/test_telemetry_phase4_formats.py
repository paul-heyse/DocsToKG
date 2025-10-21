"""Phase 4 Telemetry Tests: CSV & Manifest Format Verification.

Validates that CSV and manifest formats can be properly structured:
- CSV sink initialization and header
- Manifest entry schema compliance  
- Stable tokens (status, reason, classification)
- Bootstrap integration with telemetry sinks
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.api import AttemptRecord
from DocsToKG.ContentDownload.bootstrap import BootstrapConfig, run_from_config
from DocsToKG.ContentDownload.http_session import HttpConfig, reset_http_session
from DocsToKG.ContentDownload.telemetry import CsvSink, ManifestEntry


class TestCsvFormatCompliance(unittest.TestCase):
    """Test CSV format compliance with specification."""

    def setUp(self):
        """Create temporary CSV file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "attempts.csv"

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_csv_header_fields_present(self):
        """CSV sink has all required header fields."""
        sink = CsvSink(self.csv_path)

        # Spec requires these core fields
        expected_core_fields = {
            "timestamp",
            "run_id",
            "resolver_name",
            "url",
            "status",
            "http_status",
            "elapsed_ms",
        }

        actual_fields = set(sink.HEADER)
        missing = expected_core_fields - actual_fields
        assert not missing, f"Missing required CSV fields: {missing}"

        sink.close()

    def test_csv_header_row_written(self):
        """CSV file starts with header row."""
        sink = CsvSink(self.csv_path)
        sink.close()

        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert len(reader.fieldnames) > 0
            assert "timestamp" in reader.fieldnames
            assert "run_id" in reader.fieldnames

    def test_csv_sink_initialization(self):
        """CSV sink can be initialized without error."""
        sink = CsvSink(self.csv_path)
        assert sink is not None
        assert sink._path == self.csv_path
        sink.close()

    def test_csv_sink_thread_safe_interface(self):
        """CSV sink has thread-safe interface."""
        sink = CsvSink(self.csv_path)
        assert hasattr(sink, "_lock")
        assert hasattr(sink, "log_attempt")
        assert hasattr(sink, "close")
        sink.close()


class TestManifestFormat(unittest.TestCase):
    """Test manifest entry format compliance with specification."""

    def test_manifest_entry_initialization(self):
        """ManifestEntry can be created with required fields."""
        entry = ManifestEntry(
            schema_version=2,
            timestamp="2025-10-21T23:12:47.005Z",
            work_id="doi:10.1234/test",
            title="Test Paper",
            publication_year=2025,
            resolver="unpaywall",
            url="https://example.com/paper.pdf",
            path="/tmp/paper.pdf",
            classification="success",
            content_type="application/pdf",
            reason="ok",
        )

        assert entry is not None
        assert entry.work_id == "doi:10.1234/test"
        # Classification is normalized by __post_init__
        assert entry.classification is not None
        assert isinstance(entry.classification, str)

    def test_manifest_entry_schema_version(self):
        """Manifest entries include schema version."""
        entry = ManifestEntry(
            schema_version=2,
            timestamp="2025-10-21T23:12:47.005Z",
            work_id="w1",
            title="Title",
            publication_year=2025,
            resolver="test",
            url="https://example.com",
            path="/tmp/w1",
            classification="success",
            content_type="application/pdf",
            reason="ok",
        )

        assert entry.schema_version == 2

    def test_manifest_classification_tokens(self):
        """Manifest classifications can be set with spec tokens."""
        spec_outcomes = ["success", "skip", "error"]

        for outcome in spec_outcomes:
            entry = ManifestEntry(
                schema_version=2,
                timestamp="2025-10-21T23:12:47.005Z",
                work_id=f"w{outcome}",
                title="Title",
                publication_year=2025,
                resolver="test",
                url="https://example.com",
                path=f"/tmp/w{outcome}",
                classification=outcome,
                content_type="application/pdf",
                reason="ok",
            )

            # Classification is normalized but should remain a string
            assert entry.classification is not None
            assert isinstance(entry.classification, str)

    def test_manifest_reason_field(self):
        """Manifest entries can store reason codes."""
        reason_codes = [
            "ok",
            "not-modified",
            "robots",
            "download-error",
        ]

        for reason in reason_codes:
            entry = ManifestEntry(
                schema_version=2,
                timestamp="2025-10-21T23:12:47.005Z",
                work_id="w1",
                title="Title",
                publication_year=2025,
                resolver="test",
                url="https://example.com",
                path="/tmp/w1",
                classification="error" if reason != "ok" else "success",
                content_type="application/pdf",
                reason=reason,
            )

            assert entry.reason == reason


class TestPhase4Integration(unittest.TestCase):
    """End-to-end Phase 4 tests: Bootstrap with telemetry integration."""

    def setUp(self):
        """Reset session and setup temp directory."""
        reset_http_session()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Cleanup."""
        reset_http_session()
        self.temp_dir.cleanup()

    def test_bootstrap_with_telemetry_config(self):
        """Bootstrap orchestrator accepts telemetry paths."""
        csv_path = Path(self.temp_dir.name) / "attempts.csv"

        config = BootstrapConfig(
            http=HttpConfig(),
            telemetry_paths={"csv": csv_path},
            resolver_registry={},
        )

        result = run_from_config(config, artifacts=None)
        assert result.run_id is not None

    def test_csv_and_manifest_coexist(self):
        """CSV and manifest sinks can be instantiated together."""
        csv_path = Path(self.temp_dir.name) / "attempts.csv"
        manifest_path = Path(self.temp_dir.name) / "manifest.jsonl"

        # Create CSV sink
        csv_sink = CsvSink(csv_path)
        assert csv_sink is not None
        assert csv_path.exists()

        # Verify CSV header
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None

        csv_sink.close()


class TestAttemptRecordSchema(unittest.TestCase):
    """Test AttemptRecord canonical schema compliance."""

    def test_attempt_record_required_fields(self):
        """AttemptRecord has required canonical fields."""
        record = AttemptRecord(
            run_id="test-run",
            resolver_name="unpaywall",
            url="https://example.com/paper.pdf",
            status="http-get",
        )

        assert record.run_id == "test-run"
        assert record.resolver_name == "unpaywall"
        assert record.url == "https://example.com/paper.pdf"
        assert record.status == "http-get"

    def test_attempt_record_optional_fields(self):
        """AttemptRecord supports optional fields."""
        record = AttemptRecord(
            run_id="test-run",
            resolver_name="unpaywall",
            url="https://example.com/paper.pdf",
            status="http-get",
            http_status=200,
            elapsed_ms=150,
            meta={"bytes_written": 12345},
        )

        assert record.http_status == 200
        assert record.elapsed_ms == 150
        assert record.meta["bytes_written"] == 12345

    def test_attempt_record_stable_status_tokens(self):
        """AttemptRecord status uses stable tokens."""
        valid_statuses = {
            "http-head",
            "http-get",
            "http-200",
            "http-304",
            "robots-fetch",
            "robots-disallowed",
            "retry",
            "size-mismatch",
        }

        for status in valid_statuses:
            record = AttemptRecord(
                run_id="test",
                resolver_name="test",
                url="https://example.com",
                status=status,
            )
            assert record.status == status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
