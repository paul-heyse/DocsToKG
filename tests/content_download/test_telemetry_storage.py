"""
Tests for Telemetry Storage Integration

Tests storage loading from SQLite and JSONL formats.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import sqlite3
import tempfile

import pytest

from DocsToKG.ContentDownload.fallback.telemetry_storage import (
    TelemetryStorage,
    get_telemetry_storage,
)


class TestTelemetryStorageSQLite:
    """Test SQLite storage loading."""

    @pytest.fixture
    def sqlite_storage(self, tmp_path):
        """Create a temporary SQLite database."""
        db_path = tmp_path / "test_manifest.db"
        
        # Create database with sample data
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                run_id TEXT,
                tier TEXT,
                host TEXT,
                outcome TEXT,
                reason TEXT,
                elapsed_ms INTEGER,
                status INTEGER
            )
        """)
        
        # Insert sample records
        cursor.execute("""
            INSERT INTO attempts (timestamp, run_id, tier, host, outcome, reason, elapsed_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("2025-10-21T12:00:00", "run1", "tier_1", "unpaywall.org", "success", None, 850, 200))
        
        cursor.execute("""
            INSERT INTO attempts (timestamp, run_id, tier, host, outcome, reason, elapsed_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("2025-10-21T12:01:00", "run1", "tier_1", "arxiv.org", "error", "timeout", 2000, None))
        
        cursor.execute("""
            INSERT INTO attempts (timestamp, run_id, tier, host, outcome, reason, elapsed_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("2025-10-21T12:02:00", "run1", "tier_2", "doi.org", "success", None, 1200, 200))
        
        conn.commit()
        conn.close()
        
        return TelemetryStorage(str(db_path))

    def test_load_from_sqlite(self, sqlite_storage):
        """Test loading records from SQLite."""
        records = sqlite_storage.load_records(period="24h")
        assert len(records) >= 3

    def test_sqlite_tier_filtering(self, sqlite_storage):
        """Test tier filtering in SQLite."""
        records = sqlite_storage.load_records(period="24h", tier_filter="tier_1")
        assert all(r.get("tier") == "tier_1" for r in records)

    def test_sqlite_source_filtering(self, sqlite_storage):
        """Test source filtering in SQLite."""
        records = sqlite_storage.load_records(period="24h", source_filter="unpaywall.org")
        assert all(r.get("host") == "unpaywall.org" for r in records)


class TestTelemetryStorageJSONL:
    """Test JSONL storage loading."""

    @pytest.fixture
    def jsonl_storage(self, tmp_path):
        """Create a temporary JSONL file."""
        jsonl_path = tmp_path / "test_manifest.jsonl"
        
        # Write sample records
        records = [
            {
                "timestamp": "2025-10-21T12:00:00",
                "tier": "tier_1",
                "host": "unpaywall.org",
                "outcome": "success",
                "elapsed_ms": 850,
            },
            {
                "timestamp": "2025-10-21T12:01:00",
                "tier": "tier_1",
                "host": "arxiv.org",
                "outcome": "error",
                "reason": "timeout",
                "elapsed_ms": 2000,
            },
        ]
        
        with open(jsonl_path, "w") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")
        
        return TelemetryStorage(str(jsonl_path))

    def test_load_from_jsonl(self, jsonl_storage):
        """Test loading records from JSONL."""
        records = jsonl_storage.load_records(period="24h")
        assert len(records) >= 2

    def test_jsonl_tier_filtering(self, jsonl_storage):
        """Test tier filtering in JSONL."""
        records = jsonl_storage.load_records(period="24h", tier_filter="tier_1")
        assert all(r.get("tier") == "tier_1" for r in records)


class TestTelemetryStoragePeriodParsing:
    """Test period string parsing."""

    def test_parse_hours(self):
        """Test hour period parsing."""
        storage = TelemetryStorage()
        seconds = storage._parse_period("24h")
        assert seconds == 24 * 3600

    def test_parse_days(self):
        """Test day period parsing."""
        storage = TelemetryStorage()
        seconds = storage._parse_period("7d")
        assert seconds == 7 * 86400

    def test_parse_weeks(self):
        """Test week period parsing."""
        storage = TelemetryStorage()
        seconds = storage._parse_period("2w")
        assert seconds == 2 * 604800

    def test_parse_default(self):
        """Test default period parsing."""
        storage = TelemetryStorage()
        seconds = storage._parse_period("unknown")
        assert seconds == 86400  # Default to 24 hours


class TestTelemetryStorageWrite:
    """Test record writing functionality."""

    def test_write_to_sqlite(self, tmp_path):
        """Test writing records to SQLite."""
        db_path = tmp_path / "test_write.db"
        storage = TelemetryStorage(str(db_path))
        
        record = {
            "timestamp": "2025-10-21T12:00:00",
            "run_id": "run1",
            "tier": "tier_1",
            "host": "example.com",
            "outcome": "success",
            "elapsed_ms": 500,
        }
        
        result = storage.write_record(record, format="sqlite")
        assert result is True
        
        # Verify record was written
        records = storage.load_records(period="24h")
        assert len(records) > 0

    def test_write_to_jsonl(self, tmp_path):
        """Test writing records to JSONL."""
        jsonl_path = tmp_path / "test_write.jsonl"
        storage = TelemetryStorage(str(jsonl_path))
        
        record = {
            "timestamp": "2025-10-21T12:00:00",
            "tier": "tier_1",
            "outcome": "success",
        }
        
        result = storage.write_record(record, format="jsonl")
        assert result is True
        
        # Verify record was written
        with open(jsonl_path) as f:
            lines = f.readlines()
            assert len(lines) > 0


class TestTelemetryStorageSingleton:
    """Test singleton pattern."""

    def test_get_telemetry_storage(self):
        """Test getting telemetry storage singleton."""
        storage1 = get_telemetry_storage()
        storage2 = get_telemetry_storage()
        assert storage1 is storage2

    def test_get_with_different_path(self):
        """Test singleton with different paths."""
        storage1 = get_telemetry_storage("path1")
        storage2 = get_telemetry_storage("path2")
        # Different paths should create new instances
        assert storage1.storage_path != storage2.storage_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
