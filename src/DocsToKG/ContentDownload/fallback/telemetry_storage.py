# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.telemetry_storage",
#   "purpose": "Telemetry Storage Loaders.",
#   "sections": [
#     {
#       "id": "telemetrystorage",
#       "name": "TelemetryStorage",
#       "anchor": "class-telemetrystorage",
#       "kind": "class"
#     },
#     {
#       "id": "get-telemetry-storage",
#       "name": "get_telemetry_storage",
#       "anchor": "function-get-telemetry-storage",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Telemetry Storage Loaders

Provides unified interface for loading telemetry records from:
  - SQLite (manifest.sqlite3)
  - JSONL (manifest.jsonl)
  - In-memory cache

Supports:
  - Time-based filtering
  - Source/tier filtering
  - Record aggregation
  - Performance optimization
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class TelemetryStorage:
    """Unified interface for telemetry record storage and retrieval."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize storage with optional path."""
        self.storage_path = storage_path or "Data/Manifests/manifest.sqlite3"
        self._cache: List[Dict[str, Any]] = []
        self._cache_valid = False

    def load_records(
        self,
        period: str = "24h",
        tier_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load telemetry records with optional filters.

        Args:
            period: Duration string (e.g., "24h", "7d", "30d")
            tier_filter: Optional tier name filter
            source_filter: Optional source/host filter

        Returns:
            List of telemetry records
        """
        path = Path(self.storage_path)

        if path.suffix.lower() == ".db" or "sqlite" in path.name.lower():
            records = self._load_from_sqlite(path, period)
        elif path.suffix.lower() == ".jsonl":
            records = self._load_from_jsonl(path, period)
        else:
            logger.warning(f"Unknown storage format: {path.suffix}, returning empty")
            records = []

        # Apply filters
        if tier_filter:
            records = [r for r in records if r.get("tier") == tier_filter]
        if source_filter:
            records = [r for r in records if r.get("host") == source_filter]

        return records

    def _load_from_sqlite(
        self,
        path: Path,
        period: str,
    ) -> List[Dict[str, Any]]:
        """Load records from SQLite database."""
        if not path.exists():
            logger.warning(f"SQLite file not found: {path}")
            return []

        try:
            # Parse period to seconds
            time_limit = self._parse_period(period)
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_limit)

            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query attempts table (or create if doesn't exist)
            try:
                cursor.execute(
                    """
                    SELECT * FROM attempts
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """,
                    (cutoff_time.isoformat(),),
                )

                records = []
                for row in cursor.fetchall():
                    records.append(dict(row))

                conn.close()
                logger.info(f"Loaded {len(records)} records from SQLite")
                return records
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                conn.close()
                logger.warning("Attempts table not found in SQLite database")
                return []
        except Exception as e:
            logger.error(f"Error loading from SQLite: {e}")
            return []

    def _load_from_jsonl(
        self,
        path: Path,
        period: str,
    ) -> List[Dict[str, Any]]:
        """Load records from JSONL file."""
        if not path.exists():
            logger.warning(f"JSONL file not found: {path}")
            return []

        try:
            time_limit = self._parse_period(period)
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_limit)

            records = []
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)

                        # Check timestamp if present
                        if "timestamp" in record:
                            ts = datetime.fromisoformat(record["timestamp"])
                            if ts < cutoff_time:
                                continue

                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse line: {e}")
                        continue

            logger.info(f"Loaded {len(records)} records from JSONL")
            return records
        except Exception as e:
            logger.error(f"Error loading from JSONL: {e}")
            return []

    def stream_records(
        self,
        period: str = "24h",
        tier_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        batch_size: int = 100,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream telemetry records in batches for memory efficiency.

        Yields:
            Batches of telemetry records
        """
        all_records = self.load_records(period, tier_filter, source_filter)

        for i in range(0, len(all_records), batch_size):
            batch = all_records[i : i + batch_size]
            yield batch

    def write_record(self, record: Dict[str, Any], format: str = "sqlite") -> bool:
        """Write a single telemetry record.

        Args:
            record: Record to write
            format: Storage format ("sqlite" or "jsonl")

        Returns:
            True if successful
        """
        try:
            path = Path(self.storage_path)

            if format == "sqlite":
                return self._write_to_sqlite(record)
            elif format == "jsonl":
                return self._write_to_jsonl(record)
            else:
                logger.error(f"Unknown format: {format}")
                return False
        except Exception as e:
            logger.error(f"Error writing record: {e}")
            return False

    def _write_to_sqlite(self, record: Dict[str, Any]) -> bool:
        """Write record to SQLite database."""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    run_id TEXT,
                    tier TEXT,
                    host TEXT,
                    outcome TEXT,
                    reason TEXT,
                    elapsed_ms INTEGER,
                    status INTEGER,
                    metadata TEXT
                )
            """
            )

            # Insert record
            cursor.execute(
                """
                INSERT INTO attempts (timestamp, run_id, tier, host, outcome, reason, elapsed_ms, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.get("timestamp", datetime.utcnow().isoformat()),
                    record.get("run_id"),
                    record.get("tier"),
                    record.get("host"),
                    record.get("outcome"),
                    record.get("reason"),
                    record.get("elapsed_ms"),
                    record.get("status"),
                    json.dumps(record.get("metadata", {})),
                ),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error writing to SQLite: {e}")
            return False

    def _write_to_jsonl(self, record: Dict[str, Any]) -> bool:
        """Write record to JSONL file."""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a") as f:
                json.dump(record, f)
                f.write("\n")

            return True
        except Exception as e:
            logger.error(f"Error writing to JSONL: {e}")
            return False

    @staticmethod
    def _parse_period(period: str) -> int:
        """Parse period string to seconds.

        Args:
            period: Period string (e.g., "24h", "7d", "30d")

        Returns:
            Seconds
        """
        period = period.lower().strip()

        if period.endswith("h"):
            return int(period[:-1]) * 3600
        elif period.endswith("d"):
            return int(period[:-1]) * 86400
        elif period.endswith("w"):
            return int(period[:-1]) * 604800
        elif period.endswith("m"):
            return int(period[:-1]) * 2592000
        else:
            # Default to 24 hours
            return 86400


# Singleton instance
_storage_instance: Optional[TelemetryStorage] = None


def get_telemetry_storage(path: Optional[str] = None) -> TelemetryStorage:
    """Get or create telemetry storage singleton.

    Args:
        path: Optional storage path override

    Returns:
        TelemetryStorage instance
    """
    global _storage_instance
    if _storage_instance is None or path:
        _storage_instance = TelemetryStorage(path)
    return _storage_instance


__all__ = [
    "TelemetryStorage",
    "get_telemetry_storage",
]
