"""
Validated Telemetry Storage with Pydantic Integration

Extends TelemetryStorage with:
  - Pydantic model validation
  - Configuration management
  - Schema enforcement
  - Type-safe operations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from DocsToKG.ContentDownload.fallback.models import (
    AttemptStatus,
    TelemetryAttemptRecord,
    TelemetryBatchRecord,
    TelemetryConfig,
    TierName,
)
from DocsToKG.ContentDownload.fallback.telemetry_storage import TelemetryStorage

logger = logging.getLogger(__name__)


class ValidatedTelemetryStorage:
    """Type-safe telemetry storage with Pydantic validation."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[TelemetryConfig] = None,
    ):
        """Initialize with optional Pydantic configuration.

        Args:
            storage_path: Path to storage file
            config: TelemetryConfig (Pydantic model)
        """
        self.config = config or TelemetryConfig()
        self.storage = TelemetryStorage(storage_path or self.config.storage.path)
        self._batch_buffer: List[TelemetryAttemptRecord] = []

    def write_validated_record(
        self,
        run_id: str,
        attempt_id: str,
        tier: TierName,
        host: str,
        url: str,
        status: AttemptStatus,
        elapsed_ms: int,
        http_status: Optional[int] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> TelemetryAttemptRecord:
        """Write a validated telemetry record.

        All parameters are validated via Pydantic models.

        Args:
            run_id: Run identifier
            attempt_id: Attempt identifier
            tier: Fallback tier
            host: Target host
            url: URL attempted
            status: Attempt status
            elapsed_ms: Elapsed time
            http_status: HTTP status code
            reason: Failure reason
            **kwargs: Additional metadata

        Returns:
            Validated TelemetryAttemptRecord

        Raises:
            ValueError: If validation fails
        """
        try:
            # Create validated record via Pydantic
            record = TelemetryAttemptRecord(
                run_id=run_id,
                attempt_id=attempt_id,
                tier=tier,
                host=host,
                url=url,
                status=status,
                elapsed_ms=elapsed_ms,
                http_status=http_status,
                reason=reason,
                metadata=kwargs,
            )

            # Write to storage
            self.storage.write_record(record.model_dump(), format="sqlite")
            logger.info(f"Wrote validated record: {attempt_id}")

            return record
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

    def write_batch(
        self,
        batch_id: str,
        records: List[Dict[str, Any]],
    ) -> TelemetryBatchRecord:
        """Write a validated batch of records.

        Args:
            batch_id: Batch identifier
            records: List of record dictionaries

        Returns:
            Validated TelemetryBatchRecord

        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate individual records
            validated_records = []
            for record_dict in records:
                record = TelemetryAttemptRecord(**record_dict)
                validated_records.append(record)

            # Create batch
            batch = TelemetryBatchRecord(
                batch_id=batch_id,
                records=validated_records,
                count=len(validated_records),
            )

            # Write all records
            for record in validated_records:
                self.storage.write_record(record.model_dump(), format="sqlite")

            logger.info(f"Wrote batch {batch_id}: {len(validated_records)} records")
            return batch
        except ValueError as e:
            logger.error(f"Batch validation error: {e}")
            raise

    def load_records_validated(
        self,
        period: str = "24h",
        tier_filter: Optional[TierName] = None,
        source_filter: Optional[str] = None,
    ) -> List[TelemetryAttemptRecord]:
        """Load and validate records from storage.

        Args:
            period: Time period filter
            tier_filter: Tier filter
            source_filter: Source/host filter

        Returns:
            List of validated TelemetryAttemptRecord objects
        """
        # Load raw records
        raw_records = self.storage.load_records(
            period=period,
            tier_filter=tier_filter.value if tier_filter else None,
            source_filter=source_filter,
        )

        # Validate each record
        validated = []
        for raw in raw_records:
            try:
                record = TelemetryAttemptRecord(**raw)
                validated.append(record)
            except ValueError as e:
                logger.warning(f"Skipping invalid record: {e}")
                continue

        return validated

    def get_config(self) -> TelemetryConfig:
        """Get current configuration.

        Returns:
            TelemetryConfig (Pydantic model)
        """
        return self.config

    def update_config(self, config: TelemetryConfig) -> None:
        """Update configuration with validation.

        Args:
            config: New TelemetryConfig
        """
        self.config = config
        self.storage.storage_path = config.storage.path
        logger.info("Configuration updated")


__all__ = [
    "ValidatedTelemetryStorage",
]
