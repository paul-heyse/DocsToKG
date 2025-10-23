# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.verify",
#   "purpose": "Streaming verification and consistency checking for catalog records.",
#   "sections": [
#     {
#       "id": "verificationresult",
#       "name": "VerificationResult",
#       "anchor": "class-verificationresult",
#       "kind": "class"
#     },
#     {
#       "id": "streamingverifier",
#       "name": "StreamingVerifier",
#       "anchor": "class-streamingverifier",
#       "kind": "class"
#     },
#     {
#       "id": "verify-records-sync",
#       "name": "verify_records_sync",
#       "anchor": "function-verify-records-sync",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Streaming verification and consistency checking for catalog records.

Provides high-performance async verification of catalog records with:
  - Streaming SHA-256 computation (memory-efficient)
  - Concurrent verification with batch processing
  - Progress callbacks for operational visibility
  - Early exit on mismatch
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from DocsToKG.ContentDownload.catalog.store import CatalogStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a single record."""

    record_id: int
    expected_sha256: str
    computed_sha256: str | None
    matches: bool
    error: str | None = None
    elapsed_ms: int = 0


class StreamingVerifier:
    """High-performance async verification with streaming I/O."""

    def __init__(
        self,
        catalog: CatalogStore,
        max_concurrent: int = 5,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
    ):
        """Initialize verifier.

        Args:
            catalog: Catalog store to verify against
            max_concurrent: Max concurrent verifications (default 5)
            chunk_size: Streaming chunk size in bytes (default 8MB)
        """
        self.catalog = catalog
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def verify_single(
        self,
        record_id: int,
        storage_uri: str,
        expected_sha256: str,
    ) -> VerificationResult:
        """Verify a single record's SHA-256.

        Args:
            record_id: Catalog record ID
            storage_uri: file:// URI to verify
            expected_sha256: Expected SHA-256 hex

        Returns:
            VerificationResult with match status
        """
        import time

        start = time.time_ns()

        async with self._semaphore:
            try:
                # Extract file path from URI
                if not storage_uri.startswith("file://"):
                    return VerificationResult(
                        record_id=record_id,
                        expected_sha256=expected_sha256,
                        computed_sha256=None,
                        matches=False,
                        error=f"Non-file URI not supported: {storage_uri}",
                        elapsed_ms=int((time.time_ns() - start) / 1_000_000),
                    )

                file_path = storage_uri[7:]  # Remove "file://"
                path = Path(file_path)

                if not path.exists():
                    return VerificationResult(
                        record_id=record_id,
                        expected_sha256=expected_sha256,
                        computed_sha256=None,
                        matches=False,
                        error=f"File not found: {file_path}",
                        elapsed_ms=int((time.time_ns() - start) / 1_000_000),
                    )

                # Stream and compute SHA-256
                sha256_obj = hashlib.sha256()
                bytes_read = 0

                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break
                        sha256_obj.update(chunk)
                        bytes_read += len(chunk)

                computed_sha256 = sha256_obj.hexdigest()
                matches = computed_sha256.lower() == expected_sha256.lower()

                logger.debug(
                    f"Verified record {record_id}: "
                    f"{computed_sha256[:16]}... ({bytes_read / 1024 / 1024:.1f}MB) "
                    f"{'✓' if matches else '✗'}"
                )

                return VerificationResult(
                    record_id=record_id,
                    expected_sha256=expected_sha256,
                    computed_sha256=computed_sha256,
                    matches=matches,
                    elapsed_ms=int((time.time_ns() - start) / 1_000_000),
                )

            except Exception as e:
                logger.warning(f"Verification failed for record {record_id}: {e}")
                return VerificationResult(
                    record_id=record_id,
                    expected_sha256=expected_sha256,
                    computed_sha256=None,
                    matches=False,
                    error=str(e),
                    elapsed_ms=int((time.time_ns() - start) / 1_000_000),
                )

    async def verify_batch(
        self,
        records: list[tuple[int, str, str]],  # (record_id, storage_uri, expected_sha256)
        progress_callback: Callable[[int, int, VerificationResult], None] | None = None,
        fail_fast: bool = False,
    ) -> list[VerificationResult]:
        """Verify multiple records concurrently.

        Args:
            records: List of (record_id, storage_uri, expected_sha256) tuples
            progress_callback: Optional callback(current, total, result)
            fail_fast: Stop on first mismatch if True

        Returns:
            List of VerificationResult objects
        """
        results = []
        tasks = []

        for record_id, storage_uri, expected_sha256 in records:
            task = self.verify_single(record_id, storage_uri, expected_sha256)
            tasks.append(task)

        # Run with early exit on failure if fail_fast
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            if progress_callback:
                progress_callback(len(results), len(records), result)

            if fail_fast and not result.matches:
                logger.warning(f"Verification failed for record {result.record_id}, stopping")
                # Cancel remaining tasks
                for t in tasks:
                    t.cancel()
                break

        return results

    async def verify_all_records(
        self,
        progress_callback: Callable[[int, int, VerificationResult], None] | None = None,
        sample_rate: float = 1.0,
    ) -> dict[str, int]:
        """Verify all records in catalog.

        Args:
            progress_callback: Optional callback(current, total, result)
            sample_rate: Fraction of records to verify (0.0-1.0)

        Returns:
            Dict with verification statistics
        """
        import random
        import time

        start = time.time()

        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("Catalog does not support get_all_records()")
            return {"error": "get_all_records not supported"}

        # Apply sampling
        if sample_rate < 1.0:
            all_records = random.sample(all_records, max(1, int(len(all_records) * sample_rate)))

        records_to_verify = [(r.id, r.storage_uri, r.sha256 or "") for r in all_records if r.sha256]

        logger.info(
            f"Starting verification of {len(records_to_verify)} records "
            f"(sample_rate={sample_rate:.1%})"
        )

        results = await self.verify_batch(
            records_to_verify,
            progress_callback=progress_callback,
        )

        elapsed = time.time() - start

        # Aggregate statistics
        stats = {
            "total": len(results),
            "passed": sum(1 for r in results if r.matches),
            "failed": sum(1 for r in results if not r.matches),
            "errors": sum(1 for r in results if r.error),
            "elapsed_seconds": int(elapsed),
            "records_per_second": int(len(results) / elapsed) if elapsed > 0 else 0,
        }

        logger.info(
            f"Verification complete: {stats['passed']}/{stats['total']} passed "
            f"({stats['passed'] * 100 // stats['total']}%) in {elapsed:.1f}s"
        )

        if stats["failed"] > 0:
            failed = [r for r in results if not r.matches]
            logger.warning(f"Failed records: {[r.record_id for r in failed]}")

        return stats


def verify_records_sync(
    catalog: CatalogStore,
    record_ids: list[int] | None = None,
    max_concurrent: int = 5,
    progress_callback: Callable[[int, int, VerificationResult], None] | None = None,
) -> dict[str, int]:
    """Synchronous wrapper for batch verification.

    Args:
        catalog: Catalog store
        record_ids: Optional list of record IDs to verify (None = all)
        max_concurrent: Max concurrent verifications
        progress_callback: Optional progress callback

    Returns:
        Verification statistics
    """
    try:
        all_records = catalog.get_all_records()
    except NotImplementedError:
        logger.error("Catalog does not support get_all_records()")
        return {"error": "get_all_records not supported"}

    # Filter to requested records
    if record_ids:
        records = [r for r in all_records if r.id in record_ids]
    else:
        records = all_records

    records_to_verify = [(r.id, r.storage_uri, r.sha256 or "") for r in records if r.sha256]

    if not records_to_verify:
        return {"total": 0, "passed": 0, "failed": 0, "errors": 0}

    verifier = StreamingVerifier(catalog, max_concurrent=max_concurrent)

    # Run async verification
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(
            verifier.verify_batch(records_to_verify, progress_callback)
        )
    finally:
        loop.close()

    # Aggregate results
    stats = {
        "total": len(results),
        "passed": sum(1 for r in results if r.matches),
        "failed": sum(1 for r in results if not r.matches),
        "errors": sum(1 for r in results if r.error),
    }

    return stats
