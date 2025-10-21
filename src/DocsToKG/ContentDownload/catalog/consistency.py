"""Deep consistency checking for catalog integrity.

Provides comprehensive validation:
  - Orphan detection (files without catalog entries)
  - Missing file detection (catalog entries without files)
  - Hash mismatch detection (recomputed vs stored)
  - Referential integrity checks
  - Comprehensive audit reports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from DocsToKG.ContentDownload.catalog.gc import collect_referenced_paths, find_orphans
from DocsToKG.ContentDownload.catalog.store import CatalogStore
from DocsToKG.ContentDownload.catalog.verify import StreamingVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrphanFile:
    """An orphaned file in storage."""
    path: str
    size_bytes: int
    reason: str = "Not referenced in catalog"


@dataclass(frozen=True)
class MissingFile:
    """A catalog entry pointing to a missing file."""
    record_id: int
    artifact_id: str
    storage_uri: str
    reason: str = "File not found"


@dataclass(frozen=True)
class HashMismatch:
    """A hash mismatch between stored and recomputed."""
    record_id: int
    artifact_id: str
    expected_hash: str
    computed_hash: Optional[str]
    reason: str = "Hash mismatch"


@dataclass(frozen=True)
class AuditIssue:
    """Generic audit issue."""
    issue_type: str
    severity: str  # "warning", "error", "critical"
    details: str


@dataclass(frozen=True)
class ConsistencyAuditReport:
    """Complete consistency audit report."""
    orphan_files: list[OrphanFile]
    missing_files: list[MissingFile]
    hash_mismatches: list[HashMismatch]
    referential_issues: list[AuditIssue]
    total_issues: int
    critical_issues: int
    warnings: int


class ConsistencyChecker:
    """Deep consistency validation for catalog."""
    
    def __init__(
        self,
        catalog: CatalogStore,
        root_dir: str,
        verifier: Optional[StreamingVerifier] = None,
    ):
        """Initialize consistency checker.
        
        Args:
            catalog: Catalog store
            root_dir: Root storage directory
            verifier: Optional StreamingVerifier (creates if not provided)
        """
        self.catalog = catalog
        self.root_dir = Path(root_dir)
        self.verifier = verifier or StreamingVerifier(catalog)
    
    def check_orphans(self) -> list[OrphanFile]:
        """Find orphaned files (in storage but not in catalog).
        
        Returns:
            List of orphaned files
        """
        logger.info("Checking for orphaned files...")
        
        try:
            referenced = collect_referenced_paths(self.catalog)
        except Exception as e:
            logger.error(f"Failed to collect referenced paths: {e}")
            return []
        
        orphans_paths = find_orphans(str(self.root_dir), referenced)
        
        orphan_files = []
        for path in orphans_paths:
            try:
                size = Path(path).stat().st_size
                orphan_files.append(OrphanFile(path=path, size_bytes=size))
            except Exception as e:
                logger.warning(f"Could not stat orphan {path}: {e}")
        
        logger.info(f"Found {len(orphan_files)} orphaned files")
        return orphan_files
    
    def check_missing_files(self) -> list[MissingFile]:
        """Find missing files (catalog entries pointing to non-existent files).
        
        Returns:
            List of missing file entries
        """
        logger.info("Checking for missing files...")
        
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("get_all_records not supported")
            return []
        
        missing = []
        for record in all_records:
            if record.storage_uri.startswith("file://"):
                file_path = record.storage_uri[7:]  # Remove "file://"
                if not Path(file_path).exists():
                    missing.append(
                        MissingFile(
                            record_id=record.id,
                            artifact_id=record.artifact_id,
                            storage_uri=record.storage_uri,
                        )
                    )
        
        logger.info(f"Found {len(missing)} missing files")
        return missing
    
    def check_hash_mismatches(
        self,
        sample_rate: float = 0.1,
    ) -> list[HashMismatch]:
        """Find hash mismatches by sampling and recomputing.
        
        Args:
            sample_rate: Fraction of records to verify (0.0-1.0)
            
        Returns:
            List of hash mismatches
        """
        logger.info(f"Checking hash mismatches (sample_rate={sample_rate:.1%})...")
        
        import asyncio
        import random
        
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("get_all_records not supported")
            return []
        
        # Filter to records with hashes
        hashed_records = [r for r in all_records if r.sha256]
        
        # Sample
        if sample_rate < 1.0 and hashed_records:
            hashed_records = random.sample(
                hashed_records,
                max(1, int(len(hashed_records) * sample_rate))
            )
        
        # Verify hashes
        records_to_verify = [
            (r.id, r.storage_uri, r.sha256)
            for r in hashed_records
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self.verifier.verify_batch(records_to_verify)
            )
        finally:
            loop.close()
        
        # Collect mismatches
        mismatches = []
        for result in results:
            if not result.matches:
                mismatches.append(
                    HashMismatch(
                        record_id=result.record_id,
                        artifact_id="",  # Would need lookup
                        expected_hash=result.expected_sha256,
                        computed_hash=result.computed_sha256,
                    )
                )
        
        logger.info(f"Found {len(mismatches)} hash mismatches")
        return mismatches
    
    def check_referential_integrity(self) -> list[AuditIssue]:
        """Check referential integrity and other constraints.
        
        Returns:
            List of integrity issues
        """
        logger.info("Checking referential integrity...")
        
        issues = []
        
        try:
            all_records = self.catalog.get_all_records()
        except NotImplementedError:
            logger.error("get_all_records not supported")
            return issues
        
        if not all_records:
            return issues
        
        # Check for duplicate artifact_id + resolver combinations
        seen = set()
        duplicates = []
        for record in all_records:
            key = (record.artifact_id, record.resolver)
            if key in seen:
                duplicates.append(key)
            else:
                seen.add(key)
        
        for artifact_id, resolver in duplicates:
            issues.append(
                AuditIssue(
                    issue_type="duplicate_artifact",
                    severity="warning",
                    details=f"Multiple records for {artifact_id} from {resolver}",
                )
            )
        
        # Check for empty storage URIs
        for record in all_records:
            if not record.storage_uri:
                issues.append(
                    AuditIssue(
                        issue_type="empty_storage_uri",
                        severity="error",
                        details=f"Record {record.id} ({record.artifact_id}) has no storage URI",
                    )
                )
        
        logger.info(f"Found {len(issues)} referential integrity issues")
        return issues
    
    def run_full_audit(
        self,
        sample_rate: float = 0.1,
    ) -> ConsistencyAuditReport:
        """Run comprehensive consistency audit.
        
        Args:
            sample_rate: Fraction of records to verify for hashes
            
        Returns:
            ConsistencyAuditReport with all findings
        """
        logger.info("Starting full consistency audit...")
        
        # Run all checks
        orphans = self.check_orphans()
        missing = self.check_missing_files()
        mismatches = self.check_hash_mismatches(sample_rate=sample_rate)
        referential = self.check_referential_integrity()
        
        # Count issues by severity
        critical_count = len(missing) + len(mismatches)  # These are critical
        warning_count = len(orphans) + len(referential)
        
        total_issues = len(orphans) + len(missing) + len(mismatches) + len(referential)
        
        report = ConsistencyAuditReport(
            orphan_files=orphans,
            missing_files=missing,
            hash_mismatches=mismatches,
            referential_issues=referential,
            total_issues=total_issues,
            critical_issues=critical_count,
            warnings=warning_count,
        )
        
        logger.info(
            f"Audit complete: {total_issues} total issues "
            f"({critical_count} critical, {warning_count} warnings)"
        )
        
        return report
    
    def print_audit_report(self, report: ConsistencyAuditReport) -> None:
        """Pretty-print audit report.
        
        Args:
            report: Audit report to print
        """
        print("\n" + "=" * 80)
        print("CATALOG CONSISTENCY AUDIT REPORT")
        print("=" * 80)
        
        print(f"\nSummary:")
        print(f"  Total Issues:    {report.total_issues}")
        print(f"  Critical Issues: {report.critical_issues}")
        print(f"  Warnings:        {report.warnings}")
        
        if report.orphan_files:
            print(f"\nOrphaned Files ({len(report.orphan_files)}):")
            for orphan in report.orphan_files[:10]:
                print(f"  - {orphan.path} ({orphan.size_bytes / 1024 / 1024:.1f}MB)")
            if len(report.orphan_files) > 10:
                print(f"  ... and {len(report.orphan_files) - 10} more")
        
        if report.missing_files:
            print(f"\nMissing Files ({len(report.missing_files)}):")
            for missing in report.missing_files[:10]:
                print(f"  - Record {missing.record_id}: {missing.artifact_id}")
            if len(report.missing_files) > 10:
                print(f"  ... and {len(report.missing_files) - 10} more")
        
        if report.hash_mismatches:
            print(f"\nHash Mismatches ({len(report.hash_mismatches)}):")
            for mismatch in report.hash_mismatches[:10]:
                print(f"  - Record {mismatch.record_id}: Expected {mismatch.expected_hash[:16]}...")
            if len(report.hash_mismatches) > 10:
                print(f"  ... and {len(report.hash_mismatches) - 10} more")
        
        if report.referential_issues:
            print(f"\nReferential Issues ({len(report.referential_issues)}):")
            for issue in report.referential_issues[:10]:
                print(f"  - [{issue.severity.upper()}] {issue.issue_type}: {issue.details}")
            if len(report.referential_issues) > 10:
                print(f"  ... and {len(report.referential_issues) - 10} more")
        
        print("\n" + "=" * 80 + "\n")
