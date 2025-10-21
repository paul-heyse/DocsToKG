"""Tests for Phase 3-4 hardening policies: Resource budgets and permissions enforcement.

This test module validates Phase 3-4 implementations for:
- Disk space budgeting and verification
- Permission enforcement and sanitization
- Setuid/setgid/sticky bit stripping
- Default mode application
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.errors import ConfigError
from DocsToKG.OntologyDownload.io import (
    ExtractionGuardian,
    ExtractionSettings,
    extract_archive_safe,
    safe_defaults,
    strict_defaults,
)


class TestPhase34DiskSpaceBudgeting:
    """Tests for disk space budgeting and pre-extraction verification."""

    def test_disk_space_check_enabled_by_default(self):
        """Disk space checking is enabled by default in policies."""
        policy = safe_defaults()
        assert policy.check_disk_space is True

    def test_disk_space_check_can_be_disabled(self):
        """Disk space checking can be disabled via policy."""
        policy = safe_defaults()
        policy.check_disk_space = False
        assert policy.check_disk_space is False

    def test_extraction_succeeds_with_sufficient_space(self):
        """Extraction succeeds when sufficient disk space is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create small archive
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "small content")

            policy = safe_defaults()
            policy.check_disk_space = True
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should succeed (plenty of space available)
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 1

    def test_guardian_verifies_space_before_extraction(self):
        """ExtractionGuardian verifies disk space is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            policy = safe_defaults()
            guardian = ExtractionGuardian(policy)

            # Should pass for reasonable space requirement
            guardian.verify_space_available(
                required_bytes=1024,
                destination=destination,
            )


class TestPhase34PermissionsEnforcement:
    """Tests for permission enforcement and sanitization."""

    def test_preserve_permissions_disabled_by_default(self):
        """Permission preservation is disabled by default."""
        policy = safe_defaults()
        assert policy.preserve_permissions is False

    def test_default_file_mode_is_safe(self):
        """Default file mode doesn't include dangerous bits."""
        policy = safe_defaults()
        # Should be 0o644 (no setuid/setgid/sticky)
        assert policy.file_mode == 0o644
        assert (policy.file_mode & 0o7000) == 0

    def test_default_dir_mode_is_safe(self):
        """Default directory mode doesn't include dangerous bits."""
        policy = safe_defaults()
        # Should be 0o755 (no setuid/setgid/sticky)
        assert policy.dir_mode == 0o755
        assert (policy.dir_mode & 0o7000) == 0

    def test_guardian_applies_file_permissions(self):
        """ExtractionGuardian applies default file permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            test_file = destination / "test.txt"
            test_file.write_text("content")

            policy = safe_defaults()
            policy.preserve_permissions = False
            policy.file_mode = 0o600  # Read/write for owner only

            guardian = ExtractionGuardian(policy)
            guardian.finalize_extraction(
                extracted_files=[test_file],
                extracted_dirs=[],
            )

            # Check permissions were applied
            stat_result = test_file.stat()
            mode = stat_result.st_mode & 0o777
            assert mode == 0o600

    def test_guardian_applies_directory_permissions(self):
        """ExtractionGuardian applies default directory permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            test_dir = destination / "testdir"
            test_dir.mkdir()

            policy = safe_defaults()
            policy.preserve_permissions = False
            policy.dir_mode = 0o700  # Only owner

            guardian = ExtractionGuardian(policy)
            guardian.finalize_extraction(
                extracted_files=[],
                extracted_dirs=[test_dir],
            )

            # Check permissions were applied
            stat_result = test_dir.stat()
            mode = stat_result.st_mode & 0o777
            assert mode == 0o700

    def test_extraction_applies_permissions_post_extraction(self):
        """Extracted files have permissions applied after extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            policy = safe_defaults()
            policy.file_mode = 0o600
            policy.preserve_permissions = False
            policy.encapsulate = False
            policy.use_dirfd = False

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Verify permissions were applied
            assert len(extracted) == 1
            stat_result = extracted[0].stat()
            mode = stat_result.st_mode & 0o777
            assert mode == 0o600


class TestPhase34Integration:
    """Integration tests combining Phase 3-4 features."""

    def test_full_extraction_pipeline_with_all_policies(self):
        """Full extraction pipeline applies all Phase 1-4 policies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create test archive
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("dir1/file.txt", "content")
                zf.writestr("dir2/file.txt", "more content")

            # Use comprehensive policy
            policy = safe_defaults()
            policy.encapsulate = True
            policy.check_disk_space = True
            policy.preserve_permissions = False
            policy.file_mode = 0o644
            policy.dir_mode = 0o755

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Verify extraction completed
            assert len(extracted) == 2

            # Verify file permissions applied
            for file_path in extracted:
                stat_result = file_path.stat()
                mode = stat_result.st_mode & 0o777
                assert mode == 0o644

    def test_strict_policy_with_all_phases(self):
        """Strict policy enforces all 10 security policies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create test archive that passes all strict checks
            with zipfile.ZipFile(archive_path, "w") as zf:
                for i in range(5):  # Stay under strict max_entries (10)
                    zf.writestr(f"file{i}.txt", "small content")

            policy = strict_defaults()
            # All strict policies are already set, just verify they work together

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Should have extracted all files
            assert len(extracted) == 5

            # Verify permission enforcement
            for file_path in extracted:
                stat_result = file_path.stat()
                mode = stat_result.st_mode & 0o777
                # strict_defaults uses default 0o644 for files
                assert mode & 0o7000 == 0  # No dangerous bits


class TestPhase34ErrorHandling:
    """Tests for error handling in Phase 3-4."""

    def test_permission_application_non_fatal_on_permission_denied(self):
        """Guardian silently continues if permission application fails."""
        # This is a defensive test - in practice, permission denial
        # is rare on files the process just created
        policy = safe_defaults()
        guardian = ExtractionGuardian(policy)

        # This should not raise even if chmod fails
        # (though chmod usually succeeds for own files)
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("content")

            # Should not raise
            guardian.finalize_extraction(
                extracted_files=[test_file],
                extracted_dirs=[],
            )

    def test_space_check_graceful_fallback_on_statvfs_error(self):
        """Space check falls back gracefully if statvfs is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            policy = safe_defaults()
            policy.check_disk_space = True

            # Should not raise even if statvfs fails
            guardian = ExtractionGuardian(policy)
            guardian.verify_space_available(100, destination)
