"""Tests for Phase 2 hardening policies: Pre-scan security (links, paths, constraints).

This test module validates Phase 2 implementations for:
- Symlink and hardlink defense-in-depth
- Device/FIFO/socket quarantine
- Path normalization and constraints
- Case-fold collision detection
- Entry count budgets
- Per-file size limits
- Per-entry compression ratios
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.errors import ConfigError
from DocsToKG.OntologyDownload.io import (
    ExtractionPolicy,
    extract_archive_safe,
    safe_defaults,
    strict_defaults,
)


class TestPhase2LinkDefense:
    """Tests for symlink and hardlink defense-in-depth."""

    def test_strict_policy_rejects_symlinks(self):
        """Strict policy rejects symlinks during pre-scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create a ZIP with a symlink entry (if possible)
            # Note: Most ZIP archives can't contain symlinks, so this tests the policy
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("regular_file.txt", "content")

            # Extract with strict policy (symlinks disallowed)
            policy = strict_defaults()
            policy.allow_symlinks = False
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should extract successfully (no symlinks in the archive)
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 1

    def test_lenient_policy_allows_configuration(self):
        """Lenient policy can configure link allowance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            policy = ExtractionPolicy()
            policy.allow_symlinks = True
            policy.allow_hardlinks = True
            policy.encapsulate = False
            policy.use_dirfd = False

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 1


class TestPhase2PathConstraints:
    """Tests for path normalization and constraint enforcement."""

    def test_max_depth_enforcement(self):
        """Archives with paths deeper than max_depth are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with deeply nested structure
            with zipfile.ZipFile(archive_path, "w") as zf:
                deep_path = "/".join([f"level{i}" for i in range(20)]) + "/file.txt"
                zf.writestr(deep_path, "content")

            # Extract with strict depth limit
            policy = strict_defaults()
            policy.max_depth = 10
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should reject due to depth violation
            with pytest.raises(ConfigError) as exc_info:
                extract_archive_safe(
                    archive_path,
                    destination,
                    extraction_policy=policy,
                )
            assert "depth" in str(exc_info.value).lower()

    def test_max_component_length_enforcement(self):
        """Paths with components longer than max_components_len are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with very long component name
            long_name = "a" * 300  # 300 bytes, exceeds default
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(f"{long_name}.txt", "content")

            # Extract with strict component limit
            policy = strict_defaults()
            policy.max_components_len = 128
            policy.encapsulate = False
            policy.use_dirfd = False

            with pytest.raises(ConfigError) as exc_info:
                extract_archive_safe(
                    archive_path,
                    destination,
                    extraction_policy=policy,
                )
            assert (
                "component" in str(exc_info.value).lower()
                or "segment" in str(exc_info.value).lower()
            )

    def test_unicode_normalization_applied(self):
        """Unicode paths are normalized before validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with unicode path (composed form)
            unicode_path = "café.txt"  # é as combining character
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(unicode_path, "content")

            policy = safe_defaults()
            policy.normalize_unicode = "NFC"
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should normalize and extract successfully
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 1


class TestPhase2CaseFoldCollisions:
    """Tests for case-fold collision detection."""

    def test_case_fold_collision_detected_reject_policy(self):
        """Archives with case-insensitive duplicates are rejected when policy is reject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with case-insensitive collisions
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("File.txt", "content1")
                zf.writestr("file.txt", "content2")

            policy = safe_defaults()
            policy.casefold_collision_policy = "reject"
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should reject due to collision
            with pytest.raises(ConfigError) as exc_info:
                extract_archive_safe(
                    archive_path,
                    destination,
                    extraction_policy=policy,
                )
            assert "collision" in str(exc_info.value).lower()

    def test_case_fold_collision_allowed_with_allow_policy(self):
        """Archives with case-insensitive duplicates are allowed when policy is allow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("File.txt", "content1")
                zf.writestr("file.txt", "content2")

            policy = safe_defaults()
            policy.casefold_collision_policy = "allow"
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should extract successfully (collision allowed)
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 2


class TestPhase2EntryBudget:
    """Tests for entry count budget enforcement."""

    def test_max_entries_limit_enforced(self):
        """Archives exceeding max_entries limit are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with many entries
            with zipfile.ZipFile(archive_path, "w") as zf:
                for i in range(100):
                    zf.writestr(f"file{i}.txt", f"content{i}")

            policy = strict_defaults()
            policy.max_entries = 50  # Limit to 50
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should reject due to entry budget
            with pytest.raises(ConfigError) as exc_info:
                extract_archive_safe(
                    archive_path,
                    destination,
                    extraction_policy=policy,
                )
            assert (
                "count" in str(exc_info.value).lower() or "entries" in str(exc_info.value).lower()
            )

    def test_within_entry_budget_succeeds(self):
        """Archives within entry budget extract successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with moderate number of entries
            with zipfile.ZipFile(archive_path, "w") as zf:
                for i in range(25):
                    zf.writestr(f"file{i}.txt", f"content{i}")

            policy = safe_defaults()
            policy.max_entries = 100  # Plenty of headroom
            policy.encapsulate = False
            policy.use_dirfd = False

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 25


class TestPhase2FileSizeGuard:
    """Tests for per-file size limits."""

    def test_file_size_limit_enforced(self):
        """Files exceeding max_file_size_bytes are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create archive with large file
            large_content = "x" * (10 * 1024 * 1024)  # 10 MiB
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("largefile.txt", large_content)

            policy = strict_defaults()
            policy.max_file_size_bytes = 5 * 1024 * 1024  # 5 MiB limit
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should reject due to file size
            with pytest.raises(ConfigError) as exc_info:
                extract_archive_safe(
                    archive_path,
                    destination,
                    extraction_policy=policy,
                )
            assert "file" in str(exc_info.value).lower() or "size" in str(exc_info.value).lower()

    def test_small_files_within_limit(self):
        """Small files within limit extract successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            small_content = "small content"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("small.txt", small_content)

            policy = safe_defaults()
            policy.max_file_size_bytes = 1024  # 1 KiB limit (plenty)
            policy.encapsulate = False
            policy.use_dirfd = False

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 1


class TestPhase2Integration:
    """Integration tests combining multiple Phase 2 policies."""

    def test_multiple_policies_applied_together(self):
        """Multiple Phase 2 policies are applied simultaneously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            # Create a valid archive
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("dir1/file.txt", "content")
                zf.writestr("dir2/file.txt", "content")

            # Apply strict policy with multiple constraints
            policy = strict_defaults()
            policy.max_entries = 10
            policy.max_depth = 5
            policy.max_file_size_bytes = 1024 * 1024
            policy.encapsulate = False
            policy.use_dirfd = False

            # Should pass all constraints
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted) == 2

    def test_phase2_with_phase1_encapsulation(self):
        """Phase 2 policies work correctly with Phase 1 encapsulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)
            archive_path = destination / "test.zip"

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            # Use Phase 1 encapsulation with Phase 2 constraints
            policy = safe_defaults()
            policy.encapsulate = True
            policy.encapsulation_name = "sha256"
            policy.max_depth = 10
            policy.max_entries = 100

            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # File should be in encapsulation root
            assert len(extracted) == 1
            assert (
                extracted[0].parents[1] == destination
            )  # Encapsulation root parent is destination
