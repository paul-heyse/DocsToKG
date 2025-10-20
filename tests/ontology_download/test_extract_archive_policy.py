"""Tests for Phase 1 hardening policies: encapsulation and DirFD semantics.

This test module validates the new ExtractionPolicy infrastructure and Phase 1
implementations for single-root encapsulation and DirFD + openat semantics.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.errors import ConfigError
from DocsToKG.OntologyDownload.io import (
    ExtractionPolicy,
    extract_archive_safe,
    lenient_defaults,
    safe_defaults,
    strict_defaults,
)


class TestExtractionPolicy:
    """Tests for ExtractionPolicy configuration and validation."""

    def test_safe_defaults_creates_valid_policy(self):
        """safe_defaults() returns a valid policy with all protections enabled."""
        policy = safe_defaults()
        assert policy.is_valid()
        assert policy.encapsulate is True
        assert policy.use_dirfd is True
        assert policy.allow_symlinks is False
        assert policy.allow_hardlinks is False

    def test_lenient_defaults_creates_valid_policy(self):
        """lenient_defaults() returns a valid policy with relaxed limits."""
        policy = lenient_defaults()
        assert policy.is_valid()
        assert policy.allow_symlinks is True
        assert policy.allow_hardlinks is True
        assert policy.max_entries > safe_defaults().max_entries

    def test_strict_defaults_creates_valid_policy(self):
        """strict_defaults() returns a valid policy with maximum protection."""
        policy = strict_defaults()
        assert policy.is_valid()
        assert policy.max_depth < safe_defaults().max_depth
        assert policy.max_entries < safe_defaults().max_entries

    def test_policy_validation_rejects_invalid_encapsulation_name(self):
        """Validation rejects unknown encapsulation_name."""
        policy = safe_defaults()
        policy.encapsulation_name = "invalid"  # type: ignore
        errors = policy.validate()
        assert len(errors) > 0
        assert "encapsulation_name" in errors[0]

    def test_policy_validation_rejects_dirfd_without_encapsulation(self):
        """Validation rejects use_dirfd=True with encapsulate=False."""
        policy = safe_defaults()
        policy.encapsulate = False
        policy.use_dirfd = True
        errors = policy.validate()
        assert len(errors) > 0
        assert "use_dirfd requires encapsulate" in errors[0]

    def test_policy_validation_rejects_invalid_max_depth(self):
        """Validation rejects non-positive max_depth."""
        policy = safe_defaults()
        policy.max_depth = 0
        errors = policy.validate()
        assert len(errors) > 0
        assert "max_depth" in errors[0]

    def test_policy_validation_rejects_invalid_mode(self):
        """Validation rejects invalid file/directory modes."""
        policy = safe_defaults()
        policy.dir_mode = 0o777 + 1  # Invalid: > 0o777
        errors = policy.validate()
        assert len(errors) > 0

    def test_policy_summary_returns_readable_status(self):
        """summary() returns human-readable policy status."""
        policy = safe_defaults()
        summary = policy.summary()
        assert "Phase 1: Encapsulation" in summary
        assert "Phase 1: DirFD" in summary
        assert "enabled" in summary["Phase 1: Encapsulation"]
        assert "rejected" in summary["Phase 2: Symlinks"]


class TestEncapsulationPhase1:
    """Tests for Phase 1 encapsulation with SHA256/basename naming."""

    def test_extract_with_encapsulation_sha256(self):
        """Extraction with encapsulation creates SHA256-named subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create a small test archive
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file1.txt", "content1")
                zf.writestr("dir1/file2.txt", "content2")

            # Extract with encapsulation (sha256 policy)
            policy = safe_defaults()
            policy.encapsulation_name = "sha256"
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Verify encapsulation root was created
            encapsulation_dirs = list(destination.glob("*.d"))
            assert len(encapsulation_dirs) == 1
            encapsulation_root = encapsulation_dirs[0]

            # Verify files are extracted under encapsulation root
            assert len(extracted) == 2
            for path in extracted:
                assert path.is_relative_to(encapsulation_root)

            # Verify file contents
            assert (encapsulation_root / "file1.txt").read_text() == "content1"
            assert (encapsulation_root / "dir1" / "file2.txt").read_text() == "content2"

    def test_extract_with_encapsulation_basename(self):
        """Extraction with encapsulation creates basename-named subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create a small test archive
            archive_path = destination / "my_archive.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            # Extract with basename encapsulation
            policy = safe_defaults()
            policy.encapsulation_name = "basename"
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Verify encapsulation root has basename
            expected_root = destination / "my_archive.d"
            assert expected_root.exists()
            assert len(extracted) == 1
            assert extracted[0].is_relative_to(expected_root)

    def test_encapsulation_rejects_existing_root(self):
        """Extraction rejects if encapsulation root already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create an archive
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            # First extraction should succeed
            policy = safe_defaults()
            policy.encapsulation_name = "sha256"
            extracted1 = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )
            assert len(extracted1) == 1

            # Second extraction (same archive) should fail
            with pytest.raises(ConfigError) as exc_info:
                extract_archive_safe(
                    archive_path,
                    destination,
                    extraction_policy=policy,
                )
            assert "already exists" in str(exc_info.value).lower()

    def test_re_extract_different_archives_same_destination(self):
        """Different archives can be extracted to same destination (different roots)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create two different archives
            archive1 = destination / "archive1.zip"
            with zipfile.ZipFile(archive1, "w") as zf:
                zf.writestr("file.txt", "content1")

            archive2 = destination / "archive2.zip"
            with zipfile.ZipFile(archive2, "w") as zf:
                zf.writestr("file.txt", "content2")

            # Both should extract to different encapsulation roots
            policy = safe_defaults()
            policy.encapsulation_name = "sha256"

            extracted1 = extract_archive_safe(
                archive1,
                destination,
                extraction_policy=policy,
            )
            extracted2 = extract_archive_safe(
                archive2,
                destination,
                extraction_policy=policy,
            )

            # Verify two separate roots
            encapsulation_dirs = list(destination.glob("*.d"))
            assert len(encapsulation_dirs) == 2

            # Verify each extract is under its own root
            root1 = extracted1[0].parents[0]  # Encapsulation root is the direct parent
            root2 = extracted2[0].parents[0]  # Encapsulation root is the direct parent
            assert root1 != root2
            assert root1.parent == destination
            assert root2.parent == destination


class TestEncapsulationWithoutEncapsulation:
    """Tests for extraction without encapsulation (disabled policy)."""

    def test_extract_without_encapsulation(self):
        """Extraction with encapsulate=False extracts directly to destination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create a test archive
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            # Extract without encapsulation (and disable dirfd since it requires encapsulation)
            policy = safe_defaults()
            policy.encapsulate = False
            policy.use_dirfd = False
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Verify file is directly in destination
            assert len(extracted) == 1
            assert extracted[0] == destination / "file.txt"
            assert extracted[0].read_text() == "content"


class TestPolicyConfigurationEdgeCases:
    """Tests for policy configuration edge cases and constraints."""

    def test_policy_respects_max_entries_constraint(self):
        """Policy max_entries constraint is tracked during pre-scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create archive with 5 files
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                for i in range(5):
                    zf.writestr(f"file{i}.txt", f"content{i}")

            # Extract with safe policy (max_entries will be checked in later phase)
            policy = safe_defaults()
            policy.encapsulate = False
            policy.use_dirfd = False
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            # Should extract all 5 files (Phase 1 just counts)
            assert len(extracted) == 5

    def test_policy_preserves_directory_structure(self):
        """Encapsulation preserves nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create archive with nested structure
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("a/b/c/file.txt", "nested content")

            policy = safe_defaults()
            policy.encapsulation_name = "sha256"
            extracted = extract_archive_safe(
                archive_path,
                destination,
                extraction_policy=policy,
            )

            assert len(extracted) == 1
            assert extracted[0].name == "file.txt"
            assert extracted[0].read_text() == "nested content"
            # Verify directory hierarchy preserved (a/b/c is nested under encapsulation root)
            # Structure: encapsulation_root/a/b/c/file.txt
            # So parent 1=c, parent 2=b, parent 3=a, parent 4=encapsulation_root
            assert extracted[0].parent.name == "c"
            assert extracted[0].parent.parent.name == "b"
            assert extracted[0].parent.parent.parent.name == "a"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing extract_archive_safe calls."""

    def test_extract_without_policy_parameter(self):
        """Existing calls without extraction_policy parameter still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create test archive
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            # Call without extraction_policy parameter (uses safe defaults)
            extracted = extract_archive_safe(
                archive_path,
                destination,
            )

            # Should succeed with default encapsulation
            assert len(extracted) == 1

    def test_extract_with_old_parameters_still_work(self):
        """Calls with logger and max_uncompressed_bytes parameters still work."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            # Create test archive
            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            logger = logging.getLogger("test")
            extracted = extract_archive_safe(
                archive_path,
                destination,
                logger=logger,
                max_uncompressed_bytes=100 * 1024 * 1024,
            )

            assert len(extracted) == 1


class TestTelemetryIntegration:
    """Tests for telemetry capture during extraction."""

    def test_extraction_logs_encapsulated_root(self, caplog):
        """Extraction logs the encapsulated_root path."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir)

            archive_path = destination / "test.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("file.txt", "content")

            logger = logging.getLogger("test")
            logger.setLevel(logging.INFO)

            policy = safe_defaults()
            policy.encapsulation_name = "sha256"
            caplog.set_level(logging.INFO)

            extract_archive_safe(
                archive_path,
                destination,
                logger=logger,
                extraction_policy=policy,
            )

            # Should have logged extraction info
            assert any("extracted archive" in record.message for record in caplog.records)


# Fixtures for archive creation
@pytest.fixture
def small_zip_archive(tmp_path: Path) -> Path:
    """Create a small test ZIP archive."""
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("file1.txt", "content1")
        zf.writestr("dir1/file2.txt", "content2")
    return archive_path


@pytest.fixture
def nested_zip_archive(tmp_path: Path) -> Path:
    """Create a ZIP archive with deeply nested structure."""
    archive_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("a/b/c/d/file.txt", "deeply nested")
    return archive_path
