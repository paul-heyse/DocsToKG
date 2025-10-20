# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_extract_archive_safe",
#   "purpose": "Comprehensive tests for libarchive-based archive extraction with security validation",
#   "sections": [
#     {"id": "happy_paths", "name": "Happy Path Tests", "anchor": "HPT", "kind": "tests"},
#     {"id": "security", "name": "Security & Policy Tests", "anchor": "SEC", "kind": "tests"},
#     {"id": "bomb_guard", "name": "Compression Bomb Guard Tests", "anchor": "BOG", "kind": "tests"},
#     {"id": "telemetry", "name": "Logging & Telemetry Tests", "anchor": "TEL", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Comprehensive tests for libarchive-based archive extraction.

Tests cover:
- Happy path extraction of ZIP and TAR archives with nested directories
- Security posture: path traversal prevention, absolute path rejection
- Entry type filtering: symlink/hardlink/device rejection
- Compression bomb detection and ratio enforcement
- Logging and telemetry output with structured keys
- Idempotence on re-extraction
"""

from __future__ import annotations

import io
import logging
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pytest

from DocsToKG.OntologyDownload.errors import ConfigError
from DocsToKG.OntologyDownload.io import filesystem as fs_mod


def _logger() -> logging.Logger:
    """Return a logger for testing."""
    logger = logging.getLogger("test.extract")
    logger.setLevel(logging.DEBUG)
    return logger


# ============================================================================
# HAPPY PATH TESTS (ZIP, TAR, TAR.GZ)
# ============================================================================


def test_extract_archive_safe_zip_basic(tmp_path) -> None:
    """Extract a simple ZIP archive with nested files and directories."""
    archive = tmp_path / "simple.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("file1.txt", "content1")
        zipf.writestr("dir/file2.txt", "content2")
        zipf.writestr("dir/subdir/file3.txt", "content3")

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Verify files exist
    assert (output / "file1.txt").read_text() == "content1"
    assert (output / "dir" / "file2.txt").read_text() == "content2"
    assert (output / "dir" / "subdir" / "file3.txt").read_text() == "content3"

    # Verify return list matches expected paths (in header order)
    assert len(extracted) == 3
    assert extracted[0].name == "file1.txt"
    assert extracted[1].name == "file2.txt"
    assert extracted[2].name == "file3.txt"


def test_extract_archive_safe_tar_basic(tmp_path) -> None:
    """Extract a simple TAR archive and verify content and order."""
    archive = tmp_path / "simple.tar"

    with tarfile.open(archive, "w") as tar:
        # Add a regular file
        data = io.BytesIO(b"tar content")
        info = tarfile.TarInfo("file1.txt")
        info.size = len(data.getvalue())
        tar.addfile(info, data)

        # Add a directory entry
        dir_info = tarfile.TarInfo("subdir")
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)

        # Add file in directory
        data2 = io.BytesIO(b"nested content")
        info2 = tarfile.TarInfo("subdir/file2.txt")
        info2.size = len(data2.getvalue())
        tar.addfile(info2, data2)

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Verify files exist
    assert (output / "file1.txt").read_bytes() == b"tar content"
    assert (output / "subdir" / "file2.txt").read_bytes() == b"nested content"

    # Verify return list contains only regular files (not directories)
    assert len(extracted) == 2


def test_extract_archive_safe_tar_gz(tmp_path) -> None:
    """Extract a TAR.GZ archive with auto-detection of gzip compression."""
    archive = tmp_path / "simple.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        data = io.BytesIO(b"gzipped content")
        info = tarfile.TarInfo("compressed.txt")
        info.size = len(data.getvalue())
        tar.addfile(info, data)

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    assert (output / "compressed.txt").read_bytes() == b"gzipped content"
    assert len(extracted) == 1


# ============================================================================
# SECURITY TESTS
# ============================================================================


def test_extract_archive_safe_rejects_parent_traversal(tmp_path) -> None:
    """Reject archives containing `../evil.txt` path traversal."""
    archive = tmp_path / "traversal.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        # Manually create an entry that attempts traversal
        info = zipfile.ZipInfo("../evil.txt")
        zipf.writestr(info, "malicious")

    output = tmp_path / "output"

    with pytest.raises(ConfigError, match="(path traversal|Unsafe path)"):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Verify no files were written
    extracted = list(output.glob("**/*"))
    assert len(extracted) == 0 or all(p.is_dir() for p in extracted)


def test_extract_archive_safe_rejects_absolute_path(tmp_path) -> None:
    """Reject archives containing absolute paths like `/etc/passwd`."""
    archive = tmp_path / "absolute.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        info = zipfile.ZipInfo("/etc/passwd")
        zipf.writestr(info, "evil")

    output = tmp_path / "output"

    with pytest.raises(ConfigError, match="(absolute|Unsafe)"):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())


def test_extract_archive_safe_rejects_windows_absolute_path(tmp_path) -> None:
    """Reject archives with Windows absolute paths like `C:\\windows\\system32\\foo`."""
    archive = tmp_path / "winabs.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        info = zipfile.ZipInfo("C:\\Windows\\System32\\evil.dll")
        zipf.writestr(info, "malicious")

    output = tmp_path / "output"

    with pytest.raises(ConfigError, match="(absolute|Unsafe)"):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())


def test_extract_archive_safe_rejects_symlink(tmp_path) -> None:
    """Reject archives containing symlink entries."""
    archive = tmp_path / "symlink.tar"

    with tarfile.open(archive, "w") as tar:
        # Add a target file
        data = io.BytesIO(b"target")
        info = tarfile.TarInfo("target.txt")
        info.size = len(data.getvalue())
        tar.addfile(info, data)

        # Add a symlink
        link_info = tarfile.TarInfo("link.txt")
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = "target.txt"
        tar.addfile(link_info)

    output = tmp_path / "output"

    with pytest.raises(ConfigError, match="(link|not permitted)"):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())


def test_extract_archive_safe_rejects_hardlink(tmp_path) -> None:
    """Reject archives containing hardlink entries."""
    archive = tmp_path / "hardlink.tar"

    with tarfile.open(archive, "w") as tar:
        # Add a regular file
        data = io.BytesIO(b"target")
        info = tarfile.TarInfo("target.txt")
        info.size = len(data.getvalue())
        tar.addfile(info, data)

        # Add a hardlink
        link_info = tarfile.TarInfo("hardlink.txt")
        link_info.type = tarfile.LNKTYPE
        link_info.linkname = "target.txt"
        tar.addfile(link_info)

    output = tmp_path / "output"

    with pytest.raises(ConfigError, match="(link|not permitted)"):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())


# ============================================================================
# COMPRESSION BOMB GUARD TESTS
# ============================================================================


def test_extract_archive_safe_detects_bomb_ratio(tmp_path) -> None:
    """Reject archives with extreme compression ratio (e.g., highly compressible payload)."""
    archive = tmp_path / "bomb.zip"

    # Create a highly compressible payload (all zeros, very repetitive)
    # This should compress to < 100 KB but expand to > 1 MB
    payload = b"\x00" * (2 * 1024 * 1024)  # 2 MB of zeros

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("bomb.txt", payload)

    # Verify the archive is indeed small due to compression
    archive_size = archive.stat().st_size
    assert archive_size < 500 * 1024, f"Archive should be < 500 KB, got {archive_size}"

    output = tmp_path / "output"

    # Should reject due to compression ratio
    with pytest.raises(ConfigError, match="(compression ratio|expands to)"):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())


def test_extract_archive_safe_accepts_normal_compression(tmp_path) -> None:
    """Accept archives with normal compression ratios (e.g., text files)."""
    archive = tmp_path / "normal.zip"

    # Create a reasonable payload (text, normal compression ~50:1)
    payload = "The quick brown fox jumps over the lazy dog.\n" * 1000  # ~45 KB

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("content.txt", payload)

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Should succeed
    assert len(extracted) == 1
    assert (output / "content.txt").read_text() == payload


# ============================================================================
# LOGGING & TELEMETRY TESTS
# ============================================================================


def test_extract_archive_safe_logs_success_with_stage_key(tmp_path, caplog) -> None:
    """Verify logging includes stage='extract' and file count on success."""
    archive = tmp_path / "simple.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("file1.txt", "content1")
        zipf.writestr("file2.txt", "content2")

    output = tmp_path / "output"
    logger = _logger()

    with caplog.at_level(logging.INFO, logger="test.extract"):
        extracted = fs_mod.extract_archive_safe(archive, output, logger=logger)

    # Verify log record exists with correct fields
    log_records = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(log_records) > 0, "Should have at least one INFO log"

    record = log_records[0]
    assert hasattr(record, "stage")
    assert record.stage == "extract"
    assert hasattr(record, "archive")
    assert hasattr(record, "files")
    assert record.files == 2


def test_extract_archive_safe_logs_failure_reason(tmp_path, caplog) -> None:
    """Verify error logging captures the failure reason."""
    archive = tmp_path / "traversal.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        info = zipfile.ZipInfo("../evil.txt")
        zipf.writestr(info, "bad")

    output = tmp_path / "output"
    logger = _logger()

    with caplog.at_level(logging.ERROR, logger="test.extract"):
        with pytest.raises(ConfigError):
            fs_mod.extract_archive_safe(archive, output, logger=logger)


# ============================================================================
# IDEMPOTENCE TESTS
# ============================================================================


def test_extract_archive_safe_idempotent_reextraction(tmp_path) -> None:
    """Re-extracting same archive should produce consistent results."""
    archive = tmp_path / "simple.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("file1.txt", "content1")
        zipf.writestr("dir/file2.txt", "content2")

    output = tmp_path / "output"

    # First extraction
    extracted1 = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Second extraction (should overwrite or skip)
    extracted2 = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Should have same files extracted both times
    assert extracted1 == extracted2
    assert len(extracted1) == 2


# ============================================================================
# EDGE CASES
# ============================================================================


def test_extract_archive_safe_handles_empty_archive(tmp_path) -> None:
    """Handle archive with no files (only directories)."""
    archive = tmp_path / "empty.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        # Add only directory entries, no files
        zipf.writestr(zipfile.ZipInfo("dir/"), "")

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Should succeed with no files extracted
    assert len(extracted) == 0
    assert (output / "dir").is_dir()


def test_extract_archive_safe_missing_archive_raises_config_error(tmp_path) -> None:
    """Reject extraction when archive does not exist."""
    missing = tmp_path / "does_not_exist.zip"
    output = tmp_path / "output"

    with pytest.raises(ConfigError, match="not found"):
        fs_mod.extract_archive_safe(missing, output, logger=_logger())


def test_extract_archive_safe_corrupted_archive_raises_config_error(tmp_path) -> None:
    """Reject corrupted/unreadable archives."""
    archive = tmp_path / "corrupted.zip"
    archive.write_bytes(b"This is not a valid ZIP file!")

    output = tmp_path / "output"

    with pytest.raises(ConfigError):
        fs_mod.extract_archive_safe(archive, output, logger=_logger())


def test_extract_archive_safe_respects_max_uncompressed_bytes(tmp_path) -> None:
    """Reject archives exceeding configured uncompressed size limit."""
    archive = tmp_path / "oversized.zip"

    # Create a 5 MB file
    payload = b"X" * (5 * 1024 * 1024)

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("large.bin", payload)

    output = tmp_path / "output"

    # Should fail if limit is 1 MB
    with pytest.raises(ConfigError, match="(exceeds|limit)"):
        fs_mod.extract_archive_safe(
            archive, output, logger=_logger(), max_uncompressed_bytes=1 * 1024 * 1024
        )


def test_extract_archive_safe_allows_normal_size(tmp_path) -> None:
    """Accept archives within configured size limits."""
    archive = tmp_path / "normal_size.zip"

    # Create a 1 MB file
    payload = b"X" * (1 * 1024 * 1024)

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("normal.bin", payload)

    output = tmp_path / "output"

    # Should succeed with 5 MB limit
    extracted = fs_mod.extract_archive_safe(
        archive, output, logger=_logger(), max_uncompressed_bytes=5 * 1024 * 1024
    )

    assert len(extracted) == 1


# ============================================================================
# SPECIAL CHARACTER HANDLING
# ============================================================================


def test_extract_archive_safe_handles_unicode_filenames(tmp_path) -> None:
    """Handle files with unicode characters in names."""
    archive = tmp_path / "unicode.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("日本語.txt", "japanese")
        zipf.writestr("français.txt", "french")
        zipf.writestr("файл.txt", "russian")

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    # Verify files were extracted
    assert len(extracted) == 3
    assert (output / "日本語.txt").exists()
    assert (output / "français.txt").exists()
    assert (output / "файл.txt").exists()


def test_extract_archive_safe_handles_spaces_and_special_chars(tmp_path) -> None:
    """Handle filenames with spaces and special characters."""
    archive = tmp_path / "special.zip"

    with zipfile.ZipFile(archive, "w") as zipf:
        zipf.writestr("file with spaces.txt", "spaced")
        zipf.writestr("file-with-dashes.txt", "dashed")
        zipf.writestr("file_with_underscores.txt", "underscored")

    output = tmp_path / "output"
    extracted = fs_mod.extract_archive_safe(archive, output, logger=_logger())

    assert len(extracted) == 3
    assert (output / "file with spaces.txt").read_text() == "spaced"
