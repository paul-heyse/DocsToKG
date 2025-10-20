# Libarchive Migration: Extract Archive Safe

## Overview

The `extract_archive_safe()` function has been refactored to use **libarchive-c** instead of dispatching to format-specific handlers (`extract_zip_safe`, `extract_tar_safe`). This consolidation provides:

- **Single, robust API** for all archive formats (ZIP, TAR, TAR.GZ, TAR.XZ, 7Z, ISO, RAR, etc.)
- **Automatic format and compression detection** (no suffix-based routing)
- **Streaming architecture** with zero-copy processing
- **Enhanced security by default** with libarchive's secure extraction flags
- **Reduced code maintenance burden** (eliminating per-format branching)

## What Changed

### Public API (Unchanged)

The function signature and behavior remain the same:

```python
def extract_archive_safe(
    archive_path: Path,
    destination: Path,
    *,
    logger: Optional[logging.Logger] = None,
    max_uncompressed_bytes: Optional[int] = None,
) -> List[Path]:
    """Extract archives safely with validation and compression checks."""
```

- **Input**: Archive file path + destination directory
- **Output**: List of extracted file paths (regular files only, in header order)
- **Exceptions**: `ConfigError` on format errors, security violations, or size limits

### Call Sites (No Changes Required)

All existing call sites continue to work without modification:

```python
# Before and after - unchanged
extracted = extract_archive_safe(archive_path, destination, logger=logger)
```

### Logging (Preserved)

Structured log records still use the same keys:

```python
logger.info(
    "extracted archive",
    extra={
        "stage": "extract",           # ← Unchanged key
        "archive": str(archive),      # ← Unchanged
        "files": len(extracted),      # ← Unchanged
    },
)
```

### Format-Specific Functions (Deprecated but Preserved)

The legacy functions remain for backward compatibility:

- `extract_zip_safe(...)`
- `extract_tar_safe(...)`

These are still exported from the public API but **internally delegate to `extract_archive_safe()`** in future versions. Direct calls to these functions still work but new code should use `extract_archive_safe()`.

## Implementation Details

### Two-Phase Extraction

The new implementation uses a **pre-scan then extract** pattern:

#### Phase 1: Pre-Scan (No Writes)

```
for each archive entry:
  1. Validate entry type (reject symlinks/hardlinks/devices)
  2. Validate path (reject traversal, absolute paths)
  3. Check containment (path must stay within destination)
  4. Accumulate uncompressed size

Check compression ratio (~10:1)
Check uncompressed size ceiling
```

**Benefits:**

- Fail fast before any writes
- Never leave partial extractions on error
- Deterministic bomb detection (ratio checked before extraction)

#### Phase 2: Extract (Conditional)

```
If Phase 1 passed:
  for each entry:
    - Create directories as needed
    - Stream file contents to validated paths
    - Track extracted file paths
  return List[Path]
```

### Security Guarantees

✓ **Path Traversal**: Rejected at pre-scan (no `..`, no absolute paths)
✓ **Symlink/Hardlink**: Rejected at pre-scan (entry type check)
✓ **Device/FIFO/Socket**: Rejected at pre-scan (special file check)
✓ **Zip Bombs**: Rejected at pre-scan (10:1 ratio guard)
✓ **Containment**: Every path verified to stay within destination
✓ **No Partial Writes**: Pre-scan ensures all-or-nothing behavior

### Supported Formats

libarchive auto-detects and supports (subject to system libarchive build):

**Read Formats:**

- TAR, POSIX tar (ustar), GNU tar, PAX
- ZIP, PKWARE (encrypted)
- 7-Zip, RAR, RAR 5.0
- ISO 9660, UDF
- Cpio, SHAR
- CAB, LHA, MTREE, WARC, XAR

**Compression Filters:**

- gzip, bzip2, lzma, xz, zstd, lz4, compress, uuencode

For a format or compression filter to work, the system `libarchive` library must have been compiled with that codec support. Query available formats at runtime:

```python
import libarchive
print(libarchive.ffi.WRITE_FORMATS)   # formats supported for writing
print(libarchive.ffi.WRITE_FILTERS)   # compression filters supported
```

## Testing

A comprehensive test suite covers:

1. **Happy paths**: ZIP, TAR, TAR.GZ extraction with nested directories
2. **Security**: Path traversal, absolute paths, symlinks, hardlinks, devices
3. **Bomb guard**: Compression ratio detection and enforcement
4. **Logging**: Structured logging with correct keys and values
5. **Idempotence**: Re-extraction consistency
6. **Edge cases**: Empty archives, missing files, corrupted archives, unicode names

Run tests with:

```bash
./.venv/bin/pytest tests/ontology_download/test_extract_archive_safe.py -v
```

## Dependencies

**Added:**

- `libarchive-c>=5.3` (Python bindings to libarchive)

**System Requirement:**

- libarchive development library (e.g., `libarchive-dev` on Debian/Ubuntu, `libarchive` on macOS)

## Migration Notes

### For Existing Code

✓ **No changes required** — existing `extract_archive_safe()` calls work as-is.

### For New Code

Use `extract_archive_safe()` for all archive types; it detects format automatically.

### For Tests

- **Keep existing tests** for `extract_zip_safe` and `extract_tar_safe` (they still work)
- **Add new tests** using `extract_archive_safe()` for coverage of edge cases and security

### For Troubleshooting

If an archive format or compression codec is not supported:

1. Check system libarchive: `libarchive --version` or similar
2. Query Python binding: `libarchive.ffi.WRITE_FORMATS`
3. Install missing codec (OS-dependent) or use a different format

## References

- **libarchive-c GitHub**: <https://github.com/Changaco/python-libarchive-c>
- **libarchive Manual**: <https://www.libarchive.org/>
- **Release Notes (5.3)**: <https://github.com/Changaco/python-libarchive-c/releases>

## Acceptance Checklist

- [x] `extract_archive_safe` uses libarchive (no direct zipfile/tarfile calls)
- [x] Two-phase flow: pre-scan validation then extract
- [x] Path traversal and absolute path attempts blocked
- [x] Symlinks, hardlinks, devices rejected by policy
- [x] Zip-bomb guard (~10:1 ratio) enforced
- [x] Destination containment verified for every member
- [x] Logging keys unchanged (`stage="extract"`, `archive`, `files`)
- [x] Return type `List[Path]` preserved
- [x] Comprehensive test suite added
- [x] Format-specific functions preserved for backward compatibility
- [x] No regressions in existing tests
