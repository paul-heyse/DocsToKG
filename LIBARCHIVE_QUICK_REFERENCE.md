# LibArchive Quick Reference for DocsToKG OntologyDownload

## One-Minute Overview

The `extract_archive_safe()` function now uses **libarchive** for automatic archive detection and extraction. It replaces format-specific handlers while maintaining the same API.

```python
# Usage — exactly the same as before
from DocsToKG.OntologyDownload.io import extract_archive_safe

extracted_paths = extract_archive_safe(
    archive_path=Path("document.tar.gz"),
    destination=Path("./output"),
    logger=logger,
)
# Returns: List[Path] of files extracted
```

---

## What's New

| Aspect | Before | After |
|--------|--------|-------|
| **Formats** | ZIP, TAR, TAR.GZ | ZIP, TAR, TAR.GZ, TAR.XZ, 7Z, ISO, RAR, etc. |
| **Detection** | Suffix-based | Automatic (libarchive) |
| **Code paths** | 2+ branches | 1 unified path |
| **Security** | Per-format | Unified, hardened |

---

## When to Use

✓ Use `extract_archive_safe()` for:

- Extraction tasks in validators or pipelines
- Unknown archive formats
- User-provided archives (security critical)

✗ Don't use for:

- In-place archive modification (not supported)
- Encrypted archives (check libarchive build)
- Streaming archives > available disk (buffer to disk first)

---

## Security Guarantees

**Automatically Rejected:**

- Path traversal (`../evil`, `..\\..\\bad`)
- Absolute paths (`/etc/passwd`, `C:\Windows\...`)
- Symlinks and hardlinks
- Device nodes, FIFOs, sockets
- Zip bombs (>10:1 compression ratio)
- Extraction outside destination

**Pre-scan Promise:**

- No files written if any entry fails validation
- All-or-nothing semantics
- Fast failure (detect issues before extraction)

---

## Error Handling

```python
from DocsToKG.OntologyDownload.errors import ConfigError

try:
    extracted = extract_archive_safe(archive_path, destination)
except ConfigError as e:
    # Policy violation or format error
    # e.args[0] contains the reason (path traversal, unsupported format, etc.)
    logger.error(f"Extraction failed: {e}")
    # No files were written to destination
```

---

## Logging

Standard structured logging with JSON-friendly keys:

```python
logger.info(
    "extracted archive",
    extra={
        "stage": "extract",              # Always present
        "archive": "/path/to/archive",   # Archive path
        "files": 42,                     # Number extracted
    },
)
```

On error:

```python
logger.error(
    "archive extraction policy violation",
    extra={
        "stage": "extract",
        "archive": "/path/to/archive",
        "reason": "path traversal detected: ../evil.txt",
    },
)
```

---

## Size Limits

Control extraction size with `max_uncompressed_bytes`:

```python
# Default: uses config (typically unlimited)
extracted = extract_archive_safe(archive_path, destination)

# Custom limit: 10 MB
extracted = extract_archive_safe(
    archive_path, destination,
    max_uncompressed_bytes=10 * 1024 * 1024,  # 10 MB
)
# Raises ConfigError if uncompressed > limit
```

Check config default:

```python
from DocsToKG.OntologyDownload.settings import get_default_config

limit = get_default_config().defaults.http.max_uncompressed_bytes()
print(f"Default limit: {limit} bytes")
```

---

## Supported Formats

Query runtime support:

```python
import libarchive

# Supported formats for reading
read_formats = libarchive.ffi.WRITE_FORMATS  # Format names as strings

# Supported compression filters
read_filters = libarchive.ffi.WRITE_FILTERS   # Filter names

print("Can read:", read_formats)
print("Can decompress:", read_filters)

# Format support depends on system libarchive build
# Install `libarchive-dev` on Debian/Ubuntu or `libarchive` on macOS
```

**Common formats:**

- ZIP, TAR (ustar, GNU, POSIX pax)
- Gzip, bzip2, xz, lz4, zstd
- 7-Zip, RAR, RAR 5.0 (read-only)
- ISO 9660, UDF, Cpio, SHAR

---

## Testing

### Run Tests

```bash
# New libarchive tests
./.venv/bin/pytest tests/ontology_download/test_extract_archive_safe.py -v

# Existing tests (still pass)
./.venv/bin/pytest tests/ontology_download/test_download_behaviour.py -k extract -v
```

### Test Archive Creation (In Your Tests)

```python
import tarfile, zipfile, io
from pathlib import Path

# Create test ZIP
archive = Path("test.zip")
with zipfile.ZipFile(archive, "w") as z:
    z.writestr("file.txt", "content")

# Create test TAR
archive = Path("test.tar")
with tarfile.open(archive, "w") as t:
    data = io.BytesIO(b"content")
    info = tarfile.TarInfo("file.txt")
    info.size = len(data.getvalue())
    t.addfile(info, data)

# Create test TAR.GZ
archive = Path("test.tar.gz")
with tarfile.open(archive, "w:gz") as t:
    # ... same as above
```

---

## Troubleshooting

### "Module 'libarchive' not found"

```bash
# Install the Python binding
pip install libarchive-c>=5.3

# Install system library (if needed)
# Debian/Ubuntu:
sudo apt-get install libarchive-dev

# macOS:
brew install libarchive
```

### "Format not recognized"

1. Check if format is in `libarchive.ffi.WRITE_FORMATS`
2. Install missing codec on your system
3. Use a different format (e.g., ZIP instead of RAR)

### "Path traversal detected"

Archive contains `../` or absolute paths. This is a **security rejection** — don't bypass it. Repackage the archive.

### "Compression ratio too high"

Archive expands >10:1. Either:

- Increase limit: `max_uncompressed_bytes=larger_value`
- Inspect archive for bomb attack
- Use smaller archive

---

## Advanced: Inspection Without Extraction

For listing archive contents without extracting:

```python
import libarchive

with libarchive.file_reader("archive.zip") as archive:
    for entry in archive:
        print(f"  {entry.pathname}")
        print(f"    Size: {entry.size}")
        print(f"    Type: {'dir' if entry.isdir else 'file'}")
        # No data is extracted; just headers are read
```

---

## Migration from Old Code

### Before (Old API Still Works)

```python
from DocsToKG.OntologyDownload.io import extract_zip_safe, extract_tar_safe

if archive.endswith(".zip"):
    extracted = extract_zip_safe(archive, dest, logger=logger)
elif archive.endswith(".tar.gz"):
    extracted = extract_tar_safe(archive, dest, logger=logger)
else:
    raise ValueError("Unknown format")
```

### After (Recommended New Code)

```python
from DocsToKG.OntologyDownload.io import extract_archive_safe

# Auto-detects format — no suffix check needed
extracted = extract_archive_safe(archive, dest, logger=logger)
```

---

## Common Patterns

### Pattern 1: Extract with Error Handling

```python
from DocsToKG.OntologyDownload.errors import ConfigError
from DocsToKG.OntologyDownload.io import extract_archive_safe
from pathlib import Path

archive_path = Path("document.tar.gz")
dest_path = Path("./extracted")

try:
    extracted = extract_archive_safe(
        archive_path, dest_path, logger=logger
    )
    print(f"Extracted {len(extracted)} files")
except ConfigError as e:
    logger.error(f"Extraction failed: {e}")
    # Clean up if needed
    # (No partial files were written)
```

### Pattern 2: Extract with Size Limit

```python
from DocsToKG.OntologyDownload.io import extract_archive_safe

# Limit extraction to 100 MB
extracted = extract_archive_safe(
    archive_path, dest_path,
    logger=logger,
    max_uncompressed_bytes=100 * 1024 * 1024,
)
```

### Pattern 3: Inspect and Extract

```python
import libarchive
from DocsToKG.OntologyDownload.io import extract_archive_safe

# First, inspect
print(f"Archive contents:")
with libarchive.file_reader(str(archive_path)) as a:
    for entry in a:
        print(f"  {entry.pathname} ({entry.size} bytes)")

# Then, extract
extracted = extract_archive_safe(archive_path, dest_path)
```

---

## Configuration

Set defaults in your config YAML (under OntologyDownload settings):

```yaml
http:
  max_uncompressed_size_gb: 1  # 1 GB limit
```

Or via environment:

```bash
export ONTOFETCH_MAX_UNCOMPRESSED_SIZE_GB=1
```

---

## Performance Notes

- **Two-pass model:** Pre-scan (headers) then extract (data)
- **Streaming:** One-pass through file data; no buffering in memory
- **Overhead:** Negligible for typical archives (<100 MB)
- **Benchmark:** ZIP extraction ~same speed as before

---

## References

- **Module:** `DocsToKG.OntologyDownload.io.filesystem.extract_archive_safe`
- **Tests:** `tests/ontology_download/test_extract_archive_safe.py`
- **Doc:** `src/DocsToKG/OntologyDownload/LIBARCHIVE_MIGRATION.md`
- **libarchive-c:** <https://github.com/Changaco/python-libarchive-c>

---

**Last Updated:** October 2025
**Status:** Production Ready
