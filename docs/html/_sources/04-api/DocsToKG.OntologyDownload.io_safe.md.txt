# 1. Module: io_safe

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.io_safe``.

## 1. Overview

Filesystem and payload safety utilities for the ontology downloader.

## 2. Functions

### `sanitize_filename(filename)`

Return a filesystem-safe filename derived from ``filename``.

### `generate_correlation_id()`

Return a short-lived identifier that links related log entries.

### `mask_sensitive_data(payload)`

Return a copy of ``payload`` with common secret fields masked.

### `_enforce_idn_safety(host)`

Validate internationalized hostnames and reject suspicious patterns.

### `_rebuild_netloc(parsed, ascii_host)`

Reconstruct URL netloc with a normalized hostname.

### `validate_url_security(url, http_config)`

Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

### `sha256_file(path)`

Compute the SHA-256 digest for the provided file.

### `_validate_member_path(member_name)`

Validate archive member paths to prevent traversal attacks.

### `_check_compression_ratio()`

Ensure compressed archives do not expand beyond the permitted ratio.

### `extract_zip_safe(zip_path, destination)`

Extract a ZIP archive while preventing traversal and compression bombs.

### `extract_tar_safe(tar_path, destination)`

Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

### `extract_archive_safe(archive_path, destination)`

Extract archives by dispatching to the appropriate safe handler.
