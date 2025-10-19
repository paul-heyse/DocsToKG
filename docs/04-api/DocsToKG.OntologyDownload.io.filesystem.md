# 1. Module: filesystem

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.io.filesystem``.

## 1. Overview

Filesystem helpers for ontology downloads.

## 2. Functions

### `_resolve_max_uncompressed_bytes(limit)`

Return the effective archive expansion limit, honoring runtime overrides.

### `sanitize_filename(filename)`

Return a filesystem-safe filename derived from ``filename``.

### `generate_correlation_id()`

Return a short-lived identifier that links related log entries.

### `mask_sensitive_data(payload)`

Return a copy of ``payload`` with common secret fields masked.

### `sha256_file(path)`

Compute the SHA-256 digest for the provided file.

### `_compute_file_hash(path, algorithm)`

Compute ``algorithm`` digest for ``path``.

### `_validate_member_path(member_name)`

Validate archive member paths to prevent traversal attacks.

### `_check_compression_ratio()`

Ensure compressed archives do not expand beyond the permitted ratio.

### `_enforce_uncompressed_ceiling()`

Ensure uncompressed payload stays within configured limits.

### `extract_zip_safe(zip_path, destination)`

Extract a ZIP archive while preventing traversal and compression bombs.

### `extract_tar_safe(tar_path, destination)`

Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

### `extract_archive_safe(archive_path, destination)`

Extract archives by dispatching to the appropriate safe handler.

### `_materialize_cached_file(source, destination)`

Link or move ``source`` into ``destination`` without redundant copies.

Returns a tuple ``(artifact_path, cache_path)`` where ``artifact_path`` is the final
destination path and ``cache_path`` points to the retained cache file (which may be
identical to ``artifact_path`` when the cache entry is moved instead of linked).

### `format_bytes(num)`

Return a human-readable representation for ``num`` bytes.

### `_mask_value(value, key_hint)`

*No documentation available.*
