# 1. Module: core

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.core``.

## 1. Overview

Core primitives for the DocsToKG content download pipeline.

This module consolidates the shared taxonomy enums, payload classification
heuristics, and identifier normalisation helpers that were previously spread
across ``classifications``, ``classifier``, and ``utils``. Co-locating these
utilities keeps the public surface that other modules consume in one place,
simplifying imports for both the CLI and resolver pipeline.

## 2. Functions

### `atomic_write(path, chunks)`

Atomically write ``chunks`` to ``path`` and return the byte count.

Performance Note:
When ``hasher`` is provided, uses an optimized code path that avoids
conditional checks in the hot loop for better throughput on large files.

### `atomic_write_text(path, text)`

Atomically write ``text`` to ``path`` using :func:`atomic_write`.

### `normalize_classification(value)`

Return a normalized classification token preserving unknown custom codes.

### `normalize_reason(value)`

Return a normalized reason token preserving unknown custom codes.

### `classify_payload(head_bytes, content_type, url)`

Classify a payload as ``Classification.PDF``/``Classification.HTML`` or ``Classification.UNKNOWN``.

### `_extract_filename_from_disposition(disposition)`

Return the filename component from a Content-Disposition header.

### `parse_size(value)`

Parse human-friendly size strings like ``10GB`` into byte counts.

### `_infer_suffix(url, content_type, disposition, classification, default_suffix)`

Infer a destination suffix from HTTP hints and classification heuristics.

### `update_tail_buffer(buffer, chunk)`

Maintain a sliding window of the trailing ``limit`` bytes.

### `has_pdf_eof(path)`

Return ``True`` when the PDF at ``path`` ends with ``%%EOF`` marker.

### `tail_contains_html(tail)`

Heuristic to detect HTML signatures in the trailing payload bytes.

### `normalize_doi(doi)`

Normalize DOI identifiers by stripping common prefixes and whitespace.

### `normalize_pmcid(pmcid)`

Normalize PMCID values ensuring a canonical PMC prefix.

### `strip_prefix(value, prefix)`

Strip a case-insensitive prefix from a string when present.

### `dedupe(items)`

Remove duplicates while preserving the first occurrence order.

### `normalize_pmid(pmid)`

Extract the numeric PubMed identifier from the supplied string.

### `normalize_arxiv(arxiv_id)`

Normalize arXiv identifiers by removing common prefixes and whitespace.

### `slugify(text, keep)`

Create a filesystem-friendly slug from the provided text.

### `normalize_url(url)`

Return a canonicalised version of ``url`` suitable for deduplication.

### `__post_init__(self)`

*No documentation available.*

### `__post_init__(self)`

*No documentation available.*

### `from_mapping(cls, data)`

Construct a context instance from a mapping-based payload.

### `mark_explicit(self)`

Record that the given fields were explicitly provided by the caller.

### `is_explicit(self, field)`

Return ``True`` when ``field`` was explicitly provided by the caller.

### `to_dict(self)`

Serialize the context to a mapping for legacy integrations.

### `clone_for_download(self)`

Return a shallow clone suitable for per-download mutation.

### `_normalize_sequence(value)`

*No documentation available.*

### `_normalize_mapping(value)`

*No documentation available.*

### `_coerce_optional_positive(value)`

*No documentation available.*

### `_coerce_non_negative(value, default)`

*No documentation available.*

### `from_wire(cls, value)`

Return the enum member when ``value`` matches a known code.

### `from_wire(cls, value)`

Return the matching enum member or ``UNKNOWN``.

### `_pop(name, default)`

*No documentation available.*

## 3. Classes

### `WorkArtifact`

Normalized artifact describing an OpenAlex work to process.

### `DownloadContext`

Typed execution context shared by the CLI and resolver pipeline.

### `Classification`

Canonical classification codes for download outcomes.

### `ReasonCode`

Machine-readable reason taxonomy for download outcomes.
