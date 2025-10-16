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

### `atomic_write_bytes(path, chunks)`

Backward-compatible wrapper for :func:`atomic_write`.

### `atomic_write_text(path, text)`

Atomically write ``text`` to ``path`` using :func:`atomic_write`.

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

### `from_wire(cls, value)`

Return the enum member when ``value`` matches a known code.

### `from_wire(cls, value)`

Return the matching enum member or ``UNKNOWN``.

## 3. Classes

### `WorkArtifact`

Normalized artifact describing an OpenAlex work to process.

### `Classification`

Canonical classification codes for download outcomes.

### `ReasonCode`

Machine-readable reason taxonomy for download outcomes.
