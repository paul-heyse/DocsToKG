# 1. Module: checksums

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.checksums``.

## 1. Overview

Checksum parsing and resolution helpers.

## 2. Functions

### `_normalize_algorithm(algorithm)`

*No documentation available.*

### `_normalize_checksum(algorithm, value)`

*No documentation available.*

### `parse_checksum_extra(value)`

Normalize checksum extras to ``(algorithm, value)`` tuples.

### `parse_checksum_url_extra(value)`

Normalise checksum URL extras to ``(url, algorithm)`` tuples.

### `_extract_checksum_from_text(text)`

*No documentation available.*

### `_fetch_checksum_from_url()`

*No documentation available.*

### `resolve_expected_checksum()`

Determine the expected checksum metadata for downstream enforcement.

### `to_known_hash(self)`

Return ``algorithm:value`` string suitable for pooch known_hash.

### `to_mapping(self)`

Return mapping representation for manifest and index serialization.

### `_fetch_once()`

*No documentation available.*

### `_on_retry(attempt, exc, delay)`

*No documentation available.*

## 3. Classes

### `ExpectedChecksum`

Expected checksum derived from configuration or resolver metadata.
