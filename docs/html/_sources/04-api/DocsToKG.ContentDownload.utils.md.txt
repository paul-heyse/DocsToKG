# 1. Module: utils

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.utils``.

Content Download Utility Helpers

This module provides small string and identifier normalisation helpers shared
across resolver implementations. The utilities ensure consistent handling of
scholarly identifiers such as DOIs, PMCIDs, and arXiv IDs while also providing
lightweight string manipulation helpers used during manifest generation.

Key Features:
- Normalisation of DOI, PMCID, and arXiv identifiers from heterogeneous sources.
- Prefix stripping for resolver-specific metadata cleaning.
- Duplicate removal while preserving original ordering.

Usage:
    from DocsToKG.ContentDownload import utils

    doi = utils.normalize_doi("https://doi.org/10.1234/example")
    pmcid = utils.normalize_pmcid("PMC12345")

## 1. Functions

### `normalize_doi(doi)`

Normalize DOI identifiers by stripping prefixes and whitespace.

Args:
doi: Raw DOI string or URL provided by upstream metadata.

Returns:
Canonical DOI without protocol prefixes, or None when input is empty.

### `normalize_pmcid(pmcid)`

Normalize PMCID values ensuring a canonical PMC prefix.

Args:
pmcid: PMCID string that may contain extraneous characters.

Returns:
Normalized PMCID including the `PMC` prefix, or None if parsing fails.

### `strip_prefix(value, prefix)`

Strip a case-insensitive prefix from a string when present.

Args:
value: String that might contain the prefix.
prefix: Prefix to remove from the value.

Returns:
String without the prefix, or None if the value is empty.

### `dedupe(items)`

Remove duplicates while preserving the first occurrence order.

Args:
items: Sequence of string values that may contain duplicates.

Returns:
New list with duplicates removed while keeping original ordering.
