"""
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
"""

from __future__ import annotations

import re
from typing import List, Optional


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI identifiers by stripping prefixes and whitespace.

    Args:
        doi: Raw DOI string or URL provided by upstream metadata.

    Returns:
        Canonical DOI without protocol prefixes, or None when input is empty.
    """

    if not doi:
        return None
    doi = doi.strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:]
    return doi.strip() or None


def normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
    """Normalize PMCID values ensuring a canonical PMC prefix.

    Args:
        pmcid: PMCID string that may contain extraneous characters.

    Returns:
        Normalized PMCID including the `PMC` prefix, or None if parsing fails.
    """

    if not pmcid:
        return None
    pmcid = pmcid.strip()
    match = re.search(r"(?:PMC)?(\d+)", pmcid, flags=re.I)
    if match:
        return f"PMC{match.group(1)}"
    return None


def strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    """Strip a case-insensitive prefix from a string when present.

    Args:
        value: String that might contain the prefix.
        prefix: Prefix to remove from the value.

    Returns:
        String without the prefix, or None if the value is empty.
    """

    if not value:
        return None
    value = value.strip()
    if value.lower().startswith(prefix.lower()):
        return value[len(prefix) :]
    return value


def dedupe(items: List[str]) -> List[str]:
    """Remove duplicates while preserving the first occurrence order.

    Args:
        items: Sequence of string values that may contain duplicates.

    Returns:
        New list with duplicates removed while keeping original ordering.
    """

    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result
