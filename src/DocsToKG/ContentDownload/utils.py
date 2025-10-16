# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.utils",
#   "purpose": "Utility helpers for DocsToKG content download",
#   "sections": [
#     {
#       "id": "normalize-doi",
#       "name": "normalize_doi",
#       "anchor": "function-normalize-doi",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-pmcid",
#       "name": "normalize_pmcid",
#       "anchor": "function-normalize-pmcid",
#       "kind": "function"
#     },
#     {
#       "id": "strip-prefix",
#       "name": "strip_prefix",
#       "anchor": "function-strip-prefix",
#       "kind": "function"
#     },
#     {
#       "id": "dedupe",
#       "name": "dedupe",
#       "anchor": "function-dedupe",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Content Download Utility Helpers

This module provides small string and identifier normalisation helpers shared
across resolver implementations. The utilities ensure consistent handling of
scholarly identifiers such as DOIs, PMCIDs, and arXiv IDs while also providing
lightweight string manipulation helpers used during manifest generation.

Key Features:
- Normalisation of DOI, PMCID, and arXiv identifiers from heterogeneous sources
  including common URL and ``doi:`` prefixes.
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


# --- Globals ---

__all__ = ("normalize_doi", "normalize_pmcid", "strip_prefix", "dedupe")


# --- Public Functions ---

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI identifiers by stripping common prefixes and whitespace.

    Args:
        doi: Raw DOI string or URL provided by upstream metadata.

    Returns:
        Canonical DOI without protocol prefixes, or None when input is empty.

    Supported prefixes:

    - ``https://doi.org/``
    - ``http://doi.org/``
    - ``https://dx.doi.org/``
    - ``http://dx.doi.org/``
    - ``doi:``
    """

    if not doi:
        return None
    value = doi.strip()
    lower = value.lower()
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            value = value[len(prefix) :]
            lower = value.lower()
            break
    if lower.startswith("doi:"):
        value = value[len("doi:") :]
    return value.strip() or None


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
