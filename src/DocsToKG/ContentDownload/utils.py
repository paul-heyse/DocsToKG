"""Shared utility functions for the ContentDownload module."""

from __future__ import annotations

import re
from typing import List, Optional


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI by stripping https://doi.org/ prefix and whitespace."""

    if not doi:
        return None
    doi = doi.strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:]
    return doi.strip() or None


def normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
    """Normalize PMCID ensuring a canonical PMC prefix."""

    if not pmcid:
        return None
    pmcid = pmcid.strip()
    match = re.search(r"(?:PMC)?(\d+)", pmcid, flags=re.I)
    if match:
        return f"PMC{match.group(1)}"
    return None


def strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    """Strip a case-insensitive prefix from ``value`` if present."""

    if not value:
        return None
    value = value.strip()
    if value.lower().startswith(prefix.lower()):
        return value[len(prefix) :]
    return value


def dedupe(items: List[str]) -> List[str]:
    """Remove duplicates while preserving the first occurrence order."""

    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result

