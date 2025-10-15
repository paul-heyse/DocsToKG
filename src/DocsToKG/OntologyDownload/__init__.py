"""Ontology downloader public API.

Expose the primary fetch utilities used by external callers to plan resolver
fallback chains, download ontologies with hardened validation, stream
normalization, and emit schema-compliant manifests with deterministic
fingerprints.
"""

from .core import FetchResult, FetchSpec, PlannedFetch, fetch_all, fetch_one, plan_all, plan_one

__all__ = [
    "FetchSpec",
    "FetchResult",
    "PlannedFetch",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
]
