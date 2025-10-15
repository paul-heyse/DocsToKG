"""
Ontology Downloader Public API

Expose the primary fetch utilities used by external callers to download,
validate, and document ontology resources as part of the DocsToKG pipeline.
The entry points wrap resolver planning, HEAD-validated downloads, canonical
Turtle hashing, and manifest generation into a compact public API.
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
