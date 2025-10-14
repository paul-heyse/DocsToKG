"""
Ontology Downloader Public API

Expose the primary fetch utilities used by external callers to download,
validate, and document ontology resources as part of the DocsToKG pipeline.
"""

from .core import FetchResult, FetchSpec, fetch_all, fetch_one

__all__ = [
    "FetchSpec",
    "FetchResult",
    "fetch_one",
    "fetch_all",
]
