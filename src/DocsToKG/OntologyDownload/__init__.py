"""Ontology downloader public API."""

from .core import FetchResult, FetchSpec, fetch_all, fetch_one

__all__ = [
    "FetchSpec",
    "FetchResult",
    "fetch_one",
    "fetch_all",
]
