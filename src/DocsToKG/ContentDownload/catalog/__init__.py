# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.__init__",
#   "purpose": "Artifact Catalog and Storage Index for ContentDownload.",
#   "sections": []
# }
# === /NAVMAP ===

"""
Artifact Catalog and Storage Index for ContentDownload.

Provides persistent storage of download metadata, SHA-256 hashes, and content-addressed
paths for deduplication, verification, and garbage collection.

This module implements PR #9: Artifact Catalog & Storage Index, enabling:
  - Persistent SQLite/Postgres catalog of all successfully stored artifacts
  - SHA-256 hashing for content-addressed storage (CAS) and deduplication
  - Content deduplication via hardlinks or copies (POSIX/Windows compatible)
  - Garbage collection and retention policies
  - Verification of stored artifacts against recorded hashes
  - Migration helpers for backfilling from existing manifests
"""

from __future__ import annotations

from DocsToKG.ContentDownload.catalog.models import DocumentRecord
from DocsToKG.ContentDownload.catalog.store import CatalogStore, SQLiteCatalog

__version__ = "1.0.0"
__all__ = [
    "DocumentRecord",
    "CatalogStore",
    "SQLiteCatalog",
]
