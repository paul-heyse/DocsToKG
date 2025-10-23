# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.schemas",
#   "purpose": "Deprecated shim for DocParsing schema helpers.",
#   "sections": []
# }
# === /NAVMAP ===

"""Deprecated shim for DocParsing schema helpers.

`DocsToKG.DocParsing.schemas` historically hosted schema metadata and validators,
but the canonical surface now lives in :mod:`DocsToKG.DocParsing.formats`.
This module re-exports the public API and raises a deprecation warning so
callers can migrate before the shim is removed in DocsToKG 0.3.0.
"""

from __future__ import annotations

import warnings

from DocsToKG.DocParsing.formats import (
    CHUNK_SCHEMA_VERSION,
    COMPATIBLE_CHUNK_VERSIONS,
    COMPATIBLE_VECTOR_VERSIONS,
    VECTOR_SCHEMA_VERSION,
    SchemaKind,
    SchemaVersion,
    ensure_chunk_schema,
    get_compatible_versions,
    get_default_schema_version,
    validate_schema_version,
    validate_vector_row,
)

__all__ = [
    "SchemaKind",
    "SchemaVersion",
    "CHUNK_SCHEMA_VERSION",
    "VECTOR_SCHEMA_VERSION",
    "COMPATIBLE_CHUNK_VERSIONS",
    "COMPATIBLE_VECTOR_VERSIONS",
    "get_default_schema_version",
    "get_compatible_versions",
    "validate_schema_version",
    "ensure_chunk_schema",
    "validate_vector_row",
]

warnings.warn(
    (
        "DocsToKG.DocParsing.schemas is deprecated and will be removed in "
        "DocsToKG 0.3.0; import from DocsToKG.DocParsing.formats instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)
