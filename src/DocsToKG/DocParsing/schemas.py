"""Schema version registry for DocParsing manifests and vectors.

Chunk and embedding outputs are versioned so downstream consumers can detect
incompatible changes. This module publishes canonical schema identifiers,
compatibility tables, and lookup helpers used by pipelines when emitting
manifests or validating incoming data.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple

SchemaVersion = str


class SchemaKind(str, Enum):
    """Enumerate schema families handled by DocParsing pipelines."""

    CHUNK = "chunk"
    VECTOR = "vector"


_DEFAULT_VERSIONS: Dict[SchemaKind, SchemaVersion] = {
    SchemaKind.CHUNK: "docparse/1.1.0",
    SchemaKind.VECTOR: "embeddings/1.0.0",
}

_COMPATIBLE_VERSIONS: Dict[SchemaKind, Tuple[SchemaVersion, ...]] = {
    SchemaKind.CHUNK: ("docparse/1.0.0", "docparse/1.1.0"),
    SchemaKind.VECTOR: ("embeddings/1.0.0",),
}

CHUNK_SCHEMA_VERSION: SchemaVersion = _DEFAULT_VERSIONS[SchemaKind.CHUNK]
VECTOR_SCHEMA_VERSION: SchemaVersion = _DEFAULT_VERSIONS[SchemaKind.VECTOR]
COMPATIBLE_CHUNK_VERSIONS: Tuple[SchemaVersion, ...] = _COMPATIBLE_VERSIONS[SchemaKind.CHUNK]
COMPATIBLE_VECTOR_VERSIONS: Tuple[SchemaVersion, ...] = _COMPATIBLE_VERSIONS[SchemaKind.VECTOR]

__all__ = [
    "SchemaKind",
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


def _coerce_kind(kind: SchemaKind | str) -> SchemaKind:
    """Normalise raw schema identifiers to ``SchemaKind`` enums."""

    if isinstance(kind, SchemaKind):
        return kind
    try:
        return SchemaKind(kind)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown schema kind: {kind!r}") from exc


def get_default_schema_version(kind: SchemaKind | str) -> SchemaVersion:
    """Return the canonical schema version for ``kind``."""

    return _DEFAULT_VERSIONS[_coerce_kind(kind)]


def get_compatible_versions(kind: SchemaKind | str) -> Tuple[SchemaVersion, ...]:
    """Return the tuple of compatible versions for ``kind``."""

    return _COMPATIBLE_VERSIONS[_coerce_kind(kind)]


def validate_schema_version(
    version: SchemaVersion,
    kind: SchemaKind | str,
    *,
    compatible_versions: Optional[Iterable[SchemaVersion]] = None,
    context: Optional[str] = None,
) -> SchemaVersion:
    """Validate that ``version`` is compatible for ``kind``."""

    schema_kind = _coerce_kind(kind)
    compatible = tuple(compatible_versions or get_compatible_versions(schema_kind))
    if version not in compatible:
        suffix = f" ({context})" if context else ""
        raise ValueError(
            f"Unsupported {schema_kind.value} schema version {version!r}{suffix}; "
            f"supported versions: {', '.join(compatible)}"
        )
    return version


def ensure_chunk_schema(
    rec: dict,
    *,
    default_version: Optional[SchemaVersion] = None,
    context: Optional[str] = None,
) -> dict:
    """Ensure ``rec`` declares a compatible chunk schema version.

    When ``schema_version`` is missing it is populated with ``default_version``
    (or :data:`CHUNK_SCHEMA_VERSION` when unspecified). When present it is
    validated against the chunk compatibility set.
    """

    version = rec.get("schema_version")
    if not version:
        rec["schema_version"] = default_version or CHUNK_SCHEMA_VERSION
        return rec

    coerced = str(version)
    validate_schema_version(
        coerced,
        SchemaKind.CHUNK,
        context=context,
    )
    rec["schema_version"] = coerced
    return rec


def validate_vector_row(
    row: Dict[str, Any],
    *,
    expected_dimension: Optional[int] = None,
) -> Any:
    """Validate a vector manifest row and optionally enforce embedding dimension."""

    from DocsToKG.DocParsing.formats import (
        _pydantic_validate_vector_row,
    )

    vector_row = _pydantic_validate_vector_row(row)
    qwen_vector = vector_row.Qwen3_4B
    actual_dim = qwen_vector.dimension or len(qwen_vector.vector)
    if expected_dimension is not None and actual_dim != expected_dimension:
        raise ValueError(
            f"Qwen vector dimension {actual_dim} does not match expected {expected_dimension}"
        )
    return vector_row
