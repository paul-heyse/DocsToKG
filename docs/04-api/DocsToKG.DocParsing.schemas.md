# 1. Module: schemas

This reference documents the DocsToKG module ``DocsToKG.DocParsing.schemas``.

## 1. Overview

Schema constants and validation helpers shared across DocParsing stages.

## 2. Functions

### `_coerce_kind(kind)`

Normalise raw schema identifiers to ``SchemaKind`` enums.

### `get_default_schema_version(kind)`

Return the canonical schema version for ``kind``.

### `get_compatible_versions(kind)`

Return the tuple of compatible versions for ``kind``.

### `validate_schema_version(version, kind)`

Validate that ``version`` is compatible for ``kind``.

### `ensure_chunk_schema(rec)`

Ensure ``rec`` declares a compatible chunk schema version.

When ``schema_version`` is missing it is populated with ``default_version``
(or :data:`CHUNK_SCHEMA_VERSION` when unspecified). When present it is
validated against the chunk compatibility set.

### `validate_vector_row(row)`

Validate a vector manifest row and optionally enforce embedding dimension.

## 3. Classes

### `SchemaKind`

Enumerate schema families handled by DocParsing pipelines.
