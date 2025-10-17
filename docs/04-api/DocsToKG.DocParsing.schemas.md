# 1. Module: schemas

This reference documents the DocsToKG module ``DocsToKG.DocParsing.schemas``.

## 1. Overview

Schema constants and validation helpers shared across DocParsing stages.

## 2. Functions

### `_coerce_kind(kind)`

*No documentation available.*

### `get_default_schema_version(kind)`

Return the canonical schema version for ``kind``.

### `get_compatible_versions(kind)`

Return the tuple of compatible versions for ``kind``.

### `validate_schema_version(version, kind)`

Validate that ``version`` is compatible for ``kind``.

### `validate_vector_row(row)`

Validate a vector manifest row and optionally enforce embedding dimension.

## 3. Classes

### `SchemaKind`

Enumerate schema families handled by DocParsing pipelines.
