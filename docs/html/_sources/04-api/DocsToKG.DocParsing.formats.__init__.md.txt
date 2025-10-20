# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.DocParsing.formats``.

## 1. Overview

``DocsToKG.DocParsing.formats`` gathers the canonical DocParsing schemas,
validation helpers, and Docling serializer utilities into a single import
surface. Pydantic v2 is a hard requirement: the module instantiates real models
at import time and surfaces actionable failures if the dependency is missing.

Key features:

- Strict Pydantic models for chunk JSONL rows and embedding vector rows.
- Co-located schema-version metadata (`SchemaKind`, `SchemaVersion`,
  `validate_schema_version`, `ensure_chunk_schema`).
- Convenience validators (`validate_chunk_row`, `validate_vector_row`) that
  wrap model construction and dimension checks.
- Markdown-aware Docling serializer providers for picture/table enrichment.

Dependencies:

- ``pydantic>=2,<3`` for model parsing and validation.
- ``docling_core`` for serializer base classes and picture/table helpers.

## 2. Schema Metadata

- ``SchemaKind``: enumerates ``"chunk"`` and ``"vector"`` schema families.
- ``SchemaVersion``: alias for the version string literals used in manifests.
- ``CHUNK_SCHEMA_VERSION`` / ``VECTOR_SCHEMA_VERSION``: canonical versions
  emitted by DocParsing stages.
- ``COMPATIBLE_CHUNK_VERSIONS`` / ``COMPATIBLE_VECTOR_VERSIONS``: accepted
  historical versions.
- ``get_default_schema_version(kind)``: returns the canonical version for the
  requested schema kind.
- ``get_compatible_versions(kind)``: lists compatible versions for the schema
  kind.
- ``validate_schema_version(version, kind, *, compatible_versions=None, context=None)``:
  raises ``ValueError`` if ``version`` is incompatible.
- ``ensure_chunk_schema(rec, *, default_version=None, context=None)``: ensures a
  chunk record carries a compatible schema version, defaulting when absent.

## 3. Models

- ``ProvenanceMetadata``: provenance data attached to individual chunks.
- ``ChunkRow``: chunk JSONL schema with strict type validation and schema
  version enforcement.
- ``BM25Vector`` / ``SPLADEVector`` / ``DenseVector``: component vector types
  used inside the vector row schema.
- ``VectorRow``: embedding vector schema exposing BM25, SPLADE, and Qwen dense
  payloads plus provenance metadata.

## 4. Validation Helpers

- ``validate_chunk_row(row: dict) -> ChunkRow``: parses and validates chunk
  records, raising ``ValueError`` on failure.
- ``_pydantic_validate_vector_row(row: dict) -> VectorRow``: internal helper
  that constructs ``VectorRow`` instances directly.
- ``validate_vector_row(row: dict, *, expected_dimension: int | None = None) -> VectorRow``:
  wraps ``_pydantic_validate_vector_row`` and enforces the dense vector
  dimension when provided.

## 5. Serializer Utilities

- ``get_docling_version()``: returns the installed Docling version or
  ``"unknown"``.
- ``CaptionPlusAnnotationPictureSerializer``: enriches Docling picture items
  with captions, annotations, and confidences.
- ``RichSerializerProvider``: constructs Markdown-aware serializer pipelines
  that feed the chunking stage.
