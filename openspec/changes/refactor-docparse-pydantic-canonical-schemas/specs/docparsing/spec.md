# DocParsing Specification Deltas

## ADDED Requirements

### Requirement: Pydantic v2 Mandatory for DocParsing Schemas
DocParsing schema helpers SHALL require Pydantic v2 at import time and execute validation logic without optional fallbacks.

#### Scenario: Import Fails Fast When Pydantic Is Missing
- **GIVEN** `pydantic` is not installed in the environment
- **WHEN** `DocsToKG.DocParsing.formats` (or any submodule with schema helpers) is imported
- **THEN** the import SHALL raise a `RuntimeError` that identifies `pydantic>=2,<3` as required and instructs the operator to install it
- **AND** no stub classes or `PYDANTIC_AVAILABLE` flags SHALL be exposed in the module namespace

#### Scenario: Validators Instantiate Canonical Models
- **GIVEN** a valid chunk or vector payload
- **WHEN** `validate_chunk_row` or `validate_vector_row` is called
- **THEN** the helper SHALL instantiate the Pydantic models directly, returning `ChunkRow` / `VectorRow` instances with populated defaults
- **AND** the helpers SHALL no longer branch on dependency availability or delegate to stub implementations

### Requirement: Canonical Schema Module Export Surface
DocParsing SHALL expose a single canonical schema module that provides models, schema-version helpers, and validators.

#### Scenario: Internal Imports Target `DocParsing.formats`
- **GIVEN** embedding runtime, IO utilities, and other DocParsing modules import schema helpers
- **WHEN** those modules resolve the import paths
- **THEN** they SHALL import `ChunkRow`, `VectorRow`, `validate_schema_version`, and related constants from `DocsToKG.DocParsing.formats`
- **AND** no internal module SHALL import schema definitions from `DocsToKG.DocParsing.schemas` after the consolidation

#### Scenario: Schema Version Helpers Co-reside With Models
- **GIVEN** callers invoke `validate_schema_version` or `ensure_chunk_schema`
- **WHEN** the functions execute
- **THEN** the implementation SHALL live in `DocsToKG.DocParsing.formats` alongside the Pydantic models
- **AND** the functions SHALL operate on the same compatibility tables that drive `ChunkRow.schema_version` and `VectorRow.schema_version` defaults

### Requirement: Deprecation Shim for `DocsToKG.DocParsing.schemas`
DocParsing SHALL provide a temporary shim module that re-exports the canonical schema surface while guiding downstream callers to migrate.

#### Scenario: Shim Emits Deprecation Warning
- **GIVEN** legacy code imports `DocsToKG.DocParsing.schemas`
- **WHEN** the module is imported
- **THEN** it SHALL immediately emit a `DeprecationWarning` indicating that `DocsToKG.DocParsing.formats` is the supported path
- **AND** the shim SHALL re-export the same symbols (`ChunkRow`, `VectorRow`, validators, schema version helpers) without defining new implementations

#### Scenario: Documentation Signals Shim Timeline
- **GIVEN** developers read the DocParsing API docs or README
- **WHEN** they encounter references to schema helpers
- **THEN** the documentation SHALL note that `DocsToKG.DocParsing.schemas` is deprecated, enumerate the migration deadline (removal targeted for DocsToKG `0.3.0`), and direct readers to use the canonical module
- **AND** API reference pages SHALL list the schema classes under `DocsToKG.DocParsing.formats`
