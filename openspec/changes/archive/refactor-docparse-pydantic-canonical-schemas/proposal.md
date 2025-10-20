# Refactor DocParsing Schemas to Require Pydantic v2

## Why
The DocParsing runtime still treats Pydantic as an optional dependency even though `pyproject.toml` already lists `pydantic>=2,<3` in the core requirements. `src/DocsToKG/DocParsing/formats/__init__.py:149-314` builds stub classes when the import fails, `validate_chunk_row` and `_pydantic_validate_vector_row` branch on `PYDANTIC_AVAILABLE`, and the module exposes `_missing_pydantic_message()` that callers surface to users. At the same time `src/DocsToKG/DocParsing/schemas.py:1-140` duplicates the same models/constants, delegating back into `formats` for validation. The split schema surface leaks into `embedding/runtime.py:490-506`, tests ship fake `pydantic` packages (`tests/docparsing/fake_deps/pydantic/__init__.py`) and README guidance still describes missing-Pydantic fallbacks. This duplication risks drift, confuses downstream importers, and keeps dead code paths alive even though production has required Pydantic v2 for months.

## What Changes
- Remove the optional-dependency scaffold in `DocParsing.formats`: delete the try/except import stub, `_missing_pydantic_message()`, and the per-call `PYDANTIC_AVAILABLE` guards so validators instantiate Pydantic models directly.
- Collapse schema definitions and constants into `DocParsing.formats`: move `CHUNK_SCHEMA_VERSION`, `VECTOR_SCHEMA_VERSION`, compatibility tables, and helper functions from `DocParsing.schemas` into the canonical module, eliminating the bidirectional dependency.
- Replace `DocParsing.schemas` with a compatibility shim that re-exports the canonical symbols, emits a `DeprecationWarning`, and documents the new import path.
- Update all internal imports (embedding runtime, IO helpers, tests, docs) to use `DocParsing.formats` exclusively and delete fake `pydantic` stubs/tests that simulated missing dependency scenarios.
- Refresh API docs, runbooks, and README guidance to state that Pydantic v2 is mandatory for DocParsing and to highlight the consolidation.

## Impact
- **Affected specs:** docparsing.
- **Affected code:** `src/DocsToKG/DocParsing/formats/__init__.py`, `src/DocsToKG/DocParsing/schemas.py`, `src/DocsToKG/DocParsing/embedding/runtime.py`, `src/DocsToKG/DocParsing/io/__init__.py` (schema helpers), test scaffolding under `tests/docparsing/**`, documentation under `docs/04-api/**` and `src/DocsToKG/DocParsing/README.md`.
- **Compatibility:** maintain public names (`ChunkRow`, `VectorRow`, validators) and document the temporary shim lifecycle; no new runtime dependencies beyond the already-required Pydantic v2.
