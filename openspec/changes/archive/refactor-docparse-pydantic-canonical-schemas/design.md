## Context
DocParsing currently maintains two overlapping schema modules. `src/DocsToKG/DocParsing/formats/__init__.py:149-327` defines Pydantic models alongside serializer helpers but wraps the import in a try/except that fabricates stub classes and the `_missing_pydantic_message()` helper. The validators (`validate_chunk_row`, `_pydantic_validate_vector_row`) branch on `PYDANTIC_AVAILABLE` and raise a runtime error when the dependency is missing. `src/DocsToKG/DocParsing/schemas.py:1-140` duplicates schema constants, exposes `validate_schema_version`, and forwards `validate_vector_row` back into `formats`, creating a circular dependency that `embedding/runtime.py:490-506` navigates via dual imports. Test scaffolding (`tests/docparsing/fake_deps/pydantic/__init__.py`, `tests/docparsing/stubs.py:31-64`) keeps an ersatz Pydantic implementation alive even though production already bundles `pydantic>=2,<3` in `pyproject.toml:1-320`. Docs and API references (`docs/04-api/DocsToKG.DocParsing.schemas.md`, `src/DocsToKG/DocParsing/README.md:273-283`) still mention the optional dependency behavior.

## Goals / Non-Goals
- Goals: hard-require Pydantic v2 at runtime, consolidate schema definitions/constants into `DocParsing.formats`, provide a temporary shim module for backwards compatibility, and update documentation/tests to match the enforced dependency.
- Non-Goals: change schema field semantics, revise CLI behavior beyond import-path consolidation, remove the shim immediately (it will live for one release), or alter other optional dependency stubs (e.g., Docling, vLLM).

## Decisions
- Canonical module: `DocsToKG.DocParsing.formats` becomes the single source of truth for models, schema versions, and validators; `DocParsing.schemas` reduces to a warning-emitting re-export.
- Dependency handling: drop `PYDANTIC_AVAILABLE` guards and stubs; rely on import-time failure with a targeted `RuntimeError` narrating installation remediation.
- API stability: preserve public symbol names and `__all__` exports; re-exports from embedding/runtime stay intact, and the shim guarantees old imports continue to work during the transition.
- Test/dependency hygiene: delete fake Pydantic packages and optional-path tests; convert any remaining fixtures to use the real dependency.
- Shim lifetime: keep `DocsToKG.DocParsing.schemas` as a warning-emitting shim through release `0.3.0`, after which the module will be removed; the deprecation banner and release notes will advertise this deadline.

## Risks / Trade-offs
- Risk: lingering imports may keep pulling from `DocParsing.schemas`, bypassing consolidation. Mitigation: `rg`-based sweep plus a CI lint that fails when the module is imported outside the shim.
- Risk: removing stubs breaks environments that previously ran without Pydantic. Mitigation: the project already lists Pydantic as mandatory; we will document the dependency prominently and ensure import-time errors are explicit.
- Risk: API doc generation may need adjustments to new module layout. Mitigation: regenerate Sphinx docs and update references in `docs/04-api` to avoid stale links.

## Migration Plan
- Step 1: remove optional import scaffolding in `formats` and migrate schema constants/helpers from `schemas`.
- Step 2: implement the shim module with `DeprecationWarning` and adjust all internal imports/tests to target `formats`.
- Step 3: clean up fake Pydantic stubs, update docs/readme/API references, and add lint coverage to prevent regressions.

## Open Questions
- None. The `DocParsing.schemas` shim SHALL remain until DocsToKG `0.3.0`, and the module docstring plus release notes will record the removal target so downstream owners can plan migrations.
