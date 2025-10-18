# DocParsing Fake Dependencies

This package mirrors the module layout that `tests.docparsing.stubs.dependency_stubs()`
previously manufactured with `ModuleType`. Each submodule exposes the minimum surface
area required by the test-suite while remaining importable and type-checkable.

When a test invokes `dependency_stubs()` we insert this directory on `sys.path`
and register the fake modules under their real production names (for example
`sentence_transformers`, `vllm`, or `docling_core.transforms.serializer.base`).

## Extending the fakes

1. Identify the real module or attribute the test (or production code under test)
   expects to import.
2. Locate the corresponding mirror module inside this package. If it does not
   exist, add one that matches the same dotted path.
3. Implement the required classes or functions with deterministic behaviour suitable
   for tests. Keep dependencies minimal—prefer `typing` and `types.SimpleNamespace`
   over heavy imports.
4. Update `MIGRATION_NOTES.md` with the new module/attribute pairings so future
   contributors understand the surface area we cover.
5. Run `pre-commit run mypy --files tests/docparsing/stubs.py tests/docparsing/fake_deps`
   to confirm the fake remains type-checkable.
