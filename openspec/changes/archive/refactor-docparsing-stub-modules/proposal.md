## Why
- `tests/docparsing/stubs.py` dynamically injects stub modules using `ModuleType`, which keeps pytest fixtures easy to use but produces hundreds of `attr-defined` errors when MyPy scans the test suite.
- The dynamic modules hide the fake APIs from static analysis, so MyPy cannot validate that integration tests are wiring dependencies correctly. The hook currently fails, blocking the stricter typing posture the team wants.
- Converting the stubs into importable Python modules keeps the runtime behaviour identical while making the contract explicit and type-checkable.

## What Changes
- Replace the runtime `ModuleType` injection in `tests/docparsing/stubs.py` with a package of lightweight fake dependencies under `tests/docparsing/fake_deps/` that mirrors every namespace currently fabricated (sentence_transformers, vllm, tqdm, pydantic, transformers, docling_core and its nested subpackages).
- Relocate each stub class/function (e.g. `_StubSparseEncoder`, `_StubDoclingDocument`, `_StubChunkingDocSerializer`, serializer helpers) into the module that matches its production counterpart so logic remains identical but is now statically visible.
- Update `dependency_stubs()` so it prepends the fake package path to `sys.path`, imports the modules via `importlib`, and registers them in `sys.modules` when the real dependency is absent—preserving the existing `force=True` behaviour without overwriting genuine installations.
- Capture a dependency-to-module mapping in `tests/docparsing/fake_deps/MIGRATION_NOTES.md` and add README guidance describing how contributors should extend the fake package going forward.
- Ensure MyPy scans the fake package cleanly by adding `__all__` exports and type annotations where necessary.

## Impact
- MyPy will see concrete module definitions, eliminating the current `Module has no attribute` errors coming from the test stubs and unblocking stricter type checking.
- Tests continue to run without requiring the heavy external dependencies (vLLM, docling, etc.) because the helper still injects the fakes when optional dependencies are missing.
- Contributors gain a clearly documented, version-controlled location for extending fake dependencies, reducing the risk of reintroducing dynamic `ModuleType` stubs.

## Open Questions
- Should the fake modules live directly under `tests/docparsing/` or inside a dedicated helper package such as `tests/helpers/fake_deps/`? (Default plan: keep them local to docparsing to limit scope.)
- Do we need a top-level package name (e.g. `DocsToKG_fake`) to avoid conflicts with real modules? (Default plan: provide a dedicated package namespace that mirrors the real module names only within `tests.docparsing.fake_deps`.)

## Acceptance Criteria
- MyPy no longer reports `attr-defined` errors for the docparsing stubs when run via pre-commit.
- Tests that currently rely on `dependency_stubs()` continue to pass without requiring external dependencies.
- Documentation clarifies how to add or modify fake dependency behaviour.
