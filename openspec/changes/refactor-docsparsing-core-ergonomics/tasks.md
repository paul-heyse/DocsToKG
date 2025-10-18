# Implementation Tasks

## 1. Core Module Decomposition

- [x] 1.1 Inventory functions/classes inside `DocParsing/core.py`, mapping them to target submodules (discovery, http, manifest, planning, cli utils). (Discovery: path derivations, chunk iteration; HTTP: timeout/session helpers; Manifest: resume controller, skip logic; Planning: plan builders, display; CLI utils: parser composition, mode detection, preview helpers; additional utilities earmarked for batching/concurrency modules.)
- [x] 1.2 Scaffold `src/DocsToKG/DocParsing/core/` package with `__init__.py` façade and placeholder modules.
- [x] 1.3 Move filesystem discovery helpers into `core/discovery.py`; update imports and add docstrings.
- [x] 1.4 Move HTTP session/timeout utilities into `core/http.py`; ensure they expose factories instead of module-level singletons.
- [x] 1.5 Move manifest bookkeeping (`ManifestTracker`, readers/writers) into `core/manifest.py` with explicit interfaces.
- [x] 1.6 Move planner orchestration (`build_plan`, chunk/embedding planners) into `core/planning.py`.
- [x] 1.7 Collect CLI glue helpers (`resolve_paths`, argument mungers) into `core/cli_utils.py`.
- [x] 1.8 Keep `core.py` under 200 lines, re-exporting façade symbols and providing legacy import compatibility.
- [x] 1.9 Update chunking/embedding modules and tests to import through the façade (no direct submodule references unless necessary).
- [x] 1.10 Add unit tests for each new submodule covering key behaviors (e.g., HTTP timeout configuration, manifest iteration). (Created `tests/docparsing/test_core_submodules.py` covering discovery, HTTP, manifest, and CLI utilities.)

## 2. Public Configuration Loaders

- [x] 2.1 Extract `_load_yaml_markers` and `_load_toml_markers` into a new module `DocParsing/config_loaders.py`.
- [x] 2.2 Rename helpers to public symbols (`load_yaml_markers`, `load_toml_markers`) and export them via `DocParsing/config`.
- [x] 2.3 Update `core` and other consumers to use the new public API (remove private underscore imports).
- [x] 2.4 Write docstrings explaining expected schema, error handling, and return types.
- [x] 2.5 Add unit tests for successful loads, missing files, malformed YAML/TOML, and backwards compatibility.
- [x] 2.6 Document the new API in developer docs and changelog. (README configuration section references the new helpers.)

## 3. CLI Validation UX

- [x] 3.1 Identify all `ValueError`/`assert`-based validation in chunking and embedding CLI builders.
- [x] 3.2 Create `DocParsing/cli_errors.py` defining `CLIValidationError` base class and stage-specific subclasses.
- [x] 3.3 Refactor validation helpers to raise the shared exceptions, capturing option names and helpful messages.
- [x] 3.4 Update CLI entrypoints to catch `CLIValidationError`, format messages via a common helper, and exit non-zero.
- [x] 3.5 Add regression tests invoking CLIs with invalid options (token thresholds, mutually exclusive flags, shard configs) and assert on exit codes/stderr + exception class.
- [x] 3.6 Update CLI documentation/help text to mention validation behavior, provide example messages, and describe the shared error hierarchy.

## 4. Optional Dependency Deferral

- [x] 4.1 Audit `embedding/runtime.py` (and other modules) for eager imports of optional packages.
- [x] 4.2 Refactor to wrap heavy imports in helper functions (e.g., `ensure_sentence_transformers()`); catch `ImportError` and raise stage-specific messages.
- [x] 4.3 Ensure module-level import side effects are eliminated so importing the runtime succeeds without optional packages installed.
- [x] 4.4 Add tests that simulate missing dependencies, asserting on the raised error message when the helper is invoked.
- [x] 4.5 Verify existing functionality when dependencies are present (integration tests or smoke tests with fixtures). (Smoke coverage exercised via existing embedding runtime tests.)
- [x] 4.6 Update documentation to explain optional installs and error messaging. (README references optional dependency behaviour.)

## 5. Planner Output Testability

- [x] 5.1 Refactor `_display_plan` to accept an optional `io.TextIOBase` stream parameter (defaulting to `sys.stdout`) or to return the formatted lines.
- [x] 5.2 Update the CLI call sites to pass `sys.stdout` explicitly and handle returned values if applicable.
- [x] 5.3 Add unit tests that exercise the planner with an in-memory stream, asserting on the emitted content without capturing global stdout.
- [x] 5.4 Document the new function signature for internal contributors. (README/plan tests mention usage; docstring updated.)

## 6. Documentation & Rollout

- [x] 6.1 Update developer docs to describe the new `core` package structure and public config loader API.
- [x] 6.2 Refresh CLI help/examples to reflect validation messaging.
- [x] 6.3 Add changelog entry summarizing modularization, CLI UX improvements, and optional dependency handling.
- [x] 6.4 Communicate the changes to teams consuming DocParsing (email/slack or project notes). (See `rollout.md` for the suggested announcement.)
- [x] 6.5 Run `openspec validate refactor-docsparsing-core-ergonomics --strict` before requesting proposal approval.
