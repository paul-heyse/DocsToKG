# Refactor DocParsing Core for Modularity and Operator UX

## Why

A focused review of `src/DocsToKG/DocParsing` uncovered several structural and ergonomic issues:

1. **Monolithic `core.py` (≈2,000 LOC)** mixes filesystem discovery, HTTP helpers, manifest utilities, planner orchestration, and CLI glue inside a single module. This violates separation of concerns, inflates import costs, and obscures ownership boundaries.
2. **Private config helpers exported implicitly**: `_load_yaml_markers` and `_load_toml_markers` are imported directly from `config.py` despite the underscore convention, making downstream code brittle when config internals change.
3. **CLI validation raises raw `ValueError`**: chunking and embedding CLIs surface invalid thresholds and mutually exclusive options as Python tracebacks, degrading usability and complicating automation.
4. **Optional heavy dependencies imported eagerly**: `embedding/runtime.py` imports `sentence_transformers` and `vllm` at module import time, triggering slow startup and noisy warnings for installations that only use lighter DocParsing stages.
5. **Planner output hard-wired to stdout**: `_display_plan` prints directly, making it hard to test or reuse the planner output in other surfaces.

Addressing these issues will make DocParsing easier to maintain, reduce accidental breakage when tweaking configuration logic, and deliver a friendlier CLI experience.

## What Changes

### 1. Decompose `core.py` into focused modules

- Create submodules under `DocParsing/core/` (e.g., `discovery.py`, `http.py`, `manifest.py`, `planning.py`, `cli_utils.py`).
- Move the corresponding classes/functions out of `core.py`, keeping a thin façade (`core/__init__.py`) that re-exports the stable surface (`discover_input_paths`, `HttpClientFactory`, `ManifestTracker`, planner utilities).
- Add docstrings and typing to clarify module responsibilities and maintain backwards compatibility for existing imports.
- Update call sites (chunking, embedding, tests) to import from the new submodules via the façade.

### 2. Promote shared config loaders to public API

- Move `_load_yaml_markers` and `_load_toml_markers` into `config_loaders.py` (or rename them without underscores) and export them through `DocsToKG.DocParsing.config`.
- Document their behavior, expected schema, and failure modes.
- Update `core` and any other consumers to rely on the public helpers; remove remaining private imports.
- Add targeted tests covering YAML/TOML loading, error handling for malformed markers, and backwards compatibility.

### 3. Normalize CLI validation error handling

- Introduce a shared `CLIValidationError` exception hierarchy under `DocParsing/cli_errors.py` that encapsulates option name, invalid value, and human-readable guidance.
- Replace raw `ValueError` raises in chunking/embedding CLI builders with the new exceptions (or `parser.error` backed by the shared helper) so every stage reports errors consistently.
- Ensure entrypoints exit with non-zero status codes and print concise, user-friendly messages without Python tracebacks by catching `CLIValidationError` and formatting output via a common helper.
- Add regression tests that invoke the CLIs with invalid options and assert on exit code + stderr output, including verifying the exception hierarchy is exercised.
- Update documentation (including `DocsToKG/DocParsing/README.md`) to describe validation behavior, guidance for automation scripts, and the new shared error types.

### 4. Defer optional heavy imports until needed

- Refactor `embedding/runtime.py` (and any other modules) so imports for `sentence_transformers`, `vllm`, and similar optional dependencies occur inside the functions/methods that require them.
- Provide helper functions (e.g., `load_sentence_transformer()`) that catch `ImportError` and raise a stage-specific, actionable error message.
- Add unit tests to ensure the embedding runtime can be imported without the optional dependencies installed and that the helpers raise informative errors when invoked without the packages present.

### 5. Make plan display testable and reusable

- Refactor `_display_plan` to accept an optional text stream or return a structured representation (list of lines/strings) before writing to the stream.
- Update CLI code to pass `sys.stdout` explicitly, preserving current behavior.
- Add unit tests that assert on the returned/printed plan without monkeypatching global stdout.

## Impact

- **Primary capability**: `docparsing`
- **Key modules**: `DocParsing/core.py`, `DocParsing/config.py`, `DocParsing/embedding/runtime.py`, `DocParsing/chunking.py`, `DocParsing/embedding.py`, planner utilities.
- **Backward compatibility**: Preserve existing import contracts via façade re-exports; CLI entrypoints keep current names but return structured errors.
- **Testing**: New unit tests for config loaders, CLI validation, optional dependency loaders, and planner output formatting.
- **Documentation**: CLI help, developer docs, and administrator guides need updates to reflect the new modular structure and validation behavior.
- **README updates**: `src/DocsToKG/DocParsing/README.md` will document the shared `CLIValidationError` hierarchy and reference the new error-handling workflow.

## Success Criteria

1. `src/DocsToKG/DocParsing/core.py` shrinks to a façade under 200 lines, delegating functionality to new submodules.
2. `_load_yaml_markers`/`_load_toml_markers` are exposed as documented public helpers, with strict tests guarding their contract.
3. Invalid CLI invocations exit non-zero with friendly messages; automated tests cover representative failure scenarios.
4. Importing `DocsToKG.DocParsing.embedding.runtime` no longer requires optional packages; deferred loaders raise clear messages.
5. Planner display utilities are testable without capturing global stdout and support alternate output streams.

## Open Questions

- Where should documentation for the refactored core modules live (existing README vs. dedicated developer guide)?
- Do we need to version or tag the public config loader API to protect downstream users relying on private helpers today?
