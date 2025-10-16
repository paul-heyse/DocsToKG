# 1. Module Organization and Navigation Guide

This guide defines how source files should be structured so that classes, functions,
and helpers are easy to discover, maintain, and document. The conventions pair with
the NAVMAP metadata described in `CODE_ANNOTATION_STANDARDS.md` and are intended to
keep the DocsToKG codebase coherent as it grows.

## 2. Purpose

- Create predictable ordering within every module so contributors know where to add
  new logic.
- Support automated agents and documentation tooling by exposing consistent section
  boundaries.
- Reduce merge conflicts and code drift by keeping related functionality clustered.

## 3. Core Principles

1. **Group by responsibility, not chronology.** Keep related behaviour together even
   if it was authored at different times.
2. **Prefer narrow modules.** When a file exceeds ~500 lines or mixes unrelated
   responsibilities, split the content into focused submodules.
3. **Document the layout.** Every module must contain a NAVMAP block that mirrors the
   clusters defined below so automation can navigate the file.
4. **Keep public APIs forward.** Readers should encounter the main entry points before
   the private helpers that implement them.
5. **Minimise cross-module leakage.** Share utilities through dedicated helper
   modules instead of letting unrelated files import deep internals.

## 4. Standard Ordering Convention

Every module should follow this ordering unless a documented exception exists:

1. **Module metadata.**
   - NAVMAP block
   - Module docstring
   - Module-level constants, feature flags, and type aliases
2. **Public classes and dataclasses.**
   - Place the primary orchestrator class first
   - Follow with supporting data carriers (configs, payloads, results)
3. **Public functions.**
   - Entry points that external callers import directly
   - Keep high-level factory functions near related classes
4. **Private helpers.**
   - Pure helpers grouped by the public API they serve
   - I/O helpers together, transformation helpers together, etc.
5. **Module entry hooks.**
   - `if __name__ == "__main__":` blocks
   - CLI registration, plugin exports, or framework bindings

When a module spans multiple feature clusters (for example CLI command groups versus
shared utilities), carve the helpers into clearly labelled sections using inline
comment dividers (for example `# --- CLI Command Handlers ---`).

## 5. Navigation Metadata (NAVMAP)

- Add a single NAVMAP block at the top of each module describing the major sections.
- Use `sections` entries to match the ordering convention and provide anchors
  (for example `"anchor": "CLI"` for command handlers).
- Ensure the NAVMAP order matches the physical layout so tooling can jump to the
  right block when generating documentation or performing refactors.
- Update the NAVMAP whenever sections move, names change, or new clusters are added.

### 5.1 NAVMAP Example

```python
# === NAVMAP v1 ===
# {
#   "module": "src.DocParsing.cli",
#   "purpose": "CLI entry points for document parsing workflows",
#   "sections": [
#     {"id": "imports", "name": "Imports and Globals", "anchor": "imports", "kind": "infra"},
#     {"id": "commands", "name": "CLI Commands", "anchor": "cli", "kind": "api"},
#     {"id": "helpers", "name": "Command Helpers", "anchor": "helpers", "kind": "internal"}
#   ]
# }
# === /NAVMAP ===
```

Keep the JSON compact (two-space indentation) and ensure every section listed here
exists in the file with a matching heading.

## 6. Formatting Requirements

### 6.1 Section Divider Format

- Start each major cluster with a comment divider using three components:
  `# --- <Section Name> ---`.
- Leave a single blank line between the divider and the first definition.
- Use matching casing between the divider, NAVMAP `name`, and any related comments so
  automation can verify alignment.

### 6.2 Imports, Exports, and Constants

- Imports appear immediately after the module docstring and should follow the standard
  library → third-party → local ordering.
- Declare `__all__` alongside other module-level constants to surface the intended
  public API. Keep the tuple/list alphabetised to match the order of public classes
  and functions defined later in the module.
- Group related constants together beneath a `# --- Configuration ---` divider when
  there are more than three items.

### 6.3 Matching NAVMAP to Physical Layout

- The section order in NAVMAP must match the physical order of the divider comments.
- Avoid creating NAVMAP entries for single functions unless they represent a meaningful
  section (for example a command group in the CLI module).
- When removing a section from a module, delete the corresponding NAVMAP entry in the
  same change set to prevent stale navigation metadata.

## 7. Module Skeleton Example

```python
# === NAVMAP v1 ===
# {
#   "module": "src.DocParsing.example",
#   "purpose": "Illustrative template for module organisation",
#   "sections": [
#     {"id": "globals", "name": "Globals", "anchor": "globals", "kind": "infra"},
#     {"id": "api", "name": "Public API", "anchor": "api", "kind": "api"},
#     {"id": "helpers", "name": "Private Helpers", "anchor": "helpers", "kind": "internal"}
#   ]
# }
# === /NAVMAP ===

"""
Example Module

Short description of the module’s purpose and the workflows it supports.
"""

# --- Globals ---
__all__ = ("process_documents", "DocumentProcessor")
DEFAULT_BATCH_SIZE = 50


# --- Public API ---
class DocumentProcessor:
    ...


def process_documents(doc_ids: Sequence[str]) -> List[Result]:
    ...


# --- Private Helpers ---
def _parse_document(path: Path) -> ParsedDocument:
    ...
```

This skeleton demonstrates the ordering, divider format, and NAVMAP alignment required
for production modules.

## 8. Module-Specific Guidance

### 8.1. `DocParsing/cli.py`

1. Start with CLI-specific constants, enums, and Click shared options.
2. Group command functions by user workflow (`ingest`, `process`, `export`).
3. Place shared helper routines (`_load_config`, `_print_summary`) immediately after
   the command they support.
4. Move any reusable helpers into a future `cli_utils.py` if two or more commands use
   them, and import those helpers near the top to avoid circular dependencies.

### 8.2. `DocParsing/pipelines.py`

1. Define pipeline configuration dataclasses first.
2. Follow with pipeline builder functions ordered by execution order
   (`build_ingest_pipeline`, `build_transform_pipeline`, `build_emit_pipeline`).
3. Keep transformer/helper functions grouped under comment headers that match each
   stage to avoid mixing ingest and emit logic.

### 8.3. `DocParsing/_common.py` and `DocParsing/schemas.py`

1. Begin with Pydantic models and enums.
2. Document validation helpers next, keeping synchronous and asynchronous utilities
   in separate clusters.
3. Finish with serialization and conversion routines that depend on the earlier
   definitions.

### 8.4. `HybridSearch/*`

1. `service.py` should open with the primary service class and public orchestration
   functions, followed by scoring, request/response adapters, and finally private
   helpers.
2. Keep storage- and FAISS-specific helpers in `storage.py` and `vectorstore.py`
   rather than importing them into `service.py`.
3. Extract shared logging, metrics, and configuration adapters into `observability.py`
   or `config.py` instead of duplicating them across modules.

### 8.5. `OntologyDownload/*`

1. Organise modules by workflow phase: request resolution, download execution,
   normalization, validation, and storage.
2. Expose the module’s public entry point (`download_ontology`, `run_validators`)
   immediately after constants.
3. Cluster subprocess, file system, and network helpers separately so changes in one
   area do not require scanning the entire file.

## 9. Supporting Modules and Utilities

- Introduce `utils` modules only when three or more modules need the same helper.
- Keep side-effect-heavy helpers (network, filesystem) separated from pure
  transformations to simplify testing.
- When creating a new helper module, add a NAVMAP and follow the standard ordering
  convention so it remains discoverable.

## 10. Tooling and Automation

1. Extend `docs/scripts/validate_code_annotations.py` to assert that each module
   contains a NAVMAP block and that the sections are ordered according to the
   convention above.
2. Update `generate_api_docs.py` to include section anchors when emitting Markdown so
   the API reference mirrors the on-disk structure.
3. Add targeted unit tests for navigation helpers if we build automation that
   rewrites files based on NAVMAP data.

## 11. Rollout Strategy

1. **Document first.** Socialise this guide and confirm buy-in from module owners.
2. **Template module.** Apply the ordering and NAVMAP upgrades to a representative
   module (`DocParsing/cli.py`) to gather feedback.
3. **Incremental refactor.** Tackle modules in small batches, running the full test
   suite and documentation pipeline after each batch.
4. **Automate enforcement.** Enable the extended validation scripts once the majority
   of modules adopt the layout to prevent regressions.
5. **Review cadence.** Periodically audit large modules for drift and ensure new code
   adheres to the established sections.
