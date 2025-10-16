# Design: Modularity Refactor for DocsToKG.OntologyDownload

## Goals
- Break `ontology_download.py` into purpose-scoped modules without changing runtime behaviour or public API.
- Maintain lazy import patterns from the package facade and shims.
- Avoid circular dependencies while keeping code discoverable.

## Proposed Module Layout
| Module | Responsibility |
|--------|----------------|
| `config.py` | Pydantic models (`DownloadConfiguration`, `DefaultsConfig`, etc.), environment overrides, YAML loading, schema validation, and helper functions such as `build_resolved_config`. |
| `io_safe.py` | Filename/url sanitisation, sensitive data masking, checksum helpers, archive extraction, and URL security enforcement. |
| `net.py` | Retry helpers, token bucket, `StreamingDownloader`, `download_stream`, and related telemetry helpers. |
| `validation_core.py` | `ValidationRequest/Result`, streaming canonicaliser, validator runners, subprocess worker entrypoints, and validator registration. |
| `pipeline.py` | `FetchSpec/Result`, manifest helpers, storage integration, planner/downloader coordination (`plan_*`, `fetch_*`). |
| `plugins.py` | Resolver/validator plugin discovery + registration utilities shared by CLI and orchestration. |

`ontology_download.py` will:
- Re-export classes/functions from the modules above for backwards compatibility (maintaining `__all__`).
- Host minimal glue such as CLI entrypoint wiring and lazy plugin bootstrap.
- Provide shared constants (e.g., validator worker dispatch) by importing from the new modules as needed.

## Dependency Strategy
- `config.py` is foundational; other modules import typed models from it.
- `io_safe.py` has no inward dependencies beyond `ConfigError`.
- `net.py` depends on `config` and `io_safe`.
- `validation_core.py` depends on `config`, `io_safe`, and `net` (for URL sanitisation, downloads, and telemetry helpers).
- `pipeline.py` depends on `config`, `io_safe`, `net`, `validation_core`, and `plugins` for orchestration, manifest handling, and storage coordination.
- `plugins.py` depends only on standard library + lazy imports of resolver registries.

To prevent cycles:
- Use local imports for heavy interactions (e.g., pipeline calling `config.build_resolved_config`, validation invoking canonicaliser).
- Keep CLI entrypoints and worker dispatch in `ontology_download.py`, importing submodules lazily to avoid circular references.
- Where data classes are shared (e.g., `FetchSpec`), import types in a single direction (pipeline -> config) and expose flattened aliases for facade consumers.

## Compatibility
- `DocsToKG.OntologyDownload.__init__` continues to lazily forward to `ontology_download` which re-exports the same names.
- `optdeps` and `storage` shims continue to grab getters/constants via the orchestrator.
- Tests referencing module paths will be updated to new imports, and the public API guard/assertions will confirm `__all__` stays in sync with `PUBLIC_API_MANIFEST`.

## Testing Plan
- Run existing ontology downloader test suite, focusing on `tests/ontology_download`.
- Exercise CLI smoke tests (`direnv exec . python -m DocsToKG.OntologyDownload.cli doctor`).
- Public API guard + import time benchmark must pass unchanged.
- `openspec validate refactor-ontology-download-modularity --strict` before submission.
