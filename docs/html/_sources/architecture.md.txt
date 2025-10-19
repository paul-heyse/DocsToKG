# 1. Content Download Architecture Overview

```mermaid
graph TD
    Work[Work Artifact] --> Runner[DownloadRun]
    Runner -->|setup| Sinks[Telemetry Sinks]
    Runner -->|setup| Pipeline[ResolverPipeline]
    Runner -->|setup| Provider[OpenAlex Provider]
    Pipeline --> ResolverA[Resolver A]
    Pipeline --> ResolverB[Resolver B]
    Pipeline --> ResolverN[Resolver N]
    ResolverA -->|URL| Download[download_candidate]
    ResolverB -->|URL| Download
    ResolverN -->|URL| Download
    Download --> Strategy[DownloadStrategy]
    Strategy --> Outcome[DownloadOutcome]
    Runner --> Outcome
```

The diagram highlights the modular DocsToKG content download architecture. Each
section below summarises the key components introduced by the
``refactor-content-download-modularization`` change set.

## 1. Runner Orchestration

- ``runner.py`` now centres around :class:`DownloadRun`, a composable orchestrator
  that stages sink creation, resolver pipeline initialisation, work provider
  setup, download state preparation, worker pool management, and batch
  processing.
- Each stage is independently testable, enabling unit coverage without executing
  the full pipeline. ``DownloadRun`` also exposes ``process_work_item`` for
  programmatic reuse.

## 2. Resolver Modularisation

- Resolver implementations live under ``ContentDownload/resolvers/`` (for example
  ``arxiv.py``, ``openalex.py``, ``zenodo.py``). ``resolvers/__init__.py`` wires the
  registry, shared base classes, and concrete resolvers together.
- ``pipeline.py`` re-exports resolver classes (``OpenAlexResolver``,
  ``UnpaywallResolver``, etc.) for backward compatibility, so historical import
  paths continue to work while new integrations can import directly from the
  ``resolvers`` package.
- ``ResolverRegistry`` retains responsibility for discovery and default ordering
  while respecting configuration toggles.

## 3. Strategy-Based Downloads

- ``download.py`` introduces a ``DownloadStrategy`` protocol with concrete PDF,
  HTML, and XML implementations. Strategies control whether a download should
  proceed, how responses are classified, and how artifacts are finalised.
- Shared helpers (``validate_classification``, ``handle_resume_logic``,
  ``cleanup_sidecar_files``, ``build_download_outcome``) encapsulate previously
  inlined logic to keep strategies focused and testable.

## 4. Result Flow

Resolvers emit :class:`ResolverResult` instances that drive
``download_candidate``. The download strategy processes HTTP responses,
constructs :class:`DownloadOutcome` objects, and returns telemetry to the runner.
HEAD pre-checks, resume logic, and corruption detection are now handled by the
strategy helpers, reducing duplicated branching.
