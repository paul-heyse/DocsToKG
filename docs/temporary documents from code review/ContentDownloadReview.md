# Content Download Architecture Review (Updated)

This document captures the high-level architecture of the DocsToKG content
download subsystem after the modularisation refactor. It replaces the older code
review checklist with a concise description of the system as it exists today and
highlights the levers available to maintainers.

## 1. System Overview

DocsToKG now decomposes the download pipeline into four cooperating layers:

1. **Runner orchestration** – :class:`DocsToKG.ContentDownload.runner.DownloadRun`
   encapsulates CLI bootstrapping, resolver pipeline creation, work provider
   wiring, worker pool lifecycle management, and manifest summarisation. Each
   stage exposes a dedicated ``setup_*`` method which can be unit-tested or
   overridden for custom execution environments.
2. **Resolver modularisation** – concrete resolvers live in
   ``ContentDownload/resolvers/`` and register themselves via the shared
   :class:`ResolverRegistry`. ``pipeline.py`` now re-exports the resolver classes
   so historical imports continue to work while new integrations can import from
   the package namespace.
3. **Download strategies** – ``download.py`` defines a ``DownloadStrategy``
   protocol with PDF, HTML, and XML implementations. Strategies coordinate
   resume handling, classification validation, and sidecar cleanup through
   dedicated helpers (``validate_classification``, ``handle_resume_logic``,
   ``cleanup_sidecar_files``, ``build_download_outcome``).
4. **Telemetry and manifests** – ``telemetry.py`` continues to own manifest
   writing. The new helpers ensure consistent recording of SHA-256 hashes,
   retry-after hints, extracted text paths, and skip reasons across all
   strategies.

## 2. Operational Guidance

- **Configuration** – ``args.py`` resolves immutable ``ResolvedConfig`` instances.
  ``bootstrap_run_environment`` performs directory creation and telemetry
  bootstrapping so configuration remains side-effect free.
- **Extending resolvers** – Add new resolver modules under ``resolvers/`` and
  inherit from :class:`RegisteredResolver` or :class:`ApiResolverBase`. Importing
  the module automatically registers the resolver. ``default_resolvers()`` returns
  the priority-ordered resolver list for modifications.
- **Customising downloads** – Pass a custom ``strategy_factory`` through the
  ``DownloadContext.extra`` mapping to override strategy selection.
- **Testing** – Each stage of ``DownloadRun`` has dedicated unit tests. Strategy
  helpers and strategies are covered by ``tests/content_download/test_download_strategy_helpers.py``.

## 3. Follow-up Recommendations

- Add optional benchmarks for resolver and download throughput once GPU/CPU
  environments are available.
- Continue expanding integration coverage with recorded OpenAlex fixtures to
  exercise resolver ordering and strategy overrides without live network access.
- Monitor telemetry for unexpected ``Classification.MISS`` outcomes to refine the
  heuristics contained in ``validate_classification``.

This update reflects the current codebase and should serve as the starting point
for future reviews and onboarding discussions.
