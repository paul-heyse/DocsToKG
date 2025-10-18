# 1. Module: download_pyalex_pdfs

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.cli``.

## 1. Overview

Command-line entry point for orchestrating the OpenAlex content download
pipeline. The module exposes the ``main`` function used by ``python -m``
invocations and re-exports the primary orchestration helpers so downstream
tools can share the same configuration, pipeline, and download primitives.

The CLI couples argument parsing, configuration resolution, and run orchestration
through :class:`DocsToKG.ContentDownload.runner.DownloadRun`. Supporting helpers
are surfaced for programmatic integrations that wish to reuse the download
pipeline without shelling out to the CLI process.

Key features:

- Bootstrap utility for creating run directories and initialising telemetry.
- Access to resolver discovery and configuration helpers via ``pipeline``
  re-exports.
- Direct access to download primitives such as ``process_one_work`` and
  ``download_candidate`` for customised workflows.
- Compatibility shims (``_build_download_outcome``) maintained for legacy call
  sites while new helpers (``build_download_outcome``) offer richer telemetry
  capture.

## 2. Functions and Re-exports

### `main(argv=None)`

Entry point used by the ``docstokg-download`` console script. Parses CLI
arguments, resolves configuration via :func:`resolve_config`, bootstraps the run
environment, and executes :meth:`DownloadRun.run`. Returns a
:class:`RunResult` summarising the manifest statistics for the invocation.

### `DownloadRun`

Re-export of :class:`DocsToKG.ContentDownload.runner.DownloadRun`. The class
exposes stage-based helpers (``setup_sinks``, ``setup_resolver_pipeline``,
``process_work_item``) and the :meth:`run` orchestration method.

### `process_one_work(...)`

Re-export of :func:`DocsToKG.ContentDownload.download.process_one_work`.
Normalises an OpenAlex work, executes the resolver pipeline, and records manifest
entries for each download attempt.

### `download_candidate(...)`

Re-export of :func:`DocsToKG.ContentDownload.download.download_candidate`. Handles
single-URL downloads with streaming classification, retry bookkeeping, and
strategy dispatch.

### `build_download_outcome(...)`

Re-export of :func:`DocsToKG.ContentDownload.download.build_download_outcome`.
Constructs a :class:`DownloadOutcome` after applying corruption heuristics and
capturing HTTP metadata.

### `_build_download_outcome(...)`

Legacy wrapper maintained for historical imports. Internally forwards to
:func:`build_download_outcome`.

### `create_artifact(work, pdf_dir, html_dir, xml_dir)`

Re-export of :func:`DocsToKG.ContentDownload.download.create_artifact`. Generates
``WorkArtifact`` instances from OpenAlex payloads and determines output
locations for PDF, HTML, and XML sidecars.

### `DownloadOptions`

Re-export of :class:`DocsToKG.ContentDownload.download.DownloadOptions`.
Immutable collection of per-run options shared across work items.

### `DownloadState`

Re-export of :class:`DocsToKG.ContentDownload.download.DownloadState`. Enum
tracking the streaming state of the active download.

### `ensure_dir(path)`

Re-export of :func:`DocsToKG.ContentDownload.download.ensure_dir`. Creates output
directories on demand.

### `apply_config_overrides(config, overrides)`

Re-export of :func:`DocsToKG.ContentDownload.pipeline.apply_config_overrides` for
convenience. Applies resolver overrides loaded from YAML/JSON configurations.

### `default_resolvers()`

Returns the default resolver ordering from
:mod:`DocsToKG.ContentDownload.resolvers`. Useful for programmatic invocations
that need to inspect or extend the resolver stack.

### `load_resolver_config(path, *, overrides=None)`

Wrapper around :func:`DocsToKG.ContentDownload.pipeline.load_resolver_config`.
Loads resolver configuration files and applies optional CLI overrides.

### `load_previous_manifest(path)`

Re-export of :func:`DocsToKG.ContentDownload.telemetry.load_previous_manifest`.
Indexes previous manifest runs so resume logic can short-circuit completed
artifacts.

### `oa_config`

Re-export of :mod:`pyalex.config`. Provides access to OpenAlex API configuration
for callers that integrate the CLI module.

The module also re-exports telemetry sink classes (``AttemptSink``, ``CsvSink``,
``JsonlSink``, ``LastAttemptCsvSink``, ``ManifestIndexSink``, ``MultiSink``) and
core helpers such as :func:`iterate_openalex` and :func:`resolve_topic_id_if_needed`
for completeness. Refer to the download module documentation for detailed
behaviour of the shared download primitives.
