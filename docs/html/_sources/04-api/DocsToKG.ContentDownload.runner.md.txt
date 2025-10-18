# 1. Module: runner

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.runner``.

## 1. Overview

Execution harness for DocsToKG content download runs.

## 2. Functions

### `iterate_openalex(query, per_page, max_results)`

Iterate over OpenAlex works respecting pagination and limits.

### `run(resolved)`

Execute the download pipeline using a :class:`DownloadRun` orchestration.

### `update_from_result(self, result)`

Update aggregate counters from an individual work result.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `close(self)`

Release resources owned by the run instance.

### `setup_sinks(self, stack)`

Initialise telemetry sinks responsible for manifest and summary data.

### `setup_resolver_pipeline(self)`

Create the resolver pipeline backed by telemetry and metrics.

### `setup_work_provider(self)`

Construct the OpenAlex work provider used to yield artefacts.

### `setup_download_state(self, session_factory, robots_cache)`

Initialise download options and counters for the run.

### `setup_worker_pool(self)`

Create a thread pool when concurrency is enabled.

### `process_work_item(self, work, options, session)`

Process a single work artefact and update aggregate counters.

### `check_budget_limits(self)`

Return ``True`` when request or byte budgets have been exhausted.

### `run(self)`

Execute the content download pipeline and return the aggregate result.

### `_build_thread_session()`

*No documentation available.*

### `_submit(work_item)`

*No documentation available.*

### `_handle_future(completed_future)`

*No documentation available.*

## 3. Classes

### `DownloadRunState`

Mutable run-time state shared across the runner lifecycle.

### `DownloadRun`

Stage-oriented orchestration for executing a content download run.
