# 1. Module: runner

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.runner``.

## 1. Overview

Execution harness for DocsToKG content download runs.

## 2. Functions

### `_calculate_equal_jitter_delay(attempt)`

Return an exponential backoff delay using equal jitter.

### `iterate_openalex(query, per_page, max_results)`

Iterate over OpenAlex works respecting pagination, limits, and retry policy.

Retries honour ``Retry-After`` headers while applying an equal-jitter
exponential backoff capped by ``retry_max_delay`` to avoid unbounded sleeps.

### `run(resolved)`

Execute the download pipeline using a :class:`DownloadRun` orchestration.

### `update_from_result(self, result)`

Update aggregate counters from an individual work result.

### `record_worker_failure(self)`

Increment the worker failure counter in a thread-safe manner.

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

### `_load_resume_state(self, resume_path)`

Load resume metadata from JSON manifests with SQLite fallback.

### `setup_worker_pool(self)`

Create a thread pool when concurrency is enabled.

### `process_work_item(self, work, options, session)`

Process a single work artefact and update aggregate counters.

### `_record_worker_crash_manifest(self, artifact_context, exc)`

Record a manifest entry for a worker crash if telemetry is available.

### `_handle_worker_exception(self, state, exc)`

Apply consistent crash handling for sequential and threaded workers.

### `run(self)`

Execute the content download pipeline and return the aggregate result.

### `_retry_after_seconds(exc)`

*No documentation available.*

### `_jittered_delay(attempt_number)`

*No documentation available.*

### `_build_json_lookup()`

*No documentation available.*

### `_build_thread_session()`

*No documentation available.*

### `_submit(work_item)`

*No documentation available.*

### `_handle_future(completed_future)`

*No documentation available.*

### `_runner()`

*No documentation available.*

## 3. Classes

### `DownloadRunState`

Mutable run-time state shared across the runner lifecycle.

### `DownloadRun`

Stage-oriented orchestration for executing a content download run.
