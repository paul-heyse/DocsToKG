# 1. Module: telemetry

This reference documents the DocsToKG module ``DocsToKG.DocParsing.telemetry``.

## 1. Overview

Telemetry sink interfaces for DocParsing pipelines.

## 2. Functions

### `_acquire_lock_for(path)`

Return an advisory lock context manager for ``path``.

### `_input_bytes(path)`

Best-effort size lookup for ``path`` returning zero on failure.

### `_append_payload(self, path, payload)`

Append ``payload`` to ``path`` under a file lock.

### `write_attempt(self, attempt)`

Append ``attempt`` to the attempts log.

### `write_manifest_entry(self, entry)`

Append ``entry`` to the manifest log.

### `record_attempt(self)`

Persist an Attempt entry describing the outcome for ``doc_id``.

### `write_manifest(self)`

Append a manifest row for ``doc_id`` and optional metadata.

### `log_success(self)`

Record a successful attempt and mirror the manifest entry.

### `log_failure(self)`

Record a failure attempt and optionally log manifest metadata.

### `log_skip(self)`

Record a skipped attempt and optional manifest metadata.

### `log_config(self)`

Record the configuration manifest emitted at startup.

## 3. Classes

### `Attempt`

Describe a pipeline attempt for a single document.

### `ManifestEntry`

Describe a successful pipeline output.

### `TelemetrySink`

Persistence helper for attempt and manifest telemetry.

### `StageTelemetry`

Lightweight helper binding a sink to a specific stage/run.
