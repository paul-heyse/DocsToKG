# 1. Module: logging

This reference documents the DocsToKG module ``DocsToKG.DocParsing.logging``.

## 1. Overview

Structured logging utilities and manifest logging helpers.

The core CLI modules depend on these helpers to emit consistent JSON logs and to
record structured manifest entries. By isolating the functionality here we keep
`core.py` small and focused on orchestration code.

## 2. Functions

### `get_logger(name, level)`

Get a structured JSON logger configured for console output.

### `log_event(logger, level, message)`

Emit a structured log record using the ``extra_fields`` convention.

### `_stringify_path(value)`

Return a string representation for path-like values used in manifests.

### `manifest_log_skip()`

Record a manifest entry indicating the pipeline skipped work.

### `manifest_log_success()`

Record a manifest entry marking successful pipeline output.

### `manifest_log_failure()`

Record a manifest entry describing a failed pipeline attempt.

### `summarize_manifest(entries)`

Compute status counts and durations for manifest ``entries``.

### `set_stage_telemetry(stage_telemetry)`

Register ``stage_telemetry`` for manifest logging helpers.

### `process(self, msg, kwargs)`

Merge adapter context into ``extra`` metadata for structured output.

### `bind(self)`

Attach additional persistent fields to the adapter and return ``self``.

### `child(self)`

Create a new adapter inheriting context with optional overrides.

## 3. Classes

### `StructuredLogger`

Logger adapter that enriches structured logs with shared context.
