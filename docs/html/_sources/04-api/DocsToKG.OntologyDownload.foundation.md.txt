# 1. Module: foundation

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.foundation``.

Cross-cutting utilities shared across the ontology downloader package.

This module intentionally groups helpers that are required by multiple layers
of the pipeline—retry orchestration, safe filename generation, and correlation
id creation—so that higher-level modules such as networking, logging, and the
CLI can depend on a single lightweight utility surface without introducing
import cycles.

## 1. Functions

### `retry_with_backoff(func)`

Execute ``func`` with exponential backoff until it succeeds.

Args:
func: Zero-argument callable to invoke.
retryable: Predicate returning ``True`` when the raised exception should
trigger another attempt.
max_attempts: Maximum number of attempts including the initial call.
backoff_base: Base delay in seconds used for the exponential schedule.
jitter: Maximum random jitter (uniform) added to each delay.
callback: Optional hook invoked before sleeping with
``(attempt_number, error, delay_seconds)``.
sleep: Sleep function, overridable for deterministic tests.

Returns:
The result produced by ``func`` when it succeeds.

Raises:
ValueError: If ``max_attempts`` is less than one.
BaseException: Re-raises the last exception from ``func`` when retries
are exhausted or the predicate indicates it is not retryable.

### `sanitize_filename(filename)`

Sanitize filenames to prevent directory traversal and unsafe characters.

Args:
filename: Candidate filename provided by an upstream service.

Returns:
Safe filename compatible with local filesystem storage.

### `generate_correlation_id()`

Create a short-lived identifier that links related log entries.

### `mask_sensitive_data(payload)`

Remove secrets from structured payloads prior to logging.
