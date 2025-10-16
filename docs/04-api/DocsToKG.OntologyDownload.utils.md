# 1. Module: utils

Utility helpers originally documented under ``DocsToKG.OntologyDownload.utils`` are
consolidated into ``DocsToKG.OntologyDownload.ontology_download``.

Currently hosts the unified exponential backoff helper that underpins resolver
planning and download retries in the refactored ontology downloader.

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
