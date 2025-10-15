# 1. Module: logging_config

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.logging_config``.

Structured logging utilities for the ontology downloader.

The refactored downloader emits JSON logs with correlation identifiers so
concurrent planning, streaming normalization, and resolver fallback attempts
can be traced. This module wires log rotation, retention, masking, and
correlation id helpers that are also surfaced through the enhanced `doctor`
diagnostics command.

## 1. Functions

### `mask_sensitive_data(payload)`

Remove secrets from structured payloads prior to logging.

Args:
payload: Arbitrary key-value pairs that may contain credentials or
tokens gathered from ontology download requests.

Returns:
Copy of the payload where common secret fields are replaced with
`***masked***`.

Examples:
>>> mask_sensitive_data({"token": "secret", "status": "ok"})
{'token': '***masked***', 'status': 'ok'}

### `generate_correlation_id()`

Create a short-lived identifier that links related log entries.

Args:
None

Returns:
Twelve character hexadecimal identifier suitable for correlating log
events across the ontology download pipeline.

Raises:
None

Examples:
>>> cid = generate_correlation_id()
>>> len(cid)
12

### `_compress_old_log(path)`

Compress a log file in-place using gzip to reclaim disk space.

Args:
path: Path to the `.log` file that should be compressed.

### `_cleanup_logs(log_dir, retention_days)`

Apply rotation and retention policy to the log directory.

Args:
log_dir: Directory containing daily log files.
retention_days: Number of days to keep uncompressed or compressed logs
before deleting them.

### `setup_logging(config, log_dir)`

Configure structured logging handlers for ontology downloads.

Args:
config: Logging configuration containing level, size, and retention.
log_dir: Optional directory override for log file placement.

Returns:
Configured logger instance scoped to the ontology downloader.

Examples:
>>> logger = setup_logging(LoggingConfig(level="INFO", max_log_size_mb=1, retention_days=1))
>>> logger.name
'DocsToKG.OntologyDownload'

### `format(self, record)`

Serialize a logging record into a JSON line.

Args:
record: Log record emitted by the ontology download components.

Returns:
UTF-8 safe JSON string with masked secrets and correlation context.

## 2. Classes

### `JSONFormatter`

Formatter emitting JSON structured logs.

Attributes:
None

Examples:
>>> formatter = JSONFormatter()
>>> isinstance(formatter.format(logging.makeLogRecord({'msg': 'test'})), str)
True
