# Module: logging_config

Structured Logging Utilities

This module centralizes structured logging setup for the ontology downloader
subsystem. It provides helpers for masking sensitive fields, emitting JSON log
records, managing correlation identifiers, and rolling log files to maintain a
clean retention window.

## Functions

### `mask_sensitive_data(payload)`

Remove secrets from structured payloads prior to logging.

Args:
payload: Arbitrary key-value pairs that may contain credentials or
tokens gathered from ontology download requests.

Returns:
Copy of the payload where common secret fields are replaced with
`***masked***`.

### `generate_correlation_id()`

Create a short-lived identifier that links related log entries.

Args:
None

Returns:
Twelve character hexadecimal identifier suitable for correlating log
events across the ontology download pipeline.

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

*No documentation available.*

### `format(self, record)`

Serialize a logging record into a JSON line.

Args:
record: Log record emitted by the ontology download components.

Returns:
UTF-8 safe JSON string with masked secrets and correlation context.

## Classes

### `JSONFormatter`

Formatter emitting JSON structured logs.
