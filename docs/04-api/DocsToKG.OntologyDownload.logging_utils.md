# 1. Module: logging_utils

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.logging_utils``.

## 1. Overview

Structured logging helpers shared across ontology download components.

## 2. Functions

### `_compress_old_log(path)`

Compress ``path`` into a ``.gz`` file and remove the original.

### `_cleanup_logs(log_dir, retention_days)`

Rotate or purge log files in ``log_dir`` based on retention policy.

### `setup_logging()`

Configure ontology downloader logging with rotation and JSON sidecars.

### `format(self, record)`

Render ``record`` as a JSON string with DocsToKG-specific fields.

## 3. Classes

### `JSONFormatter`

Formatter emitting masked JSON log entries for ontology downloads.
