"""
Structured Logging Utilities

This module centralizes structured logging setup for the ontology downloader
subsystem. It provides helpers for masking sensitive fields, emitting JSON log
records, managing correlation identifiers, and rolling log files to maintain a
clean retention window.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from .config import LoggingConfig
from .download import sanitize_filename


def mask_sensitive_data(payload: Dict[str, object]) -> Dict[str, object]:
    """Remove secrets from structured payloads prior to logging.

    Args:
        payload: Arbitrary key-value pairs that may contain credentials or
            tokens gathered from ontology download requests.

    Returns:
        Copy of the payload where common secret fields are replaced with
        `***masked***`.

    Examples:
        >>> mask_sensitive_data({"token": "secret", "status": "ok"})
        {'token': '***masked***', 'status': 'ok'}
    """
    sensitive_keys = {"authorization", "api_key", "apikey", "token", "secret", "password"}
    masked: Dict[str, object] = {}
    for key, value in payload.items():
        lower = key.lower()
        if lower in sensitive_keys:
            masked[key] = "***masked***"
        elif isinstance(value, str) and "apikey" in value.lower():
            masked[key] = "***masked***"
        else:
            masked[key] = value
    return masked


def generate_correlation_id() -> str:
    """Create a short-lived identifier that links related log entries.

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
    """
    return uuid.uuid4().hex[:12]


class JSONFormatter(logging.Formatter):
    """Formatter emitting JSON structured logs.

    Attributes:
        None

    Examples:
        >>> formatter = JSONFormatter()
        >>> isinstance(formatter.format(logging.makeLogRecord({'msg': 'test'})), str)
        True
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a logging record into a JSON line.

        Args:
            record: Log record emitted by the ontology download components.

        Returns:
            UTF-8 safe JSON string with masked secrets and correlation context.
        """
        now = datetime.now(timezone.utc)
        log_obj = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "ontology_id": getattr(record, "ontology_id", None),
            "stage": getattr(record, "stage", None),
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_obj.update(record.extra_fields)
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(mask_sensitive_data(log_obj))


def _compress_old_log(path: Path) -> None:
    """Compress a log file in-place using gzip to reclaim disk space.

    Args:
        path: Path to the `.log` file that should be compressed.
    """
    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
        target.write(source.read())
    path.unlink(missing_ok=True)


def _cleanup_logs(log_dir: Path, retention_days: int) -> None:
    """Apply rotation and retention policy to the log directory.

    Args:
        log_dir: Directory containing daily log files.
        retention_days: Number of days to keep uncompressed or compressed logs
            before deleting them.
    """
    now = datetime.now(timezone.utc)
    retention_delta = timedelta(days=retention_days)
    for file in log_dir.glob("*.jsonl"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            _compress_old_log(file)
    for file in log_dir.glob("*.jsonl.gz"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            file.unlink(missing_ok=True)


def setup_logging(config: LoggingConfig, log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure structured logging handlers for ontology downloads.

    Args:
        config: Logging configuration containing level, size, and retention.
        log_dir: Optional directory override for log file placement.

    Returns:
        Configured logger instance scoped to the ontology downloader.

    Examples:
        >>> logger = setup_logging(LoggingConfig(level="INFO", max_log_size_mb=1, retention_days=1))
        >>> logger.name
        'DocsToKG.OntologyDownload'
    """
    log_dir = log_dir or Path(os.environ.get("ONTOFETCH_LOG_DIR", ""))
    if not log_dir:
        from .core import LOG_DIR  # Local import to avoid circular dependency

        log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_logs(log_dir, config.retention_days)

    logger = logging.getLogger("DocsToKG.OntologyDownload")
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    for handler in list(logger.handlers):
        if getattr(handler, "_ontofetch_managed", False):
            logger.removeHandler(handler)
            handler.close()

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_name = sanitize_filename(f"ontofetch-{today}.jsonl")
    file_handler = RotatingFileHandler(
        log_dir / file_name,
        maxBytes=int(config.max_log_size_mb * 1024 * 1024),
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    logger.propagate = True

    return logger


__all__ = ["setup_logging", "mask_sensitive_data", "generate_correlation_id"]
