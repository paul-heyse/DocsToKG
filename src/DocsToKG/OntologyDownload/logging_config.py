"""Logging configuration for the ontology downloader."""

from __future__ import annotations

import gzip
import json
import gzip
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from .config import LoggingConfig
from .download import sanitize_filename


def mask_sensitive_data(payload: Dict[str, object]) -> Dict[str, object]:
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
    return uuid.uuid4().hex[:12]


class JSONFormatter(logging.Formatter):
    """Formatter emitting JSON structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
        target.write(source.read())
    path.unlink(missing_ok=True)


def _cleanup_logs(log_dir: Path, retention_days: int) -> None:
    now = datetime.utcnow()
    retention_delta = timedelta(days=retention_days)
    for file in log_dir.glob("*.jsonl"):
        mtime = datetime.utcfromtimestamp(file.stat().st_mtime)
        if now - mtime > retention_delta:
            _compress_old_log(file)
    for file in log_dir.glob("*.jsonl.gz"):
        mtime = datetime.utcfromtimestamp(file.stat().st_mtime)
        if now - mtime > retention_delta:
            file.unlink(missing_ok=True)


def setup_logging(config: LoggingConfig, log_dir: Optional[Path] = None) -> logging.Logger:
    log_dir = log_dir or Path(os.environ.get("ONTOFETCH_LOG_DIR", ""))
    if not log_dir:
        from .core import LOG_DIR  # Local import to avoid circular dependency

        log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_logs(log_dir, config.retention_days)

    logger = logging.getLogger("DocsToKG.OntologyDownload")
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    external_handlers = [h for h in logger.handlers if not getattr(h, "_ontofetch_managed", False)]

    for handler in list(logger.handlers):
        if getattr(handler, "_ontofetch_managed", False):
            logger.removeHandler(handler)
            handler.close()

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    file_name = sanitize_filename(f"ontofetch-{datetime.utcnow().strftime('%Y%m%d')}.jsonl")
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
