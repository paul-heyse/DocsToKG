"""Structured logging helpers shared across ontology download components."""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from .io import mask_sensitive_data, sanitize_filename
from .settings import LOG_DIR

__all__ = ["JSONFormatter", "setup_logging"]


class JSONFormatter(logging.Formatter):
    """Formatter emitting masked JSON log entries for ontology downloads."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON string with DocsToKG-specific fields."""

        now = datetime.now(timezone.utc)
        payload = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "ontology_id": getattr(record, "ontology_id", None),
            "stage": getattr(record, "stage", None),
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(mask_sensitive_data(payload))


def _compress_old_log(path: Path) -> None:
    """Compress ``path`` into a ``.gz`` file and remove the original."""

    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
        target.write(source.read())
    path.unlink(missing_ok=True)


def _cleanup_logs(log_dir: Path, retention_days: int) -> List[str]:
    """Rotate or purge log files in ``log_dir`` based on retention policy."""

    actions: List[str] = []
    now = datetime.now(timezone.utc)
    retention_delta = timedelta(days=retention_days)
    for file in log_dir.glob("*.jsonl"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            target = file.with_suffix(file.suffix + ".gz")
            _compress_old_log(file)
            actions.append(f"Compressed {file.name} -> {target.name}")
    for file in log_dir.glob("*.jsonl.gz"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            file.unlink(missing_ok=True)
            actions.append(f"Deleted expired archive {file.name}")
    return actions


def setup_logging(
    *,
    level: str = "INFO",
    retention_days: int = 30,
    max_log_size_mb: int = 100,
    log_dir: Optional[Path] = None,
    propagate: bool = False,
) -> logging.Logger:
    """Configure ontology downloader logging with rotation and JSON sidecars."""

    if log_dir is not None:
        resolved_dir = log_dir
    else:
        env_value = os.environ.get("ONTOFETCH_LOG_DIR")
        env_path: Optional[Path] = None
        if env_value is not None:
            stripped = env_value.strip()
            if stripped:
                env_path = Path(stripped)
        resolved_dir = env_path or LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_logs(resolved_dir, retention_days)

    logger = logging.getLogger("DocsToKG.OntologyDownload")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(logger.handlers):
        if getattr(handler, "_ontofetch_managed", False):
            logger.removeHandler(handler)
            if isinstance(handler, logging.StreamHandler):
                stream = getattr(handler, "stream", None)
                if stream in (sys.stdout, sys.stderr):
                    continue
            handler.close()

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_name = sanitize_filename(f"ontofetch-{today}.jsonl")
    file_handler = RotatingFileHandler(
        resolved_dir / file_name,
        maxBytes=int(max_log_size_mb * 1024 * 1024),
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    logger.propagate = propagate
    return logger
