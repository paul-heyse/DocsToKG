"""
Structured logging utilities and manifest logging helpers.

The core CLI modules depend on these helpers to emit consistent JSON logs and to
record structured manifest entries. By isolating the functionality here we keep
`core.py` small and focused on orchestration code.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from DocsToKG.OntologyDownload.logging_utils import JSONFormatter

from .io import manifest_append, resolve_hash_algorithm
from .telemetry import StageTelemetry


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that enriches structured logs with shared context."""

    def __init__(
        self, logger: logging.Logger, base_fields: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store underlying logger and initial structured ``base_fields``."""

        super().__init__(logger, {})
        self.base_fields: Dict[str, Any] = dict(base_fields or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Merge adapter context into ``extra`` metadata for structured output."""

        extra = kwargs.setdefault("extra", {})
        fields = dict(self.base_fields)
        extra_fields = extra.get("extra_fields")
        if isinstance(extra_fields, dict):
            fields.update(extra_fields)
        extra["extra_fields"] = fields
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **fields: object) -> "StructuredLogger":
        """Attach additional persistent fields to the adapter and return ``self``."""

        filtered = {k: v for k, v in fields.items() if v is not None}
        self.base_fields.update(filtered)
        return self

    def child(self, **fields: object) -> "StructuredLogger":
        """Create a new adapter inheriting context with optional overrides."""

        merged = dict(self.base_fields)
        merged.update({k: v for k, v in fields.items() if v is not None})
        return StructuredLogger(self.logger, merged)


def get_logger(
    name: str, level: str = "INFO", *, base_fields: Optional[Dict[str, Any]] = None
) -> StructuredLogger:
    """Get a structured JSON logger configured for console output."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    adapter = getattr(logger, "_docparse_adapter", None)
    if not isinstance(adapter, StructuredLogger):
        adapter = StructuredLogger(logger, base_fields)
        setattr(logger, "_docparse_adapter", adapter)
    elif base_fields:
        adapter.bind(**base_fields)
    return adapter


def log_event(logger: logging.Logger, level: str, message: str, **fields: object) -> None:
    """Emit a structured log record using the ``extra_fields`` convention."""

    normalised_level = str(level).lower()
    if normalised_level in {"warning", "error"}:
        if "stage" not in fields:
            base_stage = (
                getattr(logger, "base_fields", {}).get("stage")
                if hasattr(logger, "base_fields")
                else None
            )
            fields["stage"] = base_stage or "unknown"
        if "doc_id" not in fields:
            fields["doc_id"] = "unknown"
        if "input_hash" not in fields:
            fields["input_hash"] = None
        error_code = fields.get("error_code")
        if not error_code:
            fields["error_code"] = "UNKNOWN"
        else:
            fields["error_code"] = str(error_code).upper()
    else:
        if "stage" not in fields:
            base_stage = (
                getattr(logger, "base_fields", {}).get("stage")
                if hasattr(logger, "base_fields")
                else None
            )
            if base_stage is not None:
                fields["stage"] = base_stage

    emitter = getattr(logger, normalised_level, None)
    if not callable(emitter):
        raise AttributeError(f"Logger has no level '{level}'")
    emitter(message, extra={"extra_fields": fields})


def _stringify_path(value: Path | str | None) -> str | None:
    """Return a string representation for path-like values used in manifests."""

    if value is None:
        return None
    return str(value)


def manifest_log_skip(
    *,
    stage: str,
    doc_id: str,
    input_path: Path | str,
    input_hash: str,
    output_path: Path | str,
    duration_s: float = 0.0,
    schema_version: Optional[str] = None,
    hash_alg: Optional[str] = None,
    **extra: object,
) -> None:
    """Record a manifest entry indicating the pipeline skipped work."""

    telemetry = _STAGE_TELEMETRY_VAR.get()
    if telemetry is not None:
        metadata: Dict[str, Any] = {
            "stage": stage,
            "status": "skip",
            "duration_s": float(duration_s),
            "schema_version": schema_version,
            "input_path": _stringify_path(input_path),
            "input_hash": input_hash,
            "hash_alg": hash_alg or resolve_hash_algorithm(),
            "output_path": _stringify_path(output_path),
        }
        metadata.update(extra)
        reason = metadata.get("reason")
        telemetry.log_skip(
            doc_id=doc_id,
            input_path=input_path,
            reason=str(reason) if reason is not None else None,
            metadata=metadata,
        )
        return

    payload: Dict[str, object] = {
        "stage": stage,
        "doc_id": doc_id,
        "status": "skip",
        "duration_s": float(duration_s),
        "schema_version": schema_version,
        "input_path": _stringify_path(input_path),
        "input_hash": input_hash,
        "hash_alg": hash_alg or resolve_hash_algorithm(),
        "output_path": _stringify_path(output_path),
    }
    payload.update(extra)
    manifest_append(**payload)


def manifest_log_success(
    *,
    stage: str,
    doc_id: str,
    duration_s: float,
    schema_version: str,
    input_path: Path | str,
    input_hash: str,
    output_path: Path | str,
    hash_alg: Optional[str] = None,
    **extra: object,
) -> None:
    """Record a manifest entry marking successful pipeline output."""

    telemetry = _STAGE_TELEMETRY_VAR.get()
    if telemetry is not None:
        metadata: Dict[str, Any] = {
            "stage": stage,
            "status": "success",
            "duration_s": float(duration_s),
            "schema_version": schema_version,
            "input_path": _stringify_path(input_path),
            "input_hash": input_hash,
            "hash_alg": hash_alg or resolve_hash_algorithm(),
            "output_path": _stringify_path(output_path),
        }
        metadata.update(extra)
        tokens: Optional[int] = None
        for key in ("tokens", "chunk_count", "vector_count"):
            value = metadata.get(key)
            if isinstance(value, int):
                tokens = value
                break
        telemetry.log_success(
            doc_id=doc_id,
            input_path=input_path,
            output_path=output_path,
            tokens=tokens,
            schema_version=schema_version,
            duration_s=duration_s,
            metadata=metadata,
        )
        return

    payload: Dict[str, object] = {
        "stage": stage,
        "doc_id": doc_id,
        "status": "success",
        "duration_s": float(duration_s),
        "schema_version": schema_version,
        "input_path": _stringify_path(input_path),
        "input_hash": input_hash,
        "hash_alg": hash_alg or resolve_hash_algorithm(),
        "output_path": _stringify_path(output_path),
    }
    payload.update(extra)
    manifest_append(**payload)


def manifest_log_failure(
    *,
    stage: str,
    doc_id: str,
    duration_s: float,
    schema_version: str,
    input_path: Path | str,
    input_hash: str,
    output_path: Path | str,
    error: str,
    hash_alg: Optional[str] = None,
    **extra: object,
) -> None:
    """Record a manifest entry describing a failed pipeline attempt."""

    telemetry = _STAGE_TELEMETRY_VAR.get()
    if telemetry is not None:
        metadata: Dict[str, Any] = {
            "stage": stage,
            "status": "failure",
            "duration_s": float(duration_s),
            "schema_version": schema_version,
            "input_path": _stringify_path(input_path),
            "input_hash": input_hash,
            "hash_alg": hash_alg or resolve_hash_algorithm(),
            "output_path": _stringify_path(output_path),
            "error": error,
        }
        metadata.update(extra)
        telemetry.log_failure(
            doc_id=doc_id,
            input_path=input_path,
            duration_s=duration_s,
            reason=error,
            metadata=metadata,
            manifest_metadata=metadata,
        )
        return

    payload: Dict[str, object] = {
        "stage": stage,
        "doc_id": doc_id,
        "status": "failure",
        "duration_s": float(duration_s),
        "schema_version": schema_version,
        "input_path": _stringify_path(input_path),
        "input_hash": input_hash,
        "hash_alg": hash_alg or resolve_hash_algorithm(),
        "output_path": _stringify_path(output_path),
        "error": error,
    }
    payload.update(extra)
    manifest_append(**payload)


def summarize_manifest(entries: Sequence[dict]) -> Dict[str, Any]:
    """Compute status counts and durations for manifest ``entries``."""

    status_counter: Dict[str, Counter] = defaultdict(Counter)
    duration_totals: Dict[str, float] = defaultdict(float)
    total_entries: Dict[str, int] = defaultdict(int)
    for entry in entries:
        stage = entry.get("stage", "unknown")
        status = entry.get("status", "unknown")
        total_entries[stage] += 1
        status_counter[stage][status] += 1
        try:
            duration_totals[stage] += float(entry.get("duration_s", 0.0))
        except (TypeError, ValueError):
            continue

    summary: Dict[str, Dict[str, object]] = {}
    for stage, total in total_entries.items():
        summary[stage] = {
            "total": total,
            "statuses": dict(status_counter[stage]),
            "duration_s": round(duration_totals[stage], 3),
        }
    return summary


__all__ = [
    "StructuredLogger",
    "get_logger",
    "log_event",
    "manifest_log_failure",
    "manifest_log_skip",
    "manifest_log_success",
    "set_stage_telemetry",
    "telemetry_scope",
    "summarize_manifest",
]


_STAGE_TELEMETRY_VAR: contextvars.ContextVar[Optional[StageTelemetry]] = (
    contextvars.ContextVar("docparse_stage_telemetry", default=None)
)


def set_stage_telemetry(stage_telemetry: Optional[StageTelemetry]) -> None:
    """Register ``stage_telemetry`` for manifest logging helpers."""

    _STAGE_TELEMETRY_VAR.set(stage_telemetry)


@contextlib.contextmanager
def telemetry_scope(stage_telemetry: Optional[StageTelemetry]):
    """Context manager that temporarily installs ``stage_telemetry``."""

    token = _STAGE_TELEMETRY_VAR.set(stage_telemetry)
    try:
        yield stage_telemetry
    finally:
        _STAGE_TELEMETRY_VAR.reset(token)
