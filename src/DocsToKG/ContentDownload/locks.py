# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.locks",
#   "purpose": "File locking helpers for manifests, telemetry, and artifacts",
#   "sections": [
#     {"id": "configure-lock-root", "name": "configure_lock_root", "anchor": "function-configure-lock-root", "kind": "function"},
#     {"id": "manifest-lock", "name": "manifest_lock", "anchor": "function-manifest-lock", "kind": "function"},
#     {"id": "telemetry-lock", "name": "telemetry_lock", "anchor": "function-telemetry-lock", "kind": "function"},
#     {"id": "lock-metrics-snapshot", "name": "lock_metrics_snapshot", "anchor": "function-lock-metrics-snapshot", "kind": "function"},
#     {"id": "reset-lock-root", "name": "reset_lock_root", "anchor": "function-reset-lock-root", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Centralised file-based locking utilities for DocsToKG content downloads.

Responsibilities
----------------
- Provide opinionated helpers (:func:`manifest_lock`, :func:`telemetry_lock`,
  :func:`summary_lock`, etc.) that map logical resources to well-known lock
  files under the run root.
- Expose configuration hooks (:func:`configure_lock_root`,
  :func:`reset_lock_root`) so runners and tests can isolate lock directories.
- Capture acquisition/hold timing via :func:`lock_metrics_snapshot` to feed
  telemetry and troubleshoot contention.

Design Notes
------------
- Locks are implemented with :mod:`filelock` and default to hard locks; set
  ``DOCSTOKG_LOCK_USE_SOFT`` or per-lock environment variables to opt into soft
  locks or custom timeouts.
- All helpers acquire the module-level guard before mutating lock configuration
  to remain thread-safe across runner workers.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

from filelock import FileLock, SoftFileLock, Timeout

__all__ = [
    "Timeout",
    "configure_lock_root",
    "manifest_lock",
    "telemetry_lock",
    "sqlite_lock",
    "artifact_lock",
    "summary_lock",
    "lock_metrics_snapshot",
    "reset_lock_root",
]

LOGGER = logging.getLogger("DocsToKG.ContentDownload.locks")
logging.getLogger("filelock").setLevel(logging.INFO)

_LOCK_DIR_NAME = "locks"
_SOFT_LOCK_ENV = "DOCSTOKG_LOCK_USE_SOFT"
_LOCK_ROOT_ENV = "DOCSTOKG_LOCK_ROOT"
_LOCK_TIMEOUT_ENV_PREFIX = "DOCSTOKG_LOCK_TIMEOUT_"
_LOCK_MODE_ENV = "DOCSTOKG_LOCK_MODE"
_LOCK_POLL_INTERVAL_ENV = "DOCSTOKG_LOCK_POLL_INTERVAL"

_DEFAULT_TIMEOUTS: Dict[str, float] = {
    "manifest": 5.0,
    "telemetry": 5.0,
    "summary": 5.0,
    "sqlite": 10.0,
    "artifact": 30.0,
}

_DEFAULT_POLL_INTERVAL = 0.1  # seconds
_DEFAULT_LOCK_MODE = 0o640

_lock_config_guard = threading.RLock()
_lock_root: Optional[Path] = None
_lock_dir: Optional[Path] = None


@dataclass
class _LockMetrics:
    acquire_total: int = 0
    timeout_total: int = 0
    wait_ms_sum: float = 0.0
    wait_ms_samples: List[float] = field(default_factory=list)
    hold_ms_sum: float = 0.0
    hold_ms_samples: List[float] = field(default_factory=list)


_metrics_guard = threading.RLock()
_metrics: Dict[str, _LockMetrics] = {}


def configure_lock_root(run_root: Path) -> Path:
    """Set the base directory where lock files should be created.

    Args:
        run_root: Directory representing the active run root. A ``locks``
            subdirectory will be created if needed.

    Returns:
        Path to the configured lock directory.
    """

    resolved = Path(run_root).expanduser().resolve(strict=False)
    lock_dir = resolved / _LOCK_DIR_NAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    with _lock_config_guard:
        global _lock_root, _lock_dir
        _lock_root = resolved
        _lock_dir = lock_dir
    return lock_dir


def reset_lock_root() -> None:
    """Reset the configured lock root so future calls recompute directories."""

    with _lock_config_guard:
        global _lock_root, _lock_dir
        _lock_root = None
        _lock_dir = None


def _get_lock_dir() -> Path:
    with _lock_config_guard:
        if _lock_dir is not None:
            return _lock_dir

        env_root = os.getenv(_LOCK_ROOT_ENV)
        if env_root:
            return configure_lock_root(Path(env_root))
    # Fallback: lazily configure using the current working directory
    cwd_lock_dir = Path.cwd() / _LOCK_DIR_NAME
    return configure_lock_root(cwd_lock_dir.parent)


def _select_lock_class():
    return SoftFileLock if os.getenv(_SOFT_LOCK_ENV) else FileLock


def _lock_mode() -> int:
    raw = os.getenv(_LOCK_MODE_ENV)
    if not raw:
        return _DEFAULT_LOCK_MODE
    try:
        # Accept decimal or octal strings (e.g., "0o600")
        return int(raw, 0)
    except ValueError:
        LOGGER.warning(
            "Invalid %s value '%s'; defaulting to %s", _LOCK_MODE_ENV, raw, _DEFAULT_LOCK_MODE
        )
        return _DEFAULT_LOCK_MODE


def _poll_interval() -> float:
    raw = os.getenv(_LOCK_POLL_INTERVAL_ENV)
    if not raw:
        return _DEFAULT_POLL_INTERVAL
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        LOGGER.warning(
            "Invalid %s value '%s'; defaulting to %.3fs",
            _LOCK_POLL_INTERVAL_ENV,
            raw,
            _DEFAULT_POLL_INTERVAL,
        )
        return _DEFAULT_POLL_INTERVAL


def _timeout_for(category: str, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    env_name = f"{_LOCK_TIMEOUT_ENV_PREFIX}{category.upper()}"
    raw = os.getenv(env_name)
    if raw:
        try:
            value = float(raw)
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            LOGGER.warning("Invalid %s value '%s'; falling back to default.", env_name, raw)
    return _DEFAULT_TIMEOUTS[category]


def _hash_path(target: Path) -> str:
    digest = hashlib.sha256(str(target).encode("utf-8")).hexdigest()
    return digest[:24]


def _lock_file_for(category: str, target: Path) -> Path:
    lock_dir = _get_lock_dir()
    digest = _hash_path(target)
    filename = f"{category}.{digest}.lock"
    return lock_dir / filename


def _record_timeout(category: str, wait_ms: float) -> None:
    with _metrics_guard:
        metrics = _metrics.setdefault(category, _LockMetrics())
        metrics.timeout_total += 1
        metrics.wait_ms_sum += wait_ms
        metrics.wait_ms_samples.append(wait_ms)


def _record_success(category: str, wait_ms: float, hold_ms: float) -> None:
    with _metrics_guard:
        metrics = _metrics.setdefault(category, _LockMetrics())
        metrics.acquire_total += 1
        metrics.wait_ms_sum += wait_ms
        metrics.wait_ms_samples.append(wait_ms)
        metrics.hold_ms_sum += hold_ms
        metrics.hold_ms_samples.append(hold_ms)


def _p95(samples: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in samples if value >= 0)
    if not ordered:
        return 0.0
    index = int(max(len(ordered) - 1, 0) * 0.95)
    return ordered[index]


class _LockContext:
    __slots__ = ("_lock", "_category", "_wait_ms", "_acquired_at", "_released")

    def __init__(self, lock, category: str, wait_ms: float) -> None:
        self._lock = lock
        self._category = category
        self._wait_ms = wait_ms
        self._acquired_at = time.monotonic()
        self._released = False

    def __enter__(self) -> "_LockContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        try:
            self._lock.release()
        finally:
            hold_ms = max((time.monotonic() - self._acquired_at) * 1000.0, 0.0)
            _record_success(self._category, self._wait_ms, hold_ms)
            LOGGER.debug(
                "lock-release category=%s hold_ms=%.3f wait_ms=%.3f",
                self._category,
                hold_ms,
                self._wait_ms,
            )
            self._released = True


@contextlib.contextmanager
def _category_lock(
    category: str, target: Path, *, timeout: Optional[float] = None
) -> Iterator[None]:
    resolved_target = Path(target).expanduser().resolve(strict=False)
    lock_file = _lock_file_for(category, resolved_target)
    lock_cls = _select_lock_class()
    lock_timeout = _timeout_for(category, timeout)
    lock = lock_cls(
        str(lock_file),
        timeout=lock_timeout,
        mode=_lock_mode(),
        thread_local=False,
    )
    start = time.monotonic()
    try:
        lock.acquire(timeout=lock_timeout, poll_interval=_poll_interval())
    except Timeout:
        wait_ms = max((time.monotonic() - start) * 1000.0, 0.0)
        LOGGER.info(
            "lock-timeout category=%s wait_ms=%.3f lock_file=%s target=%s",
            category,
            wait_ms,
            lock_file,
            resolved_target,
        )
        _record_timeout(category, wait_ms)
        raise

    wait_ms = max((time.monotonic() - start) * 1000.0, 0.0)
    LOGGER.debug(
        "lock-acquired category=%s wait_ms=%.3f lock_file=%s target=%s",
        category,
        wait_ms,
        lock_file,
        resolved_target,
    )

    ctx = _LockContext(lock, category, wait_ms)
    try:
        yield None
    finally:
        ctx.release()


def manifest_lock(path: Path, *, timeout: Optional[float] = None) -> Iterator[None]:
    """Return a context manager guarding manifest JSONL writes."""

    return _category_lock("manifest", path, timeout=timeout)


def telemetry_lock(path: Path, *, timeout: Optional[float] = None) -> Iterator[None]:
    """Return a context manager guarding telemetry JSONL writes."""

    return _category_lock("telemetry", path, timeout=timeout)


def sqlite_lock(path: Path, *, timeout: Optional[float] = None) -> Iterator[None]:
    """Return a context manager guarding SQLite writes."""

    return _category_lock("sqlite", path, timeout=timeout)


def artifact_lock(path: Path, *, timeout: Optional[float] = None) -> Iterator[None]:
    """Return a context manager guarding artifact writes and promotions."""

    return _category_lock("artifact", path, timeout=timeout)


def summary_lock(path: Path, *, timeout: Optional[float] = None) -> Iterator[None]:
    """Return a context manager guarding summary and report writes."""

    return _category_lock("summary", path, timeout=timeout)


def lock_metrics_snapshot(*, reset: bool = False) -> Dict[str, Dict[str, Union[int, float]]]:
    """Return a snapshot of collected lock metrics, optionally clearing them."""

    with _metrics_guard:
        snapshot: Dict[str, Dict[str, float]] = {}
        for category, metrics in _metrics.items():
            summary = {
                "acquire_total": metrics.acquire_total,
                "timeout_total": metrics.timeout_total,
                "wait_ms_sum": metrics.wait_ms_sum,
                "wait_ms_p95": _p95(metrics.wait_ms_samples),
            }
            if metrics.hold_ms_samples:
                summary["hold_ms_sum"] = metrics.hold_ms_sum
                summary["hold_ms_p95"] = _p95(metrics.hold_ms_samples)
            snapshot[category] = summary
        if reset:
            _metrics.clear()
        return snapshot
