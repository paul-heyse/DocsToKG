# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.concurrency",
#   "purpose": "Process-safety helpers for DocParsing pipelines.",
#   "sections": [
#     {
#       "id": "lock-path-for",
#       "name": "_lock_path_for",
#       "anchor": "function-lock-path-for",
#       "kind": "function"
#     },
#     {
#       "id": "acquire-lock",
#       "name": "_acquire_lock",
#       "anchor": "function-acquire-lock",
#       "kind": "function"
#     },
#     {
#       "id": "safe-write",
#       "name": "safe_write",
#       "anchor": "function-safe-write",
#       "kind": "function"
#     },
#     {
#       "id": "set-spawn-or-warn",
#       "name": "set_spawn_or_warn",
#       "anchor": "function-set-spawn-or-warn",
#       "kind": "function"
#     },
#     {
#       "id": "reservedport",
#       "name": "ReservedPort",
#       "anchor": "class-reservedport",
#       "kind": "class"
#     },
#     {
#       "id": "bind-reserved-socket",
#       "name": "_bind_reserved_socket",
#       "anchor": "function-bind-reserved-socket",
#       "kind": "function"
#     },
#     {
#       "id": "find-free-port",
#       "name": "find_free_port",
#       "anchor": "function-find-free-port",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Process-safety helpers for DocParsing pipelines.

Chunking and embedding stages parallelise work across processes and threads,
so they need lightweight primitives that keep manifests and network resources
safe. This module provides:

- safe_write(): Public API for atomic file writes with process-safe FileLock
- Portable multiprocessing spawn controls via set_spawn_or_warn()
- Free-port discovery routines via find_free_port()
- Reserved port enumeration via ReservedPort

These helpers coordinate Docling/vLLM workers without relying on heavyweight
dependencies. The safe_write() function is the recommended way to atomically
write files when multiple processes may access them concurrently.

Example:
    .. code-block:: python

        from DocsToKG.DocParsing.core import safe_write
        from pathlib import Path

        # Atomically write a file with process-safe locking
        wrote = safe_write(
            Path("output.json"),
            lambda: save_json_to_output(),
            timeout=60.0,
            skip_if_exists=True,
        )
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import socket
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from DocsToKG.DocParsing.logging import get_logger, log_event

__all__ = [
    "ReservedPort",
    "find_free_port",
    "set_spawn_or_warn",
    "safe_write",
    "_acquire_lock",
]


LOGGER = get_logger(__name__, base_fields={"stage": "core"})


_ATOMIC_WRITES_ENV = "DOCSTOKG_ATOMIC_WRITES"
_RETAIN_LOCKS_ENV = "DOCSTOKG_RETAIN_LOCK_FILES"
_TMP_SUFFIX = ".tmp"


def _lock_path_for(path: Path) -> Path:
    """Return the canonical lock path for a given payload file."""

    return path.with_suffix(path.suffix + ".lock")


def _parse_bool_env(value: str | None, default: bool) -> bool:
    """Interpret ``value`` as a boolean flag."""

    if value is None:
        return default
    text = value.strip().lower()
    if not text:
        return default
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _resolve_atomic_flag(explicit: bool | None) -> bool:
    """Return whether safe_write should use atomic temp-file replacement."""

    if explicit is not None:
        return bool(explicit)
    return _parse_bool_env(os.getenv(_ATOMIC_WRITES_ENV), False)


def _resolve_retain_lock(explicit: bool | None) -> bool:
    """Return whether ``.lock`` sentinels should remain on disk."""

    if explicit is not None:
        return bool(explicit)
    return _parse_bool_env(os.getenv(_RETAIN_LOCKS_ENV), False)


def _fsync_path(path: Path) -> None:
    """Flush ``path`` to disk if it exists."""

    if not path.exists():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(directory: Path) -> None:
    """Flush directory metadata for ``directory``."""

    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_fn_accepts_path(write_fn: Callable[..., Any]) -> bool:
    """Return True when ``write_fn`` can be invoked with a path argument."""

    try:
        signature = inspect.signature(write_fn)
    except (TypeError, ValueError):
        # Builtins or C extensions: assume they accept the path argument.
        return True

    parameters = tuple(signature.parameters.values())
    varargs = any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in parameters)
    if varargs:
        return True

    positional = [
        param
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    required_positional = [param for param in positional if param.default is inspect._empty]
    if len(required_positional) > 1:
        raise TypeError("safe_write write_fn must accept at most one positional argument")

    if positional:
        return True

    if any(
        param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect._empty
        for param in parameters
    ):
        raise TypeError("safe_write write_fn cannot require keyword-only arguments")

    return False


def _invoke_write_fn(
    write_fn: Callable[..., None],
    target_path: Path,
    *,
    require_path: bool,
) -> None:
    """Invoke ``write_fn`` respecting whether a path argument is required."""

    accepts_path = _write_fn_accepts_path(write_fn)
    if require_path and not accepts_path:
        raise TypeError(
            "safe_write write_fn must accept a path argument when atomic writes are enabled"
        )

    if accepts_path:
        write_fn(target_path)
        return

    write_fn()


def _atomic_replace(path: Path, write_fn: Callable[..., None]) -> None:
    """Write to a temporary file then atomically replace ``path``."""

    fd: int | None = None
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=_TMP_SUFFIX
        )
        os.close(fd)
        fd = None
        tmp_path = Path(tmp_name)
        _invoke_write_fn(write_fn, tmp_path, require_path=True)
        _fsync_path(tmp_path)
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


@contextlib.contextmanager
def _acquire_lock(
    path: Path,
    *,
    timeout: float = 60.0,
    retain_lock_file: bool | None = None,
) -> Iterator[bool]:
    """Acquire a process-safe lock for ``path``.

    This context manager exists for backwards compatibility with legacy code
    that previously relied on ``_acquire_lock``. It delegates locking to the
    same FileLock-based implementation used by :func:`safe_write` and yields a
    simple boolean placeholder to preserve historical semantics. When
    ``retain_lock_file`` or ``DOCSTOKG_RETAIN_LOCK_FILES`` is truthy, the lock
    sentinel remains on disk after releasing the FileLock, allowing callers to
    inspect recent lock usage.
    """

    lock_path = _lock_path_for(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    file_lock = FileLock(str(lock_path))

    try:
        file_lock.acquire(timeout=timeout)
    except Timeout as exc:
        raise TimeoutError(f"Could not acquire lock on {path} after {timeout}s") from exc

    try:
        yield True
    finally:
        with contextlib.suppress(RuntimeError):
            file_lock.release()
        should_retain = _resolve_retain_lock(retain_lock_file)
        if should_retain:
            with contextlib.suppress(OSError):
                lock_path.touch(exist_ok=True)
        else:
            with contextlib.suppress(OSError):
                lock_path.unlink()


def safe_write(
    path: Path,
    write_fn: Callable[..., None],
    *,
    timeout: float = 60.0,
    skip_if_exists: bool = True,
    atomic: bool | None = None,
    retain_lock_file: bool | None = None,
) -> bool:
    """Atomically write a file with process-safe locking.

    Acquires a FileLock via :func:`_acquire_lock`, then executes ``write_fn``.
    Returns ``True`` if a write occurred and ``False`` when ``skip_if_exists``
    short-circuits due to a pre-existing file.

    Args:
        path: File path to write
        write_fn: Callable that performs the write (e.g., lambda: file.save())
        timeout: FileLock timeout in seconds
        skip_if_exists: If True, skip write if file already exists
        atomic: When True (or configured via ``DOCSTOKG_ATOMIC_WRITES``), write to a
            temporary file and atomically replace ``path`` using ``os.replace``.
        retain_lock_file: When True (or configured via ``DOCSTOKG_RETAIN_LOCK_FILES``),
            leave the ``.lock`` sentinel on disk after releasing the FileLock for
            post-mortem inspection.

    Returns:
        True if write occurred, False otherwise

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
        TypeError: If ``write_fn`` requires unsupported parameters
    """

    use_atomic = _resolve_atomic_flag(atomic)
    retain_lock = _resolve_retain_lock(retain_lock_file)

    with _acquire_lock(path, timeout=timeout, retain_lock_file=retain_lock):
        if skip_if_exists and path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        if use_atomic:
            _atomic_replace(path, write_fn)
        else:
            _invoke_write_fn(write_fn, path, require_path=False)
        return True


def set_spawn_or_warn(logger: logging.Logger | None = None) -> None:
    """Ensure the multiprocessing start method is set to ``spawn``."""

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
        if logger is not None:
            logger.debug("Multiprocessing start method set to 'spawn'")
        return
    except RuntimeError:
        current = mp.get_start_method(allow_none=True)
        if current == "spawn":
            if logger is not None:
                logger.debug("Multiprocessing start method already 'spawn'")
            return
        message = "Multiprocessing start method is %s; CUDA workloads require 'spawn'." % (
            current or "unset"
        )
        current_method = current or "unset"
        if logger is not None:
            logger.warning(message)
            structured_logger = get_logger(__name__)
        else:
            structured_logger = logging.getLogger(__name__)
        log_event(
            structured_logger,
            "warning",
            message,
            stage="core",
            doc_id="__system__",
            input_hash=None,
            error_code="MP_SPAWN_REQUIRED",
            current_method=current_method,
        )


class ReservedPort(contextlib.AbstractContextManager["ReservedPort"]):
    """Context manager representing a reserved TCP port."""

    def __init__(self, sock: socket.socket, host: str) -> None:
        """Initialize a reserved port context manager."""
        self._socket = sock
        self._host = host
        self._closed = False

    def __enter__(self) -> ReservedPort:
        """Enter the context manager and return self."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        """Exit the context manager and close the socket."""
        self.close()
        return None

    @property
    def socket(self) -> socket.socket:
        """Return the underlying socket that keeps the reservation alive."""

        return self._socket

    @property
    def port(self) -> int:
        """Return the port reserved by this context."""

        return self._socket.getsockname()[1]

    @property
    def host(self) -> str:
        """Return the host interface the reservation is bound to."""

        return self._host

    def close(self) -> None:
        """Release the reservation if it is currently held."""

        if not self._closed:
            try:
                self._socket.close()
            finally:
                self._closed = True


def _bind_reserved_socket(host: str, port: int) -> socket.socket | None:
    """Bind a socket to the specified host and port, returning None on failure."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.listen()
        return sock
    except OSError:
        sock.close()
        return None


def find_free_port(
    start: int = 8000,
    span: int = 32,
    *,
    host: str = "127.0.0.1",
    retry_interval: float = 0.05,
    max_attempts: int = 100,
) -> ReservedPort:
    """Reserve an available TCP port and return a context manager guarding it.

    The returned context manager keeps the port reserved by holding an open
    listening socket. Callers should release the reservation (either by exiting
    the context or invoking :meth:`ReservedPort.close`) only when the consumer
    of the port is ready to bind and accept connections.
    """

    if span <= 0:
        raise ValueError("span must be positive")

    attempts = 0
    while attempts < max_attempts:
        for port in range(start, start + span):
            sock = _bind_reserved_socket(host, port)
            if sock is not None:
                return ReservedPort(sock, host)
        attempts += 1
        time.sleep(retry_interval)

    logger = get_logger(__name__)
    log_event(
        logger,
        "warning",
        "Port scan exhausted; falling back to ephemeral port",
        stage="core",
        doc_id="__system__",
        input_hash=None,
        error_code="PORT_SCAN_EXHAUSTED",
        start=start,
        span=span,
        attempts=attempts,
        action="ephemeral_port",
    )
    fallback = _bind_reserved_socket(host, 0)
    if fallback is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to bind to an ephemeral port")
    return ReservedPort(fallback, host)
