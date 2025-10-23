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
    from DocsToKG.DocParsing.core import safe_write
    from pathlib import Path

    # Atomically write a file with process-safe locking
    wrote = safe_write(
        Path("output.json"),
        lambda: save_json_to_output(),
        timeout=60.0,
        skip_if_exists=True
    )
"""

from __future__ import annotations

import contextlib
import logging
import socket
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

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


def _lock_path_for(path: Path) -> Path:
    """Return the canonical lock path for a given payload file."""

    return path.with_suffix(path.suffix + ".lock")


@contextlib.contextmanager
def _acquire_lock(path: Path, *, timeout: float = 60.0) -> Iterator[bool]:
    """Acquire a process-safe lock for ``path``.

    This context manager exists for backwards compatibility with legacy code
    that previously relied on ``_acquire_lock``. It delegates locking to the
    same FileLock-based implementation used by :func:`safe_write` and yields a
    simple boolean placeholder to preserve historical semantics.
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
        with contextlib.suppress(OSError):
            lock_path.unlink()


def safe_write(
    path: Path,
    write_fn: Callable[[], None],
    *,
    timeout: float = 60.0,
    skip_if_exists: bool = True,
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

    Returns:
        True if write occurred, False otherwise

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """

    with _acquire_lock(path, timeout=timeout):
        if skip_if_exists and path.exists():
            return False
        write_fn()
        return True


def set_spawn_or_warn(logger: Optional[logging.Logger] = None) -> None:
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

    def __enter__(self) -> "ReservedPort":
        """Enter the context manager and return self."""
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
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


def _bind_reserved_socket(host: str, port: int) -> Optional[socket.socket]:
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
