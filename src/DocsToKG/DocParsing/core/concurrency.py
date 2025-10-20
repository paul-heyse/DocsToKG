"""Process-safety helpers for DocParsing pipelines.

Chunking and embedding stages parallelise work across processes and threads,
so they need lightweight primitives that keep manifests and network resources
safe. This module provides advisory lock management, portable multiprocessing
spawn controls, and free-port discovery routines that help the CLI coordinate
Docling/vLLM workers without relying on heavyweight dependencies.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import socket
import time
from pathlib import Path
from typing import Iterator, Optional

from DocsToKG.DocParsing.logging import get_logger, log_event

__all__ = [
    "acquire_lock",
    "ReservedPort",
    "find_free_port",
    "set_spawn_or_warn",
]


LOGGER = get_logger(__name__, base_fields={"stage": "core"})
_STALE_LOCK_JITTER_RANGE = (0.01, 0.05)


@contextlib.contextmanager
def acquire_lock(path: Path, timeout: float = 60.0) -> Iterator[bool]:
    """Acquire an advisory lock using ``.lock`` sentinel files."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_dir = lock_path.parent
    start = time.time()
    lock_dir.mkdir(parents=True, exist_ok=True)
    owning_pid = str(os.getpid())
    acquired = False

    try:
        while True:
            try:
                fd = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                existing_pid, raw_pid = _read_lock_owner(lock_path)

                if raw_pid == "":
                    try:
                        age = time.time() - lock_path.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    if age < 0.1:
                        time.sleep(0.01)
                        continue
                    _evict_stale_lock(
                        lock_path,
                        reason="invalid-pid",
                        raw_pid=raw_pid,
                    )
                    continue

                if raw_pid is not None and existing_pid is None:
                    _evict_stale_lock(
                        lock_path,
                        reason="invalid-pid",
                        raw_pid=raw_pid,
                    )
                    continue

                if existing_pid is not None and not _pid_is_running(existing_pid):
                    _evict_stale_lock(
                        lock_path,
                        reason="stale-pid",
                        raw_pid=raw_pid,
                    )
                    continue

                if time.time() - start > timeout:
                    raise TimeoutError(f"Could not acquire lock on {path} after {timeout}s")

                time.sleep(0.1)
                lock_dir.mkdir(parents=True, exist_ok=True)
                continue

            try:
                os.write(fd, f"{owning_pid}\n".encode("utf-8"))
            finally:
                os.close(fd)
            acquired = True
            break

        yield True
    finally:
        if acquired:
            try:
                pid_text = lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                pid_text = ""

            if pid_text == owning_pid:
                lock_path.unlink(missing_ok=True)


def _pid_is_running(pid: int) -> bool:
    """Return ``True`` if a process with the given PID appears to be alive."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - platform specific
        return True
    except OSError:  # pragma: no cover - defensive guard
        return False
    return True


def _read_lock_owner(lock_path: Path) -> tuple[Optional[int], Optional[str]]:
    """Read the PID stored in ``lock_path`` and return both the parsed and raw forms."""

    try:
        pid_text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None, None

    if not pid_text:
        return None, ""

    try:
        return int(pid_text), pid_text
    except ValueError:
        return None, pid_text


def _evict_stale_lock(
    lock_path: Path,
    *,
    reason: str,
    raw_pid: Optional[str],
) -> None:
    """Remove a stale lock file after a brief jitter to avoid thundering herds."""

    if reason == "invalid-pid":
        log_event(
            LOGGER,
            "warning",
            "Lock file contained invalid PID; treating as stale.",
            doc_id="__system__",
            input_hash=None,
            error_code="LOCK_INVALID_PID",
            lock_path=str(lock_path),
            raw_pid=raw_pid or "",
        )

    jitter = random.uniform(*_STALE_LOCK_JITTER_RANGE)
    time.sleep(jitter)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


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
