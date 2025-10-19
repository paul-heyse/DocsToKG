"""Concurrency utilities shared across DocParsing stages."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import time
from pathlib import Path
from typing import Iterator, Optional

from DocsToKG.DocParsing.logging import get_logger, log_event

__all__ = [
    "acquire_lock",
    "find_free_port",
    "set_spawn_or_warn",
]


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
                existing_pid = _read_lock_owner(lock_path)

                if existing_pid and not _pid_is_running(existing_pid):
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue

                if time.time() - start > timeout:
                    raise TimeoutError(f"Could not acquire lock on {path} after {timeout}s")

                time.sleep(0.1)
                lock_dir.mkdir(parents=True, exist_ok=True)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(owning_pid)
                lock_file.flush()
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


def _read_lock_owner(lock_path: Path) -> Optional[int]:
    """Read the PID stored in ``lock_path`` if available."""

    try:
        pid_text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not pid_text:
        return None

    try:
        return int(pid_text)
    except ValueError:
        return None


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


def find_free_port(start: int = 8000, span: int = 32) -> int:
    """Locate an available TCP port on localhost within a range."""

    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port

    logger = get_logger(__name__)
    log_event(
        logger,
        "warning",
        "Port scan exhausted",
        stage="core",
        doc_id="__system__",
        input_hash=None,
        error_code="PORT_SCAN_EXHAUSTED",
        start=start,
        span=span,
        action="ephemeral_port",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
