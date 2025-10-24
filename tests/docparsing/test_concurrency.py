"""Tests for DocsToKG.DocParsing.core.concurrency utilities."""

from __future__ import annotations

import importlib.util
import logging
import threading
import sys
import types
import time
from pathlib import Path

import pytest

filelock_stub = types.ModuleType("filelock")


class _Timeout(Exception):
    """Stub Timeout exception compatible with filelock.Timeout."""


class _FileLock:
    """Lightweight in-memory lock emulating filelock.FileLock semantics."""

    _locks: dict[str, threading.Lock] = {}
    _registry_lock = threading.Lock()

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock: threading.Lock | None = None

    def acquire(self, timeout: float | None = None) -> bool:
        with _FileLock._registry_lock:
            shared_lock = _FileLock._locks.setdefault(self._path, threading.Lock())

        if timeout is None:
            shared_lock.acquire()
            self._lock = shared_lock
            return True

        deadline = time.monotonic() + timeout
        while True:
            if shared_lock.acquire(blocking=False):
                self._lock = shared_lock
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Timeout()
            time.sleep(min(0.01, max(remaining, 0.001)))

    def release(self) -> None:
        if self._lock is None or not self._lock.locked():
            raise RuntimeError("Lock not acquired")
        self._lock.release()


filelock_stub.FileLock = _FileLock
filelock_stub.Timeout = _Timeout
sys.modules["filelock"] = filelock_stub

from filelock import FileLock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DOCSTOKG_DIR = SRC_DIR / "DocsToKG"
DOCPARSING_DIR = DOCSTOKG_DIR / "DocParsing"
CORE_DIR = DOCPARSING_DIR / "core"

docstokg_pkg = sys.modules.setdefault("DocsToKG", types.ModuleType("DocsToKG"))
docstokg_pkg.__path__ = [str(DOCSTOKG_DIR)]
docparsing_pkg = sys.modules.setdefault(
    "DocsToKG.DocParsing", types.ModuleType("DocsToKG.DocParsing")
)
docparsing_pkg.__path__ = [str(DOCPARSING_DIR)]
core_pkg = sys.modules.setdefault(
    "DocsToKG.DocParsing.core", types.ModuleType("DocsToKG.DocParsing.core")
)
core_pkg.__path__ = [str(CORE_DIR)]

logging_stub = types.ModuleType("DocsToKG.DocParsing.logging")


def _get_logger(name: str, base_fields: dict[str, object] | None = None) -> logging.Logger:
    return logging.getLogger(name)


def _log_event(logger: logging.Logger, level: str, message: str, **_: object) -> None:
    handler = getattr(logger, level, None)
    if callable(handler):
        handler(message)


logging_stub.get_logger = _get_logger
logging_stub.log_event = _log_event
sys.modules["DocsToKG.DocParsing.logging"] = logging_stub

spec = importlib.util.spec_from_file_location(
    "DocsToKG.DocParsing.core.concurrency", CORE_DIR / "concurrency.py"
)
assert spec and spec.loader is not None
concurrency = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = concurrency
spec.loader.exec_module(concurrency)
safe_write = concurrency.safe_write


def test_safe_write_atomic_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """safe_write should honour atomic temp-file replacements when enabled."""

    monkeypatch.setenv("DOCSTOKG_ATOMIC_WRITES", "1")
    target = tmp_path / "payload.txt"
    captured: list[Path] = []

    def _write(target_path: Path) -> None:
        captured.append(target_path)
        target_path.write_text("hello world", encoding="utf-8")

    wrote = safe_write(target, _write, skip_if_exists=False)
    assert wrote
    assert target.read_text(encoding="utf-8") == "hello world"
    assert captured and captured[0] != target
    leftovers = list(target.parent.glob(f".{target.name}.*"))
    assert leftovers == []


def test_safe_write_atomic_failure_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Temporary files should be cleaned up when atomic writes fail."""

    monkeypatch.setenv("DOCSTOKG_ATOMIC_WRITES", "1")
    target = tmp_path / "failure.txt"
    emitted: list[Path] = []

    def _write_then_fail(target_path: Path) -> None:
        emitted.append(target_path)
        target_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        safe_write(target, _write_then_fail, skip_if_exists=False)

    assert not target.exists()
    assert emitted and emitted[0] != target
    assert list(target.parent.glob(f".{target.name}.*")) == []


def test_safe_write_retain_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lock files remain on disk when retention is enabled."""

    monkeypatch.setenv("DOCSTOKG_RETAIN_LOCK_FILES", "1")
    target = tmp_path / "lock.txt"

    def _write_initial(path: Path) -> None:
        path.write_text("first", encoding="utf-8")

    wrote = safe_write(target, _write_initial, skip_if_exists=False, atomic=False)
    assert wrote

    lock_path = target.with_suffix(target.suffix + ".lock")
    assert lock_path.exists()

    def _write_second(path: Path) -> None:
        path.write_text("second", encoding="utf-8")

    wrote = safe_write(target, _write_second, skip_if_exists=False, atomic=False)
    assert wrote
    assert lock_path.exists()
    assert target.read_text(encoding="utf-8") == "second"


def test_safe_write_atomic_requires_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Atomic mode rejects write callbacks without a path argument."""

    monkeypatch.setenv("DOCSTOKG_ATOMIC_WRITES", "1")
    target = tmp_path / "bad.txt"

    def _write_without_path() -> None:  # pragma: no cover - intentional failure path
        target.write_text("oops", encoding="utf-8")

    with pytest.raises(TypeError):
        safe_write(target, _write_without_path, skip_if_exists=False, atomic=True)


def test_safe_write_logs_contention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lock acquisition waits should emit structured contention events."""

    events: list[tuple[str, str, dict[str, object]]] = []

    def _capture_event(
        logger: logging.Logger, level: str, message: str, **fields: object
    ) -> None:
        events.append((level, message, dict(fields)))

    monkeypatch.setattr(concurrency, "log_event", _capture_event)

    target = tmp_path / "contended.txt"
    lock_path = concurrency._lock_path_for(target)
    file_lock = FileLock(str(lock_path))
    file_lock.acquire(timeout=1.0)

    def _release_later() -> None:
        time.sleep(0.05)
        file_lock.release()

    releaser = threading.Thread(target=_release_later)
    releaser.start()

    try:
        wrote = safe_write(
            target,
            lambda path: path.write_text("data", encoding="utf-8"),
            skip_if_exists=False,
            atomic=False,
        )
    finally:
        releaser.join()

    assert wrote
    assert events, "Expected contention log events"
    level, message, fields = events[0]
    assert level == "debug"
    assert "acquired" in message.lower()
    assert fields["lock_path"] == str(lock_path)
    assert fields["lock_target"] == str(target)
    assert fields["timeout_seconds"] == 60.0
    assert fields["wait_seconds"] >= concurrency._CONTENTED_WAIT_THRESHOLD_SECONDS


def test_safe_write_no_contention_skips_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Immediate lock acquisition should not emit contention events."""

    events: list[tuple[str, str, dict[str, object]]] = []

    def _capture_event(
        logger: logging.Logger, level: str, message: str, **fields: object
    ) -> None:
        events.append((level, message, dict(fields)))

    monkeypatch.setattr(concurrency, "log_event", _capture_event)

    target = tmp_path / "no_contention.txt"

    wrote = safe_write(
        target,
        lambda path: path.write_text("clean", encoding="utf-8"),
        skip_if_exists=False,
        atomic=False,
    )

    assert wrote
    assert events == []
