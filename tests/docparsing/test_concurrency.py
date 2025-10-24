"""Tests for DocsToKG.DocParsing.core.concurrency utilities."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

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
