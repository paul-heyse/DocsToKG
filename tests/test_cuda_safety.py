"""Tests for CUDA safety guarantees in DocTags conversion scripts."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import sys
from types import SimpleNamespace
from typing import Iterable, Iterator, List
from unittest import mock

import pytest


@pytest.fixture()
def reset_start_method() -> Iterator[None]:
    """Restore the multiprocessing start method after each test."""

    original = mp.get_start_method(allow_none=True)
    mp.set_start_method("fork", force=True)
    try:
        yield
    finally:
        restore = original or "fork"
        mp.set_start_method(restore, force=True)


class _DummyFuture:
    """Synchronously executed future used to stub ProcessPoolExecutor."""

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _DummyExecutor:
    """Context manager that mimics ProcessPoolExecutor sequentially."""

    def __init__(self, *_, **__):
        self.submitted: List[_DummyFuture] = []

    def __enter__(self) -> "_DummyExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
        return None

    def submit(self, fn, arg):
        future = _DummyFuture(fn(arg))
        self.submitted.append(future)
        return future


def _dummy_as_completed(futures: Iterable[_DummyFuture]) -> Iterator[_DummyFuture]:
    """Yield futures immediately in submission order."""

    yield from futures


class _DummyTqdm:
    """Minimal tqdm replacement for deterministic unit tests."""

    def __init__(self, *_, **__):
        pass

    def __enter__(self) -> "_DummyTqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
        return None

    def update(self, *_args, **_kwargs) -> None:
        return None


def _stub_main_setup(monkeypatch, module, tmp_path, list_return: List) -> SimpleNamespace:
    """Prepare patched environment for invoking the PDF conversion main()."""

    args = SimpleNamespace(
        data_root=None,
        input=tmp_path / "input",
        output=tmp_path / "output",
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "ensure_vllm", lambda *_a, **_k: (module.PREFERRED_PORT, None, False))
    monkeypatch.setattr(module, "stop_vllm", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "list_pdfs", lambda _dir: list_return)
    monkeypatch.setattr(module, "ProcessPoolExecutor", _DummyExecutor)
    monkeypatch.setattr(module, "as_completed", _dummy_as_completed)
    monkeypatch.setattr(module, "tqdm", lambda *a, **k: _DummyTqdm())
    return args


def _import_pdf_module(monkeypatch):
    """Import the PDF conversion script with external deps stubbed."""

    monkeypatch.setitem(sys.modules, "requests", mock.MagicMock())
    fake_tqdm = mock.MagicMock()
    fake_tqdm.tqdm = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    module = importlib.import_module(
        "DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug"
    )
    return importlib.reload(module)


def test_spawn_enforced_on_main(monkeypatch, tmp_path, reset_start_method):
    """Calling main() must force the multiprocessing start method to spawn."""

    module = _import_pdf_module(monkeypatch)

    _stub_main_setup(monkeypatch, module, tmp_path, list_return=[])

    module.main()

    assert mp.get_start_method() == "spawn"


def test_spawn_prevents_cuda_reinitialization(monkeypatch, tmp_path, reset_start_method):
    """Mock CUDA work should not hit fork-based reinitialization errors under spawn."""

    module = _import_pdf_module(monkeypatch)

    pdf_dir = tmp_path / "input"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    _stub_main_setup(monkeypatch, module, tmp_path, list_return=[pdf_path])

    calls = []

    def _fake_convert(task):
        calls.append(task)
        start_method = mp.get_start_method()
        if start_method != "spawn":
            raise RuntimeError("Cannot re-initialize CUDA in forked subprocess")
        return task[0].name, "ok"

    monkeypatch.setattr(module, "convert_one", _fake_convert)

    module.main()

    assert mp.get_start_method() == "spawn"
    assert len(calls) == 1

