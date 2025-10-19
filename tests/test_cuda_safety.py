# === NAVMAP v1 ===
# {
#   "module": "tests.test_cuda_safety",
#   "purpose": "Pytest coverage for cuda safety scenarios",
#   "sections": [
#     {
#       "id": "reset-start-method",
#       "name": "reset_start_method",
#       "anchor": "function-reset-start-method",
#       "kind": "function"
#     },
#     {
#       "id": "dummyfuture",
#       "name": "_DummyFuture",
#       "anchor": "class-dummyfuture",
#       "kind": "class"
#     },
#     {
#       "id": "dummyexecutor",
#       "name": "_DummyExecutor",
#       "anchor": "class-dummyexecutor",
#       "kind": "class"
#     },
#     {
#       "id": "dummy-as-completed",
#       "name": "_dummy_as_completed",
#       "anchor": "function-dummy-as-completed",
#       "kind": "function"
#     },
#     {
#       "id": "dummytqdm",
#       "name": "_DummyTqdm",
#       "anchor": "class-dummytqdm",
#       "kind": "class"
#     },
#     {
#       "id": "stub-main-setup",
#       "name": "_stub_main_setup",
#       "anchor": "function-stub-main-setup",
#       "kind": "function"
#     },
#     {
#       "id": "import-pdf-module",
#       "name": "_import_pdf_module",
#       "anchor": "function-import-pdf-module",
#       "kind": "function"
#     },
#     {
#       "id": "test-spawn-enforced-on-main",
#       "name": "test_spawn_enforced_on_main",
#       "anchor": "function-test-spawn-enforced-on-main",
#       "kind": "function"
#     },
#     {
#       "id": "test-spawn-prevents-cuda-reinitialization",
#       "name": "test_spawn_prevents_cuda_reinitialization",
#       "anchor": "function-test-spawn-prevents-cuda-reinitialization",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

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


# --- Helper Functions ---


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


def _stub_main_setup(patcher, module, tmp_path, list_return: List) -> SimpleNamespace:
    """Prepare patched environment for invoking the PDF conversion main()."""

    args = module.pdf_parse_args([])
    args.data_root = None
    args.input = tmp_path / "input"
    args.output = tmp_path / "output"
    args.model = str(tmp_path / "model")
    args.served_model_names = ("granite-docling-258M",)
    args.workers = 1
    module.DEFAULT_WORKERS = 1
    module.PROFILE_PRESETS.setdefault("", {})["workers"] = 1

    # Mark input and output as explicit CLI overrides to prevent defaults from overriding them
    from DocsToKG.DocParsing.config import annotate_cli_overrides

    explicit_keys = [name for name in vars(args) if not name.startswith("_")]
    annotate_cli_overrides(args, explicit=explicit_keys, defaults={})

    patcher.setattr(module, "ensure_vllm", lambda *_a, **_k: (module.PREFERRED_PORT, None, False))
    patcher.setattr(module, "start_vllm", lambda *_a, **_k: SimpleNamespace(poll=lambda: None))
    patcher.setattr(module, "wait_for_vllm", lambda *_a, **_k: ["stub-model"])
    patcher.setattr(module, "validate_served_models", lambda *_a, **_k: None)
    patcher.setattr(module, "stop_vllm", lambda *_a, **_k: None)
    patcher.setattr(module, "list_pdfs", lambda _dir: iter(list_return))
    patcher.setattr(module, "manifest_append", lambda *a, **k: None)
    patcher.setattr(module, "ProcessPoolExecutor", _DummyExecutor)
    patcher.setattr(module, "as_completed", _dummy_as_completed)
    patcher.setattr(module, "tqdm", lambda *a, **k: _DummyTqdm())
    return args


def _import_pdf_module(patcher):
    """Import the PDF conversion script with external deps stubbed."""

    # Create a proper mock for requests that returns JSON-serializable values
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.text = "test"
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"data": [{"id": "test-model"}]}

    mock_session = mock.MagicMock()
    mock_session.get.return_value = mock_response

    mock_requests = mock.MagicMock()
    mock_requests.Session.return_value = mock_session
    import requests as real_requests

    mock_requests.RequestException = real_requests.RequestException
    mock_requests.exceptions = real_requests.exceptions

    patcher.setitem(sys.modules, "requests", mock_requests)

    class _TqdmStub:
        def __call__(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *_args, **_kwargs):
            return None

    patcher.setitem(sys.modules, "tqdm", mock.MagicMock(tqdm=_TqdmStub()))
    module = importlib.import_module("DocsToKG.DocParsing.doctags")
    return importlib.reload(module)


# --- Test Cases ---


def test_spawn_enforced_on_main(patcher, tmp_path, reset_start_method):
    """Calling main() must force the multiprocessing start method to spawn."""

    module = _import_pdf_module(patcher)

    args = _stub_main_setup(patcher, module, tmp_path, list_return=[])

    module.pdf_main(args)

    assert mp.get_start_method() == "spawn"


def test_spawn_prevents_cuda_reinitialization(patcher, tmp_path, reset_start_method):
    """Mock CUDA work should not hit fork-based reinitialization errors under spawn."""

    module = _import_pdf_module(patcher)

    pdf_dir = tmp_path / "input"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    args = _stub_main_setup(patcher, module, tmp_path, list_return=[pdf_path])

    calls = []

    def _fake_convert(task):
        calls.append(task)
        return module.PdfConversionResult(
            doc_id=task.doc_id,
            status="success",
            duration_s=0.1,
            input_path=str(task.pdf_path),
            input_hash=task.input_hash,
            output_path=str(task.output_path),
        )

    patcher.setattr(module, "pdf_convert_one", _fake_convert)

    module.pdf_main(args)

    assert mp.get_start_method() == "spawn"
    assert len(calls) == 1
