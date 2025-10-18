"""DocTags CLI validation and error formatting behaviour."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _create_cli_stubs(tmp_path: Path) -> Path:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "yaml.py").write_text("def safe_load(raw):\n    return {}\n", encoding="utf-8")

    pydantic_core_dir = stub_dir / "pydantic_core"
    pydantic_core_dir.mkdir(parents=True, exist_ok=True)
    (pydantic_core_dir / "__init__.py").write_text(
        "class ValidationError(Exception):\n    pass\n",
        encoding="utf-8",
    )

    pydantic_settings_dir = stub_dir / "pydantic_settings"
    pydantic_settings_dir.mkdir(parents=True, exist_ok=True)
    (pydantic_settings_dir / "__init__.py").write_text(
        "class BaseSettings:\n    def __init__(self, **kwargs):\n        for key, value in kwargs.items():\n"
        "            setattr(self, key, value)\n\n"
        "class SettingsConfigDict(dict):\n    pass\n",
        encoding="utf-8",
    )

    return stub_dir


def _prepare_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_dir = _create_cli_stubs(tmp_path)
    fake_deps = Path(__file__).resolve().parent / "fake_deps"
    monkeypatch.syspath_prepend(str(fake_deps))
    monkeypatch.syspath_prepend(str(stub_dir))
    monkeypatch.setitem(
        sys.modules,
        "pooch",
        types.SimpleNamespace(
            HTTPDownloader=object,
            retrieve=lambda *args, **kwargs: Path(kwargs.get("path", "")),
        ),
    )
    requests_module = types.ModuleType("requests")
    class _RequestException(Exception):
        pass

    class _HTTPError(_RequestException):
        pass

    class _ConnectionError(_RequestException):
        pass

    class _Timeout(_RequestException):
        pass

    class _Session:
        pass

    adapters_module = types.ModuleType("requests.adapters")
    class _HTTPAdapter:
        pass

    adapters_module.HTTPAdapter = _HTTPAdapter
    requests_module.Session = _Session
    requests_module.RequestException = _RequestException
    requests_module.HTTPError = _HTTPError
    requests_module.ConnectionError = _ConnectionError
    requests_module.Timeout = _Timeout
    requests_module.adapters = adapters_module
    requests_module.exceptions = types.SimpleNamespace(SSLError=_RequestException)
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "requests.adapters", adapters_module)

    urllib3_module = types.ModuleType("urllib3")
    urllib3_util_module = types.ModuleType("urllib3.util")
    urllib3_retry_module = types.ModuleType("urllib3.util.retry")

    class _Retry:
        def __init__(self, *args, **kwargs):
            pass

        def new(self, **kwargs):  # pragma: no cover - compatibility helper
            return self

    urllib3_retry_module.Retry = _Retry
    urllib3_util_module.retry = urllib3_retry_module
    urllib3_module.util = urllib3_util_module
    monkeypatch.setitem(sys.modules, "urllib3", urllib3_module)
    monkeypatch.setitem(sys.modules, "urllib3.util", urllib3_util_module)
    monkeypatch.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_module)


@pytest.mark.parametrize(
    ("layout", "message_factory", "hint"),
    [
        (
            "mixed",
            lambda html_dir, pdf_dir: (
                "Cannot auto-detect mode: found HTML sources in "
                f"{html_dir} and PDF sources in {pdf_dir}"
            ),
            "Specify --mode html or --mode pdf to disambiguate the sources",
        ),
        (
            "missing",
            lambda html_dir, pdf_dir: (
                "Cannot auto-detect mode: expected HTML files in "
                f"{html_dir} or PDF files in {pdf_dir}"
            ),
            "Provide --input or set --mode html/--mode pdf explicitly",
        ),
    ],
)
def test_docparse_doctags_auto_detection_defaults(
    tmp_path: Path,
    layout: str,
    message_factory,
    hint: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """docparse doctags reports friendly errors when defaults are ambiguous."""

    _prepare_runtime(tmp_path, monkeypatch)

    data_root = tmp_path / "data-root"
    html_dir = data_root / "HTML"
    pdf_dir = data_root / "PDFs"

    if layout == "mixed":
        html_dir.mkdir(parents=True, exist_ok=True)
        (html_dir / "sample.html").write_text("<html></html>", encoding="utf-8")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / "sample.pdf").write_bytes(b"%PDF-1.4")
    elif layout == "missing":
        html_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
    else:  # pragma: no cover - defensive branch for unexpected parametrisations
        raise AssertionError(f"Unexpected layout {layout}")

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    core_cli = importlib.import_module("DocsToKG.DocParsing.core.cli")

    exit_code = core_cli._run_stage(core_cli.doctags, [])
    assert exit_code == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    expected_message = message_factory(html_dir.resolve(), pdf_dir.resolve())
    expected = f"[doctags] --mode: {expected_message}. Hint: {hint}"
    assert captured.err.strip() == expected


def test_docparse_doctags_auto_detection_with_mixed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """docparse doctags surfaces formatted errors for mixed explicit inputs."""

    _prepare_runtime(tmp_path, monkeypatch)

    data_root = tmp_path / "data-root"
    input_dir = tmp_path / "sources"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "page.html").write_text("<html></html>", encoding="utf-8")
    (input_dir / "page.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    core_cli = importlib.import_module("DocsToKG.DocParsing.core.cli")

    exit_code = core_cli._run_stage(core_cli.doctags, ["--input", str(input_dir)])
    assert exit_code == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    expected = (
        f"[doctags] --mode: Cannot auto-detect mode in {input_dir.resolve()}: "
        "found both PDF and HTML files. Hint: Specify --mode html or --mode pdf to override auto-detection"
    )
    assert captured.err.strip() == expected
