# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_doctags_cli_errors",
#   "purpose": "Ensure DocTags Typer command surfaces the same validation messages and exit codes.",
#   "sections": [
#     {"id": "module-overview", "name": "Module Overview", "anchor": "module-overview"},
#     {"id": "stubs", "name": "Runtime Stub Helpers", "anchor": "_create_cli_stubs"},
#     {"id": "tests-missing-input", "name": "Missing Input Checks", "anchor": "test_missing_input_directory_triggers_error"},
#     {"id": "tests-conflicting-mode", "name": "Conflicting Mode Checks", "anchor": "test_mode_auto_conflicting_sources"},
#     {"id": "tests-arg-forwarding", "name": "Arg Forwarding", "anchor": "test_respects_legacy_option_names"}
#   ]
# }
# === /NAVMAP ===

"""Exercise DocTags CLI validation errors and user-facing messaging.

The DocTags CLI provides structured hints when operators supply bad input.
These tests stub optional dependencies to simulate misconfiguration and then
assert that error messages, option names, and exit codes remain descriptive,
protecting the UX around CLI validation.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import PatchManager


def _create_cli_stubs(tmp_path: Path) -> Path:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "yaml.py").write_text("def safe_load(raw):\n    return {}\n", encoding="utf-8")

    return stub_dir


def _prepare_runtime(tmp_path: Path, patcher: PatchManager) -> None:
    stub_dir = _create_cli_stubs(tmp_path)
    fake_deps = Path(__file__).resolve().parent / "fake_deps"

    patcher.syspath_prepend(str(fake_deps))
    patcher.syspath_prepend(str(stub_dir))

    ontology_io_module = types.ModuleType("DocsToKG.OntologyDownload.io")
    ontology_io_module.mask_sensitive_data = lambda value: value
    ontology_io_module.sanitize_filename = lambda path: str(path)
    patcher.setitem(sys.modules, "DocsToKG.OntologyDownload.io", ontology_io_module)

    logging_utils_module = types.ModuleType("DocsToKG.OntologyDownload.logging_utils")

    class _StubJSONFormatter:
        def format(self, record):  # pragma: no cover - simple passthrough
            return getattr(record, "message", "")

    logging_utils_module.JSONFormatter = _StubJSONFormatter
    logging_utils_module.mask_sensitive_data = ontology_io_module.mask_sensitive_data
    logging_utils_module.sanitize_filename = ontology_io_module.sanitize_filename
    patcher.setitem(sys.modules, "DocsToKG.OntologyDownload.logging_utils", logging_utils_module)

    patcher.setitem(
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
    patcher.setitem(sys.modules, "requests", requests_module)
    patcher.setitem(sys.modules, "requests.adapters", adapters_module)

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
    patcher.setitem(sys.modules, "urllib3", urllib3_module)
    patcher.setitem(sys.modules, "urllib3.util", urllib3_util_module)
    patcher.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_module)


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
    patcher: PatchManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """docparse doctags reports friendly errors when defaults are ambiguous."""

    _prepare_runtime(tmp_path, patcher)

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

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    core_cli = importlib.import_module("DocsToKG.DocParsing.core.cli")

    exit_code = core_cli._run_stage(core_cli._execute_doctags, [])
    assert exit_code == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    expected_message = message_factory(html_dir.resolve(), pdf_dir.resolve())
    expected = f"[doctags] --mode: {expected_message}. Hint: {hint}"
    assert captured.err.strip() == expected


def test_docparse_doctags_auto_detection_with_mixed_input(
    tmp_path: Path,
    patcher: PatchManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """docparse doctags surfaces formatted errors for mixed explicit inputs."""

    _prepare_runtime(tmp_path, patcher)

    data_root = tmp_path / "data-root"
    input_dir = tmp_path / "sources"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "page.html").write_text("<html></html>", encoding="utf-8")
    (input_dir / "page.pdf").write_bytes(b"%PDF-1.4")

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    core_cli = importlib.import_module("DocsToKG.DocParsing.core.cli")

    exit_code = core_cli._run_stage(core_cli._execute_doctags, ["--input", str(input_dir)])
    assert exit_code == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    expected = (
        f"[doctags] --mode: Cannot auto-detect mode in {input_dir.resolve()}: "
        "found both PDF and HTML files. Hint: Specify --mode html or --mode pdf to override auto-detection"
    )
    assert captured.err.strip() == expected


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
def test_doctags_cli_structured_auto_detection_errors(
    tmp_path: Path,
    layout: str,
    message_factory,
    hint: str,
    patcher: PatchManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`core.cli.doctags` emits structured errors for auto-detection failures."""

    _prepare_runtime(tmp_path, patcher)

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

    patcher.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    core_cli = importlib.import_module("DocsToKG.DocParsing.core.cli")

    exit_code = core_cli._execute_doctags([])
    assert exit_code == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    expected_message = message_factory(html_dir.resolve(), pdf_dir.resolve())
    expected = f"[doctags] --mode: {expected_message}. Hint: {hint}"
    assert captured.err.strip() == expected
