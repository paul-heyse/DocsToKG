#!/usr/bin/env python3
"""Unit tests for legacy DocParsing script shims."""

from __future__ import annotations

import argparse
import sys
import types
import warnings
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stub_cli_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Provide a stub CLI module injected into ``sys.modules`` for tests."""

    fake_module = types.ModuleType("DocsToKG.DocParsing.cli.doctags_convert")
    fake_module.main = MagicMock(return_value=0)
    fake_module.build_parser = MagicMock(return_value=argparse.ArgumentParser())

    monkeypatch.setitem(
        sys.modules,
        "DocsToKG.DocParsing.cli.doctags_convert",
        fake_module,
    )
    import DocsToKG.DocParsing.cli as cli_pkg

    monkeypatch.setattr(cli_pkg, "doctags_convert", fake_module, raising=False)

    yield fake_module

    monkeypatch.delattr(cli_pkg, "doctags_convert", raising=False)


@contextmanager
def stub_legacy_dependencies() -> None:
    """Temporarily install stub modules required by legacy scripts."""

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda *args, **kwargs: []

    fake_requests = types.ModuleType("requests")
    fake_requests.get = MagicMock()
    fake_requests.Session = MagicMock(return_value=MagicMock())

    fake_packaging = types.ModuleType("packaging")
    fake_packaging_version = types.ModuleType("packaging.version")
    fake_packaging_version.Version = MagicMock()
    fake_packaging_version.InvalidVersion = MagicMock()
    fake_packaging.version = fake_packaging_version

    with patch.dict(
        sys.modules,
        {
            "tqdm": fake_tqdm,
            "requests": fake_requests,
            "packaging": fake_packaging,
            "packaging.version": fake_packaging_version,
        },
    ):
        yield


class TestHTMLShim:
    """Test suite for HTML converter shim."""

    def test_module_imports(self) -> None:
        """HTML shim module should import without errors."""
        from DocsToKG.DocParsing import run_docling_html_to_doctags_parallel

        assert hasattr(run_docling_html_to_doctags_parallel, "main")
        assert hasattr(run_docling_html_to_doctags_parallel, "build_parser")
        assert hasattr(run_docling_html_to_doctags_parallel, "parse_args")

    def test_main_callable(self) -> None:
        """HTML shim main() should be callable."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        assert callable(main)

    def test_emits_deprecation_warning_on_main(self, stub_cli_module: types.ModuleType) -> None:
        """HTML shim should emit DeprecationWarning when main() is called."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            main(["--help"])

        assert warning_list, "No warnings captured"
        assert issubclass(warning_list[0].category, DeprecationWarning)
        assert "deprecated" in str(warning_list[0].message).lower()

    def test_emits_deprecation_warning_on_build_parser(
        self, stub_cli_module: types.ModuleType
    ) -> None:
        """HTML shim should emit warning when build_parser() is called."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import (
            build_parser,
        )

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            build_parser()

        assert warning_list, "No warnings captured"
        assert issubclass(warning_list[0].category, DeprecationWarning)

    def test_forwards_to_unified_cli_with_html_mode(
        self, stub_cli_module: types.ModuleType
    ) -> None:
        """HTML shim should forward to unified CLI with --mode html."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        stub_cli_module.main.return_value = 42
        result = main(["--resume", "--workers", "4"])

        assert stub_cli_module.main.called, "Unified CLI main() not called"
        assert stub_cli_module.main.call_count == 1

        call_args = stub_cli_module.main.call_args.args[0]
        assert isinstance(call_args, list)
        assert "--mode" in call_args, "Missing --mode flag"
        assert "html" in call_args, "Missing html mode"
        assert "--resume" in call_args, "Missing --resume flag"
        assert "--workers" in call_args, "Missing --workers flag"
        assert result == 42

    def test_preserves_existing_mode_flag(self, stub_cli_module: types.ModuleType) -> None:
        """HTML shim should not duplicate --mode if already present."""
        from DocsToKG.DocParsing.run_docling_html_to_doctags_parallel import main

        main(["--mode", "pdf", "--resume"])

        call_args: list[str] = stub_cli_module.main.call_args.args[0]
        mode_indices = [idx for idx, token in enumerate(call_args) if token == "--mode"]
        assert len(mode_indices) == 1, f"--mode appears {len(mode_indices)} times (should be 1)"
        assert call_args[mode_indices[0] + 1] == "pdf"


class TestPDFShim:
    """Test suite for PDF converter shim."""

    def test_module_imports(self) -> None:
        """PDF shim module should import without errors."""
        from DocsToKG.DocParsing import run_docling_parallel_with_vllm_debug

        assert hasattr(run_docling_parallel_with_vllm_debug, "main")
        assert hasattr(run_docling_parallel_with_vllm_debug, "build_parser")
        assert hasattr(run_docling_parallel_with_vllm_debug, "parse_args")

    def test_main_callable(self) -> None:
        """PDF shim main() should be callable."""
        from DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug import main

        assert callable(main)

    def test_emits_deprecation_warning(self, stub_cli_module: types.ModuleType) -> None:
        """PDF shim should emit DeprecationWarning."""
        from DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug import main

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            main(["--help"])

        assert warning_list, "No warnings captured"
        assert issubclass(warning_list[0].category, DeprecationWarning)
        assert "deprecated" in str(warning_list[0].message).lower()

    def test_forwards_to_unified_cli_with_pdf_mode(self, stub_cli_module: types.ModuleType) -> None:
        """PDF shim should forward to unified CLI with --mode pdf."""
        from DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug import main

        main(["--workers", "2"])

        assert stub_cli_module.main.called
        call_args = stub_cli_module.main.call_args.args[0]
        assert "--mode" in call_args
        assert "pdf" in call_args
        assert "--workers" in call_args


class TestLegacyModuleAccess:
    """Test that legacy modules are still accessible from legacy package."""

    def test_legacy_html_importable(self) -> None:
        """Legacy HTML module should be importable from legacy package."""
        with stub_legacy_dependencies():
            from DocsToKG.DocParsing.legacy import run_docling_html_to_doctags_parallel

        assert hasattr(run_docling_html_to_doctags_parallel, "main")

    def test_legacy_pdf_importable(self) -> None:
        """Legacy PDF module should be importable from legacy package."""
        with stub_legacy_dependencies():
            from DocsToKG.DocParsing.legacy import run_docling_parallel_with_vllm_debug

        assert hasattr(run_docling_parallel_with_vllm_debug, "main")

    def test_legacy_package_has_all(self) -> None:
        """Legacy package should export expected modules in __all__."""
        from DocsToKG.DocParsing import legacy

        assert hasattr(legacy, "__all__")
        assert "run_docling_html_to_doctags_parallel" in legacy.__all__
        assert "run_docling_parallel_with_vllm_debug" in legacy.__all__
