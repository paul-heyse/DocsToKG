"""Ensure validation helpers degrade gracefully when ``psutil`` is unavailable."""

from __future__ import annotations

import builtins
import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import DocsToKG.OntologyDownload.plugins as plugins_mod


def test_validation_import_and_run_without_psutil(tmp_path: Path) -> None:
    """Validation module should import and execute validators without ``psutil``."""

    original_module = sys.modules.get("DocsToKG.OntologyDownload.validation")

    real_import = builtins.__import__
    original_register = plugins_mod.register_plugin_registry
    original_psutil = sys.modules.pop("psutil", None)
    sys.modules.pop("DocsToKG.OntologyDownload.validation", None)

    def fake_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "psutil":
            raise ImportError("psutil missing")
        return real_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = fake_import  # type: ignore[assignment]
        plugins_mod.register_plugin_registry = lambda *args, **kwargs: None  # type: ignore[assignment]

        validation = importlib.import_module("DocsToKG.OntologyDownload.validation")

        normalized_dir = tmp_path / "normalized"
        validation_dir = tmp_path / "validation"
        normalized_dir.mkdir()
        validation_dir.mkdir()
        file_path = tmp_path / "example.ttl"
        file_path.write_text("@prefix ex: <http://example.org/> . ex:a ex:b ex:c .\n")

        validation_defaults = SimpleNamespace(
            max_concurrent_validators=1,
            parser_timeout_sec=5,
            use_process_pool=False,
            process_pool_validators=[],
        )
        config = SimpleNamespace(defaults=SimpleNamespace(validation=validation_defaults))

        request = validation.ValidationRequest(
            name="stub",
            file_path=file_path,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=config,  # type: ignore[arg-type]
        )

        def stub_validator(req: validation.ValidationRequest, logger: logging.Logger) -> validation.ValidationResult:  # pragma: no cover - trivial shim
            return validation.ValidationResult(ok=True, details={}, output_files=[])

        validation.VALIDATORS.clear()
        validation.VALIDATORS["stub"] = stub_validator

        logger = logging.getLogger("DocsToKG.tests.validation.psutil")
        logger.setLevel(logging.DEBUG)

        results = validation.run_validators([request], logger)

        assert "stub" in results
        stub_result = results["stub"]
        assert stub_result.ok
        metrics = stub_result.details["metrics"]
        assert metrics["rss_mb_before"] == pytest.approx(0.0)
        assert metrics["rss_mb_after"] == pytest.approx(0.0)
        assert validation._current_memory_mb() == pytest.approx(0.0)
    finally:
        builtins.__import__ = real_import  # type: ignore[assignment]
        plugins_mod.register_plugin_registry = original_register  # type: ignore[assignment]
        sys.modules.pop("DocsToKG.OntologyDownload.validation", None)
        if original_module is not None:
            sys.modules["DocsToKG.OntologyDownload.validation"] = original_module
        else:
            importlib.import_module("DocsToKG.OntologyDownload.validation")
        if original_psutil is not None:
            sys.modules["psutil"] = original_psutil
        else:
            sys.modules.pop("psutil", None)
