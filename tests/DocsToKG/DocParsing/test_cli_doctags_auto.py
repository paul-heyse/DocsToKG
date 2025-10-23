import sys
import types

import typer
import pytest

if "docling_core.transforms.chunker.hybrid_chunker" not in sys.modules:
    stub_module = types.ModuleType("docling_core.transforms.chunker.hybrid_chunker")
    stub_module.HybridChunker = object
    sys.modules.setdefault("docling_core", types.ModuleType("docling_core"))
    sys.modules.setdefault("docling_core.transforms", types.ModuleType("docling_core.transforms"))
    sys.modules.setdefault(
        "docling_core.transforms.chunker", types.ModuleType("docling_core.transforms.chunker")
    )
    sys.modules["docling_core"].__path__ = []  # mark as package for nested imports
    sys.modules["docling_core.transforms"].__path__ = []
    sys.modules["docling_core.transforms.chunker"].__path__ = []
    sys.modules["docling_core.transforms.chunker.hybrid_chunker"] = stub_module
    sys.modules["docling_core"].transforms = sys.modules["docling_core.transforms"]
    sys.modules["docling_core.transforms"].chunker = sys.modules[
        "docling_core.transforms.chunker"
    ]

if "DocsToKG.DocParsing.chunking" not in sys.modules:
    chunking_runtime_stub = types.ModuleType("DocsToKG.DocParsing.chunking.runtime")
    chunking_runtime_stub._main_inner = lambda *args, **kwargs: 0
    chunking_module_stub = types.ModuleType("DocsToKG.DocParsing.chunking")
    chunking_module_stub.__path__ = []
    chunking_module_stub.runtime = chunking_runtime_stub
    sys.modules["DocsToKG.DocParsing.chunking"] = chunking_module_stub
    sys.modules["DocsToKG.DocParsing.chunking.runtime"] = chunking_runtime_stub

if "DocsToKG.DocParsing.embedding" not in sys.modules:
    embedding_runtime_stub = types.ModuleType("DocsToKG.DocParsing.embedding.runtime")
    embedding_runtime_stub._main_inner = lambda *args, **kwargs: 0
    embedding_module_stub = types.ModuleType("DocsToKG.DocParsing.embedding")
    embedding_module_stub.__path__ = []
    embedding_module_stub.runtime = embedding_runtime_stub
    sys.modules["DocsToKG.DocParsing.embedding"] = embedding_module_stub
    sys.modules["DocsToKG.DocParsing.embedding.runtime"] = embedding_runtime_stub

from DocsToKG.DocParsing import cli_unified
from DocsToKG.DocParsing.app_context import build_app_context
from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter


def test_doctags_auto_prefers_html(monkeypatch, tmp_path):
    input_dir = tmp_path / "html"
    input_dir.mkdir()
    (input_dir / "example.html").write_text("<html></html>")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    app_ctx = build_app_context()
    app_ctx.settings.app.data_root = tmp_path
    app_ctx.settings.doctags.input_dir = input_dir
    app_ctx.settings.doctags.output_dir = output_dir
    app_ctx.settings.doctags.mode = "auto"

    ctx = types.SimpleNamespace(obj=app_ctx)

    invoked_modes: list[str | None] = []
    real_to_doctags = ConfigurationAdapter.to_doctags

    def spy_to_doctags(app_ctx_arg, mode=None):
        invoked_modes.append(mode)
        return real_to_doctags(app_ctx_arg, mode=mode)

    monkeypatch.setattr(
        ConfigurationAdapter,
        "to_doctags",
        staticmethod(spy_to_doctags),
    )

    pdf_called = False

    def fake_pdf_main(*, config_adapter):
        nonlocal pdf_called
        pdf_called = True
        return 0

    captured_config = {}

    def fake_html_main(*, config_adapter):
        captured_config["mode"] = config_adapter.mode
        captured_config["input"] = config_adapter.input
        return 0

    monkeypatch.setattr(cli_unified.doctags_module, "pdf_main", fake_pdf_main)
    monkeypatch.setattr(cli_unified.doctags_module, "html_main", fake_html_main)

    class ExitCapture(BaseException):
        def __init__(self, code: int = 0):
            self.exit_code = code

    monkeypatch.setattr(cli_unified.typer, "Exit", ExitCapture)

    with pytest.raises(ExitCapture) as excinfo:
        cli_unified.doctags(
            ctx,
            input_dir=input_dir,
            output_dir=output_dir,
            mode="auto",
        )

    assert excinfo.value.exit_code == 0
    assert invoked_modes[0] == "auto"
    assert pdf_called is False
    assert captured_config["mode"] == "html"
    assert captured_config["input"] == input_dir


def test_configuration_adapter_preserves_auto_mode(tmp_path):
    app_ctx = build_app_context()
    app_ctx.settings.app.data_root = tmp_path
    app_ctx.settings.doctags.mode = "auto"

    cfg = ConfigurationAdapter.to_doctags(app_ctx)

    assert cfg.mode == "auto"
