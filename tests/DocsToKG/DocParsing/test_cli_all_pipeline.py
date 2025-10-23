import sys
import types
from pathlib import Path

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

if "DocsToKG.DocParsing.chunking.runtime" not in sys.modules:
    chunking_runtime_stub = types.ModuleType("DocsToKG.DocParsing.chunking.runtime")
    chunking_runtime_stub._main_inner = lambda *args, **kwargs: 0
    sys.modules["DocsToKG.DocParsing.chunking.runtime"] = chunking_runtime_stub

if "DocsToKG.DocParsing.chunking.config" not in sys.modules:
    chunking_config_stub = types.ModuleType("DocsToKG.DocParsing.chunking.config")

    class ChunkerCfg:
        def __init__(self) -> None:
            self.log_level = "INFO"
            self.data_root = None
            self.in_dir = None
            self.out_dir = None
            self.min_tokens = 256
            self.max_tokens = 512
            self.shard_count = 1
            self.shard_index = 0
            self.tokenizer_model = "cl100k_base"
            self.format = "parquet"
            self.resume = False
            self.force = False
            self.workers = 1

        def finalize(self) -> None:
            if isinstance(self.data_root, str):
                self.data_root = Path(self.data_root)
            if isinstance(self.in_dir, str):
                self.in_dir = Path(self.in_dir)
            if isinstance(self.out_dir, str):
                self.out_dir = Path(self.out_dir)
            self.resume = bool(self.resume)
            self.force = bool(self.force)

    chunking_config_stub.ChunkerCfg = ChunkerCfg
    sys.modules["DocsToKG.DocParsing.chunking.config"] = chunking_config_stub

if "DocsToKG.DocParsing.chunking" not in sys.modules:
    chunking_module_stub = types.ModuleType("DocsToKG.DocParsing.chunking")
    chunking_module_stub.__path__ = []
    chunking_module_stub.runtime = sys.modules["DocsToKG.DocParsing.chunking.runtime"]
    chunking_module_stub.config = sys.modules["DocsToKG.DocParsing.chunking.config"]
    sys.modules["DocsToKG.DocParsing.chunking"] = chunking_module_stub

if "DocsToKG.DocParsing.embedding.runtime" not in sys.modules:
    embedding_runtime_stub = types.ModuleType("DocsToKG.DocParsing.embedding.runtime")
    embedding_runtime_stub._main_inner = lambda *args, **kwargs: 0
    sys.modules["DocsToKG.DocParsing.embedding.runtime"] = embedding_runtime_stub

if "DocsToKG.DocParsing.embedding.config" not in sys.modules:
    embedding_config_stub = types.ModuleType("DocsToKG.DocParsing.embedding.config")

    class EmbedCfg:
        def __init__(self) -> None:
            self.log_level = "INFO"
            self.data_root = None
            self.chunks_dir = None
            self.out_dir = None
            self.vector_format = "parquet"
            self.resume = False
            self.force = False
            self.files_parallel = 1

        def finalize(self) -> None:
            if isinstance(self.data_root, str):
                self.data_root = Path(self.data_root)
            if isinstance(self.chunks_dir, str):
                self.chunks_dir = Path(self.chunks_dir)
            if isinstance(self.out_dir, str):
                self.out_dir = Path(self.out_dir)
            self.resume = bool(self.resume)
            self.force = bool(self.force)

    embedding_config_stub.EmbedCfg = EmbedCfg
    sys.modules["DocsToKG.DocParsing.embedding.config"] = embedding_config_stub

if "DocsToKG.DocParsing.embedding" not in sys.modules:
    embedding_module_stub = types.ModuleType("DocsToKG.DocParsing.embedding")
    embedding_module_stub.__path__ = []
    embedding_module_stub.runtime = sys.modules["DocsToKG.DocParsing.embedding.runtime"]
    embedding_module_stub.config = sys.modules["DocsToKG.DocParsing.embedding.config"]
    sys.modules["DocsToKG.DocParsing.embedding"] = embedding_module_stub

from DocsToKG.DocParsing import cli_unified
from DocsToKG.DocParsing.app_context import build_app_context


def test_docparse_all_no_resume_routes_html(monkeypatch, tmp_path):
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / "example.html").write_text("<html></html>")

    doctags_out = tmp_path / "DocTags"
    doctags_out.mkdir()
    chunks_out = tmp_path / "Chunks"
    chunks_out.mkdir()
    vectors_out = tmp_path / "Vectors"
    vectors_out.mkdir()

    app_ctx = build_app_context()
    app_ctx.settings.app.data_root = tmp_path
    app_ctx.settings.doctags.input_dir = html_dir
    app_ctx.settings.doctags.output_dir = doctags_out
    app_ctx.settings.doctags.mode = "auto"
    app_ctx.settings.chunk.input_dir = doctags_out
    app_ctx.settings.chunk.output_dir = chunks_out
    app_ctx.settings.embed.input_chunks_dir = chunks_out
    app_ctx.settings.embed.output_vectors_dir = vectors_out

    ctx = types.SimpleNamespace(obj=app_ctx)

    captured: dict[str, object] = {}
    pdf_called = False

    def fake_pdf_main(*, config_adapter):
        nonlocal pdf_called
        pdf_called = True
        return 0

    def fake_html_main(*, config_adapter):
        captured["doctags"] = config_adapter
        return 0

    def fake_chunk_main(*, config_adapter):
        captured["chunk"] = config_adapter
        return 0

    def fake_embed_main(*, config_adapter):
        captured["embed"] = config_adapter
        return 0

    monkeypatch.setattr(cli_unified.doctags_module, "pdf_main", fake_pdf_main)
    monkeypatch.setattr(cli_unified.doctags_module, "html_main", fake_html_main)
    monkeypatch.setattr(cli_unified.chunking_runtime, "_main_inner", fake_chunk_main)
    monkeypatch.setattr(cli_unified.embedding_runtime, "_main_inner", fake_embed_main)

    cli_unified.all(ctx, resume=False, force=False, stop_on_fail=True)

    assert pdf_called is False
    assert "doctags" in captured
    assert captured["doctags"].mode == "html"
    assert captured["doctags"].resume is False
    assert captured["chunk"].resume is False
    assert captured["embed"].resume is False
