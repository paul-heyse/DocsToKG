"""Smoke tests for ConfigurationAdapter pattern.

Tests verify that ConfigurationAdapter successfully converts Pydantic settings
to stage config classes, enabling direct injection into stage runtimes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
from DocsToKG.DocParsing.chunking.config import ChunkerCfg
from DocsToKG.DocParsing.doctags import DoctagsCfg
from DocsToKG.DocParsing.embedding.config import EmbedCfg


class TestConfigurationAdapterSmoke:
    """Smoke tests for ConfigurationAdapter functionality."""

    def test_adapter_to_doctags_creates_instance(self) -> None:
        """Verify that to_doctags creates a DoctagsCfg instance."""
        ctx = MagicMock()
        ctx.settings.app.data_root = Path("Data")
        ctx.settings.app.log_level = "INFO"
        ctx.settings.doctags.input_dir = Path("Data/PDFs")
        ctx.settings.doctags.output_dir = Path("Data/DocTags")
        ctx.settings.doctags.model_id = "granite-docling"
        ctx.settings.runner.workers = 4

        cfg = ConfigurationAdapter.to_doctags(ctx, mode="pdf")

        # Verify it's the correct type
        assert isinstance(cfg, DoctagsCfg)
        # Verify mode was set
        assert cfg.mode == "pdf"
        # Verify workers applied
        assert cfg.workers == 4

    def test_adapter_to_chunk_creates_instance(self) -> None:
        """Verify that to_chunk creates a ChunkerCfg instance."""
        ctx = MagicMock()
        ctx.settings.app.data_root = Path("Data")
        ctx.settings.app.log_level = "INFO"
        ctx.settings.chunk.input_dir = Path("Data/DocTags")
        ctx.settings.chunk.output_dir = Path("Data/Chunks")
        ctx.settings.chunk.min_tokens = 120
        ctx.settings.chunk.max_tokens = 800
        ctx.settings.chunk.tokenizer_model = "cl100k_base"
        ctx.settings.runner.workers = 4

        cfg = ConfigurationAdapter.to_chunk(ctx)

        # Verify it's the correct type
        assert isinstance(cfg, ChunkerCfg)
        # Verify token limits applied
        assert cfg.min_tokens == 120
        assert cfg.max_tokens == 800
        # Verify workers applied
        assert cfg.workers == 4

    def test_adapter_to_embed_creates_instance(self) -> None:
        """Verify that to_embed creates an EmbedCfg instance."""
        ctx = MagicMock()
        ctx.settings.app.data_root = Path("Data")
        ctx.settings.app.log_level = "INFO"
        ctx.settings.embed.input_chunks_dir = Path("Data/Chunks")
        ctx.settings.embed.output_vectors_dir = Path("Data/Vectors")
        ctx.settings.runner.workers = 4

        cfg = ConfigurationAdapter.to_embed(ctx)

        # Verify it's the correct type
        assert isinstance(cfg, EmbedCfg)
        # Verify workers applied (via files_parallel)
        assert cfg.files_parallel == 4

    def test_adapter_mode_override(self) -> None:
        """Verify that mode parameter overrides AppContext mode."""
        ctx = MagicMock()
        ctx.settings.app.data_root = Path("Data")
        ctx.settings.app.log_level = "INFO"
        ctx.settings.doctags.input_dir = Path("Data/PDFs")
        ctx.settings.doctags.output_dir = Path("Data/DocTags")
        ctx.settings.doctags.model_id = "granite-docling"
        ctx.settings.doctags.mode = "auto"
        ctx.settings.runner.workers = 1

        # Override with pdf
        cfg_pdf = ConfigurationAdapter.to_doctags(ctx, mode="pdf")
        assert cfg_pdf.mode == "pdf"

        # Override with html
        cfg_html = ConfigurationAdapter.to_doctags(ctx, mode="html")
        assert cfg_html.mode == "html"

    def test_stage_entry_points_accept_adapter_parameter(self) -> None:
        """Verify stage entry points have config_adapter parameter."""
        from DocsToKG.DocParsing import doctags as doctags_module
        from DocsToKG.DocParsing.chunking import runtime as chunking_runtime
        from DocsToKG.DocParsing.embedding import runtime as embedding_runtime

        import inspect

        # Check doctags
        pdf_sig = inspect.signature(doctags_module.pdf_main)
        assert "config_adapter" in pdf_sig.parameters

        html_sig = inspect.signature(doctags_module.html_main)
        assert "config_adapter" in html_sig.parameters

        # Check chunking
        chunk_sig = inspect.signature(chunking_runtime._main_inner)
        assert "config_adapter" in chunk_sig.parameters

        # Check embedding
        embed_sig = inspect.signature(embedding_runtime._main_inner)
        assert "config_adapter" in embed_sig.parameters

    def test_stage_entry_points_maintain_backward_compat(self) -> None:
        """Verify stage entry points still accept args=None for backward compat."""
        from DocsToKG.DocParsing import doctags as doctags_module
        from DocsToKG.DocParsing.chunking import runtime as chunking_runtime
        from DocsToKG.DocParsing.embedding import runtime as embedding_runtime

        import inspect

        # Check doctags
        pdf_sig = inspect.signature(doctags_module.pdf_main)
        assert "args" in pdf_sig.parameters
        assert pdf_sig.parameters["args"].default is None

        # Check chunking
        chunk_sig = inspect.signature(chunking_runtime._main_inner)
        assert "args" in chunk_sig.parameters
        assert chunk_sig.parameters["args"].default is None

        # Check embedding
        embed_sig = inspect.signature(embedding_runtime._main_inner)
        assert "args" in embed_sig.parameters
        assert embed_sig.parameters["args"].default is None
