"""Configuration adapter for converting Pydantic settings to stage config dataclasses.

This module provides adapters that bridge the unified Typer CLI (which uses Pydantic
settings) with stage runtimes (which expect legacy argparse-based config dataclasses).

The adapter pattern eliminates the need for removed from_args() classmethods by
directly building stage config objects from Pydantic settings, enabling:
  - Direct configuration injection (testable, no sys.argv re-parsing)
  - Single source of truth (unified settings)
  - Backward compatibility (stages still work with old paths)
  - Clean architecture (clear separation of concerns)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app_context import AppContext
    from .chunking.config import ChunkerCfg
    from .doctags import DoctagsCfg
    from .embedding.config import EmbedCfg


class ConfigurationAdapter:
    """Convert AppContext Pydantic settings to stage config dataclasses.

    This adapter is the bridge between the new unified CLI (Pydantic-based)
    and existing stage runtimes (argparse-based). It normalizes configuration
    by directly building stage config objects from the merged, validated settings.

    Example:
        >>> from DocsToKG.DocParsing.app_context import AppContext
        >>> from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
        >>> app_ctx = AppContext(...)  # Built by CLI
        >>> chunk_cfg = ConfigurationAdapter.to_chunk(app_ctx)
        >>> exit_code = chunking_runtime._main_inner(args=None, config_adapter=chunk_cfg)
    """

    @staticmethod
    def to_doctags(app_ctx: AppContext, mode: str = "pdf") -> DoctagsCfg:
        """Build DoctagsCfg from AppContext settings.

        Args:
            app_ctx: The application context containing merged settings
            mode: Override mode (pdf or html). If set, overrides app_ctx.settings.doctags.mode

        Returns:
            Configured DoctagsCfg instance ready for stage runtime

        Raises:
            ValueError: If required settings are missing or invalid
        """
        from .doctags import DoctagsCfg

        cfg = DoctagsCfg()

        # Apply app-level settings
        if app_ctx.settings.app.data_root:
            cfg.data_root = app_ctx.settings.app.data_root
        if app_ctx.settings.app.log_level:
            cfg.log_level = app_ctx.settings.app.log_level

        # Apply doctags-specific settings
        if app_ctx.settings.doctags.input_dir:
            cfg.input = Path(app_ctx.settings.doctags.input_dir)
        if app_ctx.settings.doctags.output_dir:
            cfg.output = Path(app_ctx.settings.doctags.output_dir)
        if app_ctx.settings.doctags.model_id:
            cfg.model = app_ctx.settings.doctags.model_id

        # Apply runner settings (workers override)
        if app_ctx.settings.runner.workers:
            cfg.workers = app_ctx.settings.runner.workers

        # Set mode explicitly (overrides any existing mode)
        cfg.mode = mode

        # Finalize normalizes paths and validates
        cfg.finalize()
        return cfg

    @staticmethod
    def to_chunk(app_ctx: AppContext) -> ChunkerCfg:
        """Build ChunkerCfg from AppContext settings.

        Args:
            app_ctx: The application context containing merged settings

        Returns:
            Configured ChunkerCfg instance ready for stage runtime

        Raises:
            ValueError: If required settings are missing or invalid
        """
        from .chunking.config import ChunkerCfg

        cfg = ChunkerCfg()

        # Apply app-level settings
        if app_ctx.settings.app.data_root:
            cfg.data_root = app_ctx.settings.app.data_root
        if app_ctx.settings.app.log_level:
            cfg.log_level = app_ctx.settings.app.log_level

        # Apply chunk-specific settings
        if app_ctx.settings.chunk.input_dir:
            cfg.in_dir = Path(app_ctx.settings.chunk.input_dir)
        if app_ctx.settings.chunk.output_dir:
            cfg.out_dir = Path(app_ctx.settings.chunk.output_dir)

        cfg.min_tokens = app_ctx.settings.chunk.min_tokens or cfg.min_tokens
        cfg.max_tokens = app_ctx.settings.chunk.max_tokens or cfg.max_tokens

        # Apply runner settings (workers override)
        if app_ctx.settings.runner.workers:
            cfg.workers = app_ctx.settings.runner.workers

        # Finalize normalizes paths and validates
        cfg.finalize()
        return cfg

    @staticmethod
    def to_embed(app_ctx: AppContext) -> EmbedCfg:
        """Build EmbedCfg from AppContext settings.

        Args:
            app_ctx: The application context containing merged settings

        Returns:
            Configured EmbedCfg instance ready for stage runtime

        Raises:
            ValueError: If required settings are missing or invalid
        """
        from .embedding.config import EmbedCfg

        cfg = EmbedCfg()

        # Apply app-level settings
        if app_ctx.settings.app.data_root:
            cfg.data_root = app_ctx.settings.app.data_root
        if app_ctx.settings.app.log_level:
            cfg.log_level = app_ctx.settings.app.log_level

        # Apply embed-specific settings
        if app_ctx.settings.embed.input_chunks_dir:
            cfg.chunks_dir = Path(app_ctx.settings.embed.input_chunks_dir)
        if app_ctx.settings.embed.output_vectors_dir:
            cfg.out_dir = Path(app_ctx.settings.embed.output_vectors_dir)

        # Apply runner settings (workers override)
        if app_ctx.settings.runner.workers:
            cfg.files_parallel = app_ctx.settings.runner.workers

        # Finalize normalizes paths and validates
        cfg.finalize()
        return cfg
