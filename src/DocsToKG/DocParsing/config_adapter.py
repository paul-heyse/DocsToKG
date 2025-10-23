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

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

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
    def _normalize_mode(mode: Any) -> Optional[str]:
        """Normalize ``mode`` values from CLI overrides or settings."""

        if mode is None:
            return None
        if isinstance(mode, Enum):
            mode_value: Any = mode.value
        else:
            mode_value = mode
        if mode_value is None:
            return None
        normalized = str(mode_value).strip()
        if not normalized:
            return None
        return normalized.lower()

    @staticmethod
    def to_doctags(app_ctx: AppContext, mode: Any = None) -> DoctagsCfg:
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

        doctags_settings = app_ctx.settings.doctags

        # Apply app-level settings
        if app_ctx.settings.app.data_root:
            cfg.data_root = app_ctx.settings.app.data_root
        if app_ctx.settings.app.log_level:
            cfg.log_level = app_ctx.settings.app.log_level

        # Apply doctags-specific settings
        if doctags_settings.input_dir:
            cfg.input = Path(doctags_settings.input_dir)
        if doctags_settings.output_dir:
            cfg.output = Path(doctags_settings.output_dir)
        if doctags_settings.model_id:
            cfg.model = doctags_settings.model_id

        cfg.resume = bool(doctags_settings.resume)
        cfg.force = bool(doctags_settings.force)
        cfg.vllm_wait_timeout = int(doctags_settings.vllm_wait_timeout_s)

        # Apply runner settings (workers override)
        if app_ctx.settings.runner.workers:
            cfg.workers = app_ctx.settings.runner.workers

        override_mode = ConfigurationAdapter._normalize_mode(mode)
        configured_mode = ConfigurationAdapter._normalize_mode(doctags_settings.mode)

        if override_mode is not None:
            cfg.mode = override_mode
        elif configured_mode is not None:
            cfg.mode = configured_mode

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

        chunk_settings = app_ctx.settings.chunk

        # Apply app-level settings
        if app_ctx.settings.app.data_root:
            cfg.data_root = app_ctx.settings.app.data_root
        if app_ctx.settings.app.log_level:
            cfg.log_level = app_ctx.settings.app.log_level

        # Apply chunk-specific settings
        if chunk_settings.input_dir:
            cfg.in_dir = Path(chunk_settings.input_dir)
        if chunk_settings.output_dir:
            cfg.out_dir = Path(chunk_settings.output_dir)

        cfg.min_tokens = chunk_settings.min_tokens
        cfg.max_tokens = chunk_settings.max_tokens
        if chunk_settings.tokenizer_model:
            cfg.tokenizer_model = chunk_settings.tokenizer_model
        cfg.shard_count = chunk_settings.shard_count
        cfg.shard_index = chunk_settings.shard_index
        cfg.format = chunk_settings.format.value
        cfg.resume = bool(chunk_settings.resume)
        cfg.force = bool(chunk_settings.force)

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

        embed_settings = app_ctx.settings.embed

        # Apply app-level settings
        if app_ctx.settings.app.data_root:
            cfg.data_root = app_ctx.settings.app.data_root
        if app_ctx.settings.app.log_level:
            cfg.log_level = app_ctx.settings.app.log_level

        # Apply embed-specific settings
        if embed_settings.input_chunks_dir:
            cfg.chunks_dir = Path(embed_settings.input_chunks_dir)
        if embed_settings.output_vectors_dir:
            cfg.out_dir = Path(embed_settings.output_vectors_dir)

        cfg.vector_format = embed_settings.vectors.format.value
        cfg.dense_backend = embed_settings.dense.backend.value
        cfg.resume = bool(embed_settings.resume)
        cfg.force = bool(embed_settings.force)

        # Apply runner settings (workers override)
        if app_ctx.settings.runner.workers:
            cfg.files_parallel = app_ctx.settings.runner.workers

        # Finalize normalizes paths and validates
        cfg.finalize()
        return cfg
