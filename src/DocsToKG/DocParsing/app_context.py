# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.app_context",
#   "purpose": "Application context builder for DocParsing.",
#   "sections": [
#     {
#       "id": "appcontext",
#       "name": "AppContext",
#       "anchor": "class-appcontext",
#       "kind": "class"
#     },
#     {
#       "id": "build-app-context",
#       "name": "build_app_context",
#       "anchor": "function-build-app-context",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Application context builder for DocParsing.

Orchestrates the complete settings layering (defaults → profile → ENV → CLI)
with proper precedence and provides a unified context object for stages.

NAVMAP:
- AppContext: Main context dataclass holding all configs
- build_app_context: Builder function with full layering logic
- apply_cli_overrides_to_settings: Convert CLI args to Settings updates
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .profile_loader import ConfigLoadError, SettingsBuilder
from .settings import (
    AppCfg,
    ChunkCfg,
    DocTagsCfg,
    EmbedCfg,
    RunnerCfg,
    Settings,
)


@dataclass
class AppContext:
    """
    Application context holding all effective configuration.

    Contains merged and validated settings for all stages, plus
    metadata about how the configuration was loaded.
    """

    settings: Settings
    cfg_hashes: dict[str, str]  # Per-stage content hashes
    profile: str | None = None
    strict_config: bool = True
    source_tracking: dict[str, str] = None  # Optional: key → source layer

    def __post_init__(self) -> None:
        """Initialize source tracking if not provided."""
        if self.source_tracking is None:
            self.source_tracking = {}


def build_app_context(
    profile: str | None = None,
    profile_file: Path | None = None,
    strict_config: bool = True,
    track_sources: bool = False,
    # Global overrides
    data_root: Path | None = None,
    log_level: str | None = None,
    log_format: str | None = None,
    # Runner overrides
    workers: int | None = None,
    policy: str | None = None,
    # Doctags overrides
    doctags_input_dir: Path | None = None,
    doctags_output_dir: Path | None = None,
    doctags_mode: str | None = None,
    # Chunk overrides
    chunk_input_dir: Path | None = None,
    chunk_output_dir: Path | None = None,
    chunk_format: str | None = None,
    chunk_min_tokens: int | None = None,
    chunk_max_tokens: int | None = None,
    # Embed overrides
    embed_input_chunks_dir: Path | None = None,
    embed_output_vectors_dir: Path | None = None,
    embed_vector_format: str | None = None,
    # Additional overrides (CLI can pass arbitrary kwargs)
    **extra_overrides: Any,
) -> AppContext:
    """
    Build an application context with full configuration layering.

    Precedence: CLI args > ENV > profile > defaults

    Args:
        profile: Profile name to load
        profile_file: Override profile file path
        strict_config: If True, raise on unknown keys; else warn
        track_sources: If True, track which layer each key came from
        data_root: Override global data root
        log_level: Override log level
        log_format: Override log format
        workers: Override runner workers
        policy: Override runner policy
        doctags_input_dir: Override doctags input
        doctags_output_dir: Override doctags output
        doctags_mode: Override doctags mode
        chunk_input_dir: Override chunk input
        chunk_output_dir: Override chunk output
        chunk_format: Override chunk format
        chunk_min_tokens: Override min tokens
        chunk_max_tokens: Override max tokens
        embed_input_chunks_dir: Override embed chunks input
        embed_output_vectors_dir: Override embed vectors output
        embed_vector_format: Override embed vector format
        **extra_overrides: Additional CLI overrides (dot-path or flat)

    Returns:
        AppContext with effective settings and metadata

    Raises:
        ConfigLoadError: If profile file not found or malformed
        ValidationError: If final settings fail Pydantic validation
    """

    # Build CLI overrides dict from all provided parameters
    cli_overrides: dict[str, Any] = {}

    if data_root is not None:
        cli_overrides.setdefault("app", {})["data_root"] = str(data_root)
    if log_level is not None:
        cli_overrides.setdefault("app", {})["log_level"] = log_level
    if log_format is not None:
        cli_overrides.setdefault("app", {})["log_format"] = log_format

    if workers is not None:
        cli_overrides.setdefault("runner", {})["workers"] = workers
    if policy is not None:
        cli_overrides.setdefault("runner", {})["policy"] = policy

    if doctags_input_dir is not None:
        cli_overrides.setdefault("doctags", {})["input_dir"] = str(doctags_input_dir)
    if doctags_output_dir is not None:
        cli_overrides.setdefault("doctags", {})["output_dir"] = str(doctags_output_dir)
    if doctags_mode is not None:
        cli_overrides.setdefault("doctags", {})["mode"] = doctags_mode

    if chunk_input_dir is not None:
        cli_overrides.setdefault("chunk", {})["input_dir"] = str(chunk_input_dir)
    if chunk_output_dir is not None:
        cli_overrides.setdefault("chunk", {})["output_dir"] = str(chunk_output_dir)
    if chunk_format is not None:
        cli_overrides.setdefault("chunk", {})["format"] = chunk_format
    if chunk_min_tokens is not None:
        cli_overrides.setdefault("chunk", {})["min_tokens"] = chunk_min_tokens
    if chunk_max_tokens is not None:
        cli_overrides.setdefault("chunk", {})["max_tokens"] = chunk_max_tokens

    if embed_input_chunks_dir is not None:
        cli_overrides.setdefault("embed", {})["input_chunks_dir"] = str(embed_input_chunks_dir)
    if embed_output_vectors_dir is not None:
        cli_overrides.setdefault("embed", {})["output_vectors_dir"] = str(embed_output_vectors_dir)
    if embed_vector_format is not None:
        cli_overrides.setdefault("embed", {}).setdefault("vectors", {})[
            "format"
        ] = embed_vector_format

    # Add any extra overrides (for extensibility)
    if extra_overrides:
        for key, value in extra_overrides.items():
            if "." in key:
                # Dot-path like "embed.dense.backend"
                parts = key.split(".")
                current = cli_overrides
                for part in parts[:-1]:
                    current.setdefault(part, {})
                    current = current[part]
                current[parts[-1]] = value
            else:
                cli_overrides[key] = value

    # Build default settings
    defaults = {
        "app": AppCfg().model_dump(exclude_none=False),
        "runner": RunnerCfg().model_dump(exclude_none=False),
        "doctags": DocTagsCfg().model_dump(exclude_none=False),
        "chunk": ChunkCfg().model_dump(exclude_none=False),
        "embed": EmbedCfg().model_dump(exclude_none=False),
    }

    # Use SettingsBuilder to layer configs
    builder = SettingsBuilder()
    builder.add_defaults(defaults)

    try:
        builder.add_profile(profile, profile_file)
    except ConfigLoadError:
        if profile is not None:
            raise  # Profile was explicitly requested
        # Silently ignore missing profile if not explicitly set

    builder.add_env_overrides(env_prefix="DOCSTOKG_")
    builder.add_cli_overrides(cli_overrides)

    merged_config, sources = builder.build(track_sources=track_sources)

    # Validate and build Settings object
    try:
        # Handle the nested conversion carefully
        settings_dict = {
            "app": merged_config.get("app", {}),
            "runner": merged_config.get("runner", {}),
            "doctags": merged_config.get("doctags", {}),
            "chunk": merged_config.get("chunk", {}),
            "embed": merged_config.get("embed", {}),
        }

        # Construct each config object
        app_cfg = AppCfg(**settings_dict.get("app", {}))
        runner_cfg = RunnerCfg(**settings_dict.get("runner", {}))
        doctags_cfg = DocTagsCfg(**settings_dict.get("doctags", {}))
        chunk_cfg = ChunkCfg(**settings_dict.get("chunk", {}))
        embed_cfg = EmbedCfg(**settings_dict.get("embed", {}))

        settings = Settings(
            app=app_cfg,
            runner=runner_cfg,
            doctags=doctags_cfg,
            chunk=chunk_cfg,
            embed=embed_cfg,
        )
    except ValidationError as e:
        # Provide helpful error messages
        raise ValueError(
            f"Configuration validation failed:\n{e}\n\n"
            "Check your profile, environment variables, and CLI arguments."
        ) from e

    # Compute content hashes
    cfg_hashes = settings.compute_stage_hashes()

    # Build context
    return AppContext(
        settings=settings,
        cfg_hashes=cfg_hashes,
        profile=profile,
        strict_config=strict_config,
        source_tracking=sources if track_sources else {},
    )


__all__ = [
    "AppContext",
    "build_app_context",
]
