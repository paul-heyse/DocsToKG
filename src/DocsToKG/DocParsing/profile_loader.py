# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.profile_loader",
#   "purpose": "Profile loader and configuration builder for DocParsing settings.",
#   "sections": [
#     {
#       "id": "load-profile-file",
#       "name": "load_profile_file",
#       "anchor": "function-load-profile-file",
#       "kind": "function"
#     },
#     {
#       "id": "merge-dicts",
#       "name": "merge_dicts",
#       "anchor": "function-merge-dicts",
#       "kind": "function"
#     },
#     {
#       "id": "apply-dot-path-override",
#       "name": "apply_dot_path_override",
#       "anchor": "function-apply-dot-path-override",
#       "kind": "function"
#     },
#     {
#       "id": "env-var-to-dot-path",
#       "name": "_env_var_to_dot_path",
#       "anchor": "function-env-var-to-dot-path",
#       "kind": "function"
#     },
#     {
#       "id": "settingsbuilder",
#       "name": "SettingsBuilder",
#       "anchor": "class-settingsbuilder",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Profile loader and configuration builder for DocParsing settings.

Handles loading of profile files (docstokg.toml/.yaml), merging profile
configurations onto defaults with proper precedence (CLI > ENV > profile > defaults),
and computing stable hashes for config change detection.

NAVMAP:
- load_profile_file: Load TOML/YAML profile into dict
- merge_dicts: Deep merge profile onto defaults
- apply_precedence: Apply layering with source tracking
- SettingsBuilder: Main builder class with layering algorithm
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        tomllib = None

try:
    import yaml
except ImportError:
    yaml = None

from .config_loaders import ConfigLoadError, load_toml_markers, load_yaml_markers


def load_profile_file(path: Path) -> Dict[str, Any]:
    """
    Load a profile file (TOML or YAML) into a dict.

    Args:
        path: Path to profile file (.toml or .yaml/.yml)

    Returns:
        Nested dict with profile configuration

    Raises:
        ConfigLoadError: If file is not found or malformed
    """
    if not path.exists():
        raise ConfigLoadError(f"Profile file not found: {path}")

    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    try:
        if suffix == ".toml":
            return load_toml_markers(raw)
        elif suffix in {".yaml", ".yml"}:
            return load_yaml_markers(raw)
        else:
            raise ConfigLoadError(f"Unsupported profile file format: {suffix}")
    except Exception as e:
        raise ConfigLoadError(f"Failed to parse profile file {path}: {e}") from e


def merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deep merge override dict into base dict (override wins on conflicts).

    Args:
        base: Base configuration dict
        override: Override dict (takes precedence)

    Returns:
        Merged dict with override values winning
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def apply_dot_path_override(
    config: Dict[str, Any],
    dot_path: str,
    value: Any,
) -> Dict[str, Any]:
    """
    Apply a dot-path override (e.g., 'embed.dense.backend=tei') to config.

    Args:
        config: Base config dict
        dot_path: Dot-separated path with optional =value
        value: Value to set (overrides parsing from dot_path)

    Returns:
        Modified config dict

    Raises:
        ValueError: If dot_path is malformed
    """
    if "=" in dot_path:
        path_str, val_str = dot_path.split("=", 1)
        # Try to coerce the value to appropriate type
        try:
            if val_str.lower() in {"true", "false"}:
                value = val_str.lower() == "true"
            elif val_str.isdigit():
                value = int(val_str)
            else:
                try:
                    value = float(val_str)
                except ValueError:
                    value = val_str
        except Exception:
            value = val_str
    else:
        path_str = dot_path

    keys = path_str.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ValueError(f"Cannot apply override to non-dict at '{key}'")
        current = current[key]

    current[keys[-1]] = value
    return config


def _env_var_to_dot_path(
    env_name: str,
    *,
    prefix: str,
    root_keys: Set[str],
) -> Optional[str]:
    """
    Convert an environment variable name into a dot-path understood by SettingsBuilder.

    Supports both legacy single-underscore stage prefixes (e.g. ``DOCSTOKG_CHUNK_MIN_TOKENS``)
    and nested ``__`` segments used by Pydantic BaseSettings (e.g.
    ``DOCSTOKG_EMBED__DENSE__BACKEND``). Field names retain their underscores so multi-word
    attributes map correctly.
    """

    if not env_name.startswith(prefix):
        return None

    suffix = env_name[len(prefix) :].strip("_")
    if not suffix:
        return None

    raw_segments = [segment for segment in suffix.split("__") if segment]
    if not raw_segments:
        return None

    # Normalise to lowercase while preserving embedded underscores for field names.
    segments: List[str] = [segment.lower() for segment in raw_segments]
    path_parts: List[str] = []

    # Split legacy single-underscore prefixes like "chunk_min_tokens".
    first = segments[0]
    remainder: List[str] = segments[1:]
    if "_" in first:
        candidate_root, candidate_tail = first.split("_", 1)
        candidate_root = candidate_root.strip("_")
        if candidate_root in root_keys:
            path_parts.append(candidate_root)
            candidate_tail = candidate_tail.strip("_")
            if candidate_tail:
                remainder.insert(0, candidate_tail)
    if not path_parts:
        if first in root_keys:
            path_parts.append(first)
        else:
            fallback_root = "app" if "app" in root_keys else None
            if fallback_root is not None:
                path_parts.append(fallback_root)
                remainder.insert(0, first)
            else:
                path_parts.append(first)

    path_parts.extend(remainder)
    return ".".join(part for part in path_parts if part)


class SettingsBuilder:
    """
    Builder for layering configuration from multiple sources.

    Implements the canonical precedence: CLI > ENV > profile > defaults
    with optional source tracking for debugging.
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self.defaults: Dict[str, Any] = {}
        self.profile_data: Dict[str, Any] = {}
        self.env_overrides: Dict[str, Any] = {}
        self.cli_overrides: Dict[str, Any] = {}
        self.sources: Dict[str, str] = {}  # Track source of each key

    def add_defaults(self, defaults: Dict[str, Any]) -> SettingsBuilder:
        """Add default configuration."""
        self.defaults = defaults
        return self

    def add_profile(
        self,
        profile_name: Optional[str] = None,
        profile_file: Optional[Path] = None,
    ) -> SettingsBuilder:
        """
        Load and add profile configuration.

        Args:
            profile_name: Name of the profile (e.g., 'gpu', 'local')
            profile_file: Path to profile file (uses default if None)

        Returns:
            Self for chaining
        """
        if profile_name is None:
            return self

        if profile_file is None:
            # Try default locations
            for candidate in [Path("docstokg.toml"), Path("config/docstokg.toml")]:
                if candidate.exists():
                    profile_file = candidate
                    break

        if profile_file is None or not profile_file.exists():
            raise ConfigLoadError(
                f"Profile file not found for profile '{profile_name}'. "
                "Searched: docstokg.toml, config/docstokg.toml"
            )

        all_profiles = load_profile_file(profile_file)
        if "profile" not in all_profiles or profile_name not in all_profiles["profile"]:
            raise ConfigLoadError(
                f"Profile '{profile_name}' not found in {profile_file}. "
                f"Available: {list(all_profiles.get('profile', {}).keys())}"
            )

        self.profile_data = all_profiles["profile"][profile_name]
        return self

    def add_env_overrides(self, env_prefix: str = "DOCSTOKG_") -> SettingsBuilder:
        """
        Extract environment variable overrides.

        Args:
            env_prefix: Prefix for env vars to consider (e.g., 'DOCSTOKG_')

        Returns:
            Self for chaining
        """
        self.env_overrides = {}
        root_keys: Set[str] = {key.lower() for key in self.defaults.keys()}
        if not root_keys:
            root_keys = {"app", "runner", "doctags", "chunk", "embed"}
        for key, value in os.environ.items():
            dot_path = _env_var_to_dot_path(key, prefix=env_prefix, root_keys=root_keys)
            if dot_path is None:
                continue
            # Try to coerce value to appropriate type
            try:
                if value.lower() in {"true", "false"}:
                    val = value.lower() == "true"
                elif value.isdigit():
                    val = int(value)
                else:
                    try:
                        val = float(value)
                    except ValueError:
                        val = value
            except Exception:
                val = value
            apply_dot_path_override(self.env_overrides, dot_path, val)
        return self

    def add_cli_overrides(self, overrides: Dict[str, Any]) -> SettingsBuilder:
        """
        Add CLI argument overrides.

        Args:
            overrides: Dict of CLI overrides

        Returns:
            Self for chaining
        """
        self.cli_overrides = overrides
        return self

    def build(self, track_sources: bool = False) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Build the effective configuration with proper precedence.

        Precedence: CLI > ENV > profile > defaults

        Args:
            track_sources: If True, return source tracking dict

        Returns:
            Tuple of (effective config, source tracking dict if requested)
        """
        # Start with defaults
        result = dict(self.defaults)

        # Layer profile (if any)
        if self.profile_data:
            result = merge_dicts(result, self.profile_data)

        # Layer ENV
        if self.env_overrides:
            result = merge_dicts(result, self.env_overrides)

        # Layer CLI (highest precedence)
        if self.cli_overrides:
            result = merge_dicts(result, self.cli_overrides)

        if track_sources:
            sources = self._compute_sources(result)
            return result, sources
        else:
            return result, {}

    def _compute_sources(self, final_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Compute source tracking (which layer each key came from).

        Args:
            final_config: Final merged config

        Returns:
            Dict mapping dot-paths to source layer
        """
        sources: Dict[str, str] = {}

        def track_layer(path: str, layer: Dict[str, Any], layer_name: str) -> None:
            for key, value in layer.items():
                dot_key = f"{path}.{key}" if path else key
                if isinstance(value, dict):
                    track_layer(dot_key, value, layer_name)
                else:
                    sources[dot_key] = layer_name

        # Track each layer
        track_layer("", self.defaults, "default")
        track_layer("", self.profile_data, "profile")
        track_layer("", self.env_overrides, "env")
        track_layer("", self.cli_overrides, "cli")

        return sources


__all__ = [
    "load_profile_file",
    "merge_dicts",
    "apply_dot_path_override",
    "SettingsBuilder",
]
