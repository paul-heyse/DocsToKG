# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.loader",
#   "purpose": "Configuration Loading with File/Env/CLI Precedence.",
#   "sections": [
#     {
#       "id": "read-file",
#       "name": "_read_file",
#       "anchor": "function-read-file",
#       "kind": "function"
#     },
#     {
#       "id": "assign-nested",
#       "name": "_assign_nested",
#       "anchor": "function-assign-nested",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-env-value",
#       "name": "_coerce_env_value",
#       "anchor": "function-coerce-env-value",
#       "kind": "function"
#     },
#     {
#       "id": "merge-env-overrides",
#       "name": "_merge_env_overrides",
#       "anchor": "function-merge-env-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "merge-cli-overrides",
#       "name": "_merge_cli_overrides",
#       "anchor": "function-merge-cli-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "load-config",
#       "name": "load_config",
#       "anchor": "function-load-config",
#       "kind": "function"
#     },
#     {
#       "id": "validate-config-file",
#       "name": "validate_config_file",
#       "anchor": "function-validate-config-file",
#       "kind": "function"
#     },
#     {
#       "id": "export-config-schema",
#       "name": "export_config_schema",
#       "anchor": "function-export-config-schema",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Configuration Loading with File/Env/CLI Precedence

Implements three-level config composition:
1. **File level** (YAML/JSON) — base configuration
2. **Environment level** — DTKG_* prefixed variables override file
3. **CLI level** — programmatic overrides win

Environment variables use double-underscore notation:
  DTKG_HTTP__USER_AGENT="Custom UA"  →  http.user_agent="Custom UA"
  DTKG_RESOLVERS__ORDER='["arxiv","landing"]'  →  resolvers.order=[...]

JSON values are automatically parsed; strings are type-coerced when possible.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None

from .models import ContentDownloadConfig

_LOGGER = logging.getLogger(__name__)

# ============================================================================
# Helpers
# ============================================================================


def _read_file(path: str) -> dict[str, Any]:
    """
    Read YAML or JSON config file.

    Args:
        path: File path (suffix determines format: .yaml/.yml or .json)

    Returns:
        Parsed config dictionary

    Raises:
        ValueError: If file cannot be read or parsed
        RuntimeError: If YAML requested but PyYAML not available
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Config file not found: {path}")

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Cannot read config file {path}: {e}") from e

    suffix = p.suffix.lower()

    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError(
                "YAML config requested but PyYAML not installed. Install: pip install pyyaml"
            )
        try:
            return yaml.safe_load(text) or {}
        except Exception as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if suffix == ".json":
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

    raise ValueError(f"Unsupported file format: {suffix}. Use .yaml or .json")


def _assign_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Assign value to nested dict using dot notation.

    Example:
        _assign_nested(data, "http.user_agent", "MyUA")
        → data["http"]["user_agent"] = "MyUA"

    Args:
        data: Target dictionary (modified in place)
        dotted_key: Dot-separated path (e.g., "http.user_agent")
        value: Value to assign
    """
    keys = dotted_key.split(".")
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def _coerce_env_value(value: str) -> Any:
    """
    Attempt to coerce environment variable string to appropriate type.

    Tries JSON parsing first (handles lists, dicts, bools, numbers).
    Falls back to string if JSON fails.

    Args:
        value: Environment variable string value

    Returns:
        Parsed/coerced value
    """
    # Try JSON first (handles lists, dicts, bools, numbers, null)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try booleans
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try int
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Default: string
    return value


def _merge_env_overrides(data: dict[str, Any], env_prefix: str = "DTKG_") -> dict[str, Any]:
    """
    Overlay environment variables onto config dict.

    Looks for DTKG_* prefixed variables and maps double-underscore
    notation to nested dicts.

    Args:
        data: Base config dict (will be modified)
        env_prefix: Environment variable prefix (default: DTKG_)

    Returns:
        Modified data dict
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue

        # Remove prefix and convert to dot notation
        relative_key = env_key[len(env_prefix) :].lower()
        dotted_key = relative_key.replace("__", ".")

        # Coerce value
        coerced_value = _coerce_env_value(env_value)

        # Assign to nested structure
        _assign_nested(data, dotted_key, coerced_value)
        _LOGGER.debug(f"Environment override: {env_key} → {dotted_key} = {coerced_value!r}")

    return data


def _merge_cli_overrides(
    data: dict[str, Any], cli_overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """
    Recursively merge CLI overrides into base config dict.

    Later values win (standard dict.update() semantics).

    Args:
        data: Base config dict
        cli_overrides: CLI override dict (or None)

    Returns:
        Merged dict
    """
    if not cli_overrides:
        return data

    for key, value in cli_overrides.items():
        if isinstance(value, dict) and key in data and isinstance(data[key], dict):
            # Recursive merge for nested dicts
            data[key] = _merge_cli_overrides(data[key], value)
        else:
            data[key] = value
        _LOGGER.debug(f"CLI override: {key} = {value!r}")

    return data


# ============================================================================
# Public API
# ============================================================================


def load_config(
    path: str | None = None,
    env_prefix: str = "DTKG_",
    cli_overrides: Mapping[str, Any] | None = None,
) -> ContentDownloadConfig:
    """
    Load ContentDownloadConfig from file, environment, and CLI with proper precedence.

    **Precedence:** file < environment < CLI

    Args:
        path: Path to YAML/JSON config file (optional)
        env_prefix: Environment variable prefix (default: DTKG_)
        cli_overrides: CLI overrides dict (optional)

    Returns:
        Validated ContentDownloadConfig instance

    Raises:
        ValueError: If config is invalid or file cannot be read
        RuntimeError: If required dependencies missing
    """
    data: dict[str, Any] = {}

    # Step 1: Load file
    if path:
        try:
            data = _read_file(path)
            _LOGGER.info(f"Loaded config from {path}")
        except Exception as e:
            _LOGGER.error(f"Failed to load config: {e}")
            raise

    # Step 2: Overlay environment variables
    data = _merge_env_overrides(data, env_prefix)

    # Step 3: Overlay CLI overrides
    data = _merge_cli_overrides(data, cli_overrides)

    # Step 4: Validate and create model
    try:
        config = ContentDownloadConfig.model_validate(data)
        _LOGGER.info(f"Configuration validated. Config hash: {config.config_hash()[:8]}...")
        return config
    except Exception as e:
        _LOGGER.error(f"Configuration validation failed: {e}")
        raise


def validate_config_file(path: str) -> bool:
    """
    Validate a config file without loading it fully.

    Useful for `validate-config` CLI command.

    Args:
        path: Path to config file

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    try:
        _ = load_config(path=path)
        return True
    except Exception as e:
        _LOGGER.error(f"Config validation failed: {e}")
        raise


def export_config_schema() -> dict[str, Any]:
    """
    Export JSON Schema for ContentDownloadConfig.

    Useful for documentation and IDE integration.

    Returns:
        JSON schema dict (Pydantic v2 format)
    """
    return ContentDownloadConfig.model_json_schema()
