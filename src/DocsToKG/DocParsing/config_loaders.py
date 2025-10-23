# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.config_loaders",
#   "purpose": "Utility loaders that hydrate DocParsing configs from YAML and TOML sources.",
#   "sections": [
#     {
#       "id": "configloaderror",
#       "name": "ConfigLoadError",
#       "anchor": "class-configloaderror",
#       "kind": "class"
#     },
#     {
#       "id": "load-yaml-markers",
#       "name": "load_yaml_markers",
#       "anchor": "function-load-yaml-markers",
#       "kind": "function"
#     },
#     {
#       "id": "load-toml-markers",
#       "name": "load_toml_markers",
#       "anchor": "function-load-toml-markers",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Utility loaders that hydrate DocParsing configs from YAML and TOML sources.

DocParsing stages rely on human-editable marker files to tune structural
heuristics and tokenizer behaviour. This module centralises the deserialisation
logic for those documents, wrapping third-party parsers with helpful error
messages so operators understand when optional dependencies such as PyYAML are
required. It intentionally keeps the surface minimal—returning plain Python data
structures—so higher level configuration modules can compose and validate them
without inheriting parser-specific concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ConfigLoadError",
    "load_toml_markers",
    "load_yaml_markers",
]


@dataclass(slots=True)
class ConfigLoadError(RuntimeError):
    """Raised when configuration documents cannot be deserialized."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - dataclass convenience
        """Return the stored error message for human-facing output."""

        return self.message


def load_yaml_markers(raw: str) -> Any:
    """Deserialize structural marker configuration expressed as YAML."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ConfigLoadError(
            "Loading YAML configuration requires the 'PyYAML' package (pip install PyYAML)."
        ) from exc

    try:
        return yaml.safe_load(raw)
    except Exception as exc:  # pragma: no cover - delegated to yaml for detail
        raise ConfigLoadError("Failed to parse YAML configuration payload") from exc


def load_toml_markers(raw: str) -> Any:
    """Deserialize structural marker configuration expressed as TOML."""

    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - fallback path for <3.11
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigLoadError(
                "Loading TOML configuration requires the 'tomli' package (pip install tomli)."
            ) from exc

    try:
        return tomllib.loads(raw)
    except (tomllib.TOMLDecodeError, ValueError, TypeError) as exc:
        raise ConfigLoadError("Failed to parse TOML configuration payload") from exc
