"""Public helpers for loading DocParsing configuration documents."""

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
