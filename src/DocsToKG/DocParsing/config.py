"""
Configuration helpers for DocParsing stages.

This module isolates the machinery required to hydrate stage configuration
objects from environment variables, CLI arguments, and on-disk configuration
files. Keeping these helpers separate from the CLI entry points makes unit
testing less cumbersome and prevents import-time side effects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Set, Tuple


def _load_yaml_markers(raw: str) -> object:
    """Deserialize YAML configuration content, raising for missing dependencies."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Loading YAML configuration requires the 'PyYAML' package (pip install PyYAML)."
        ) from exc
    return yaml.safe_load(raw)


def _load_toml_markers(raw: str) -> object:
    """Deserialize TOML configuration content with compatibility fallbacks."""

    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - fallback path
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Loading TOML configuration requires the 'tomli' package (pip install tomli)."
            ) from exc
    return tomllib.loads(raw)


def load_config_mapping(path: Path) -> Dict[str, Any]:
    """Load a configuration mapping from JSON, YAML, or TOML."""

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml_markers(raw)
    elif suffix == ".toml":
        data = _load_toml_markers(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"Stage configuration file {path} must contain an object; received {type(data).__name__}."
        )
    return data


def _manifest_value(value: Any) -> Any:
    """Convert values to manifest-friendly representations."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    if isinstance(value, list):
        return [_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _manifest_value(val) for key, val in value.items()}
    return value


def _coerce_path(value: object, base_dir: Optional[Path] = None) -> Path:
    """Convert ``value`` into an absolute :class:`Path`."""

    if isinstance(value, Path):
        path = value
    else:
        path = Path(str(value))
    if base_dir is not None and not path.is_absolute():
        path = (base_dir / path).expanduser()
    else:
        path = path.expanduser()
    return path.resolve()


def _coerce_optional_path(value: object, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Convert optional path-like values."""

    if value in (None, "", False):
        return None
    return _coerce_path(value, base_dir)


def _coerce_bool(value: object, _base_dir: Optional[Path] = None) -> bool:
    """Convert truthy strings or numbers to boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(normalized)


def _coerce_int(value: object, _base_dir: Optional[Path] = None) -> int:
    """Convert ``value`` to ``int``."""

    if isinstance(value, int):
        return value
    return int(str(value).strip())


def _coerce_float(value: object, _base_dir: Optional[Path] = None) -> float:
    """Convert ``value`` to ``float``."""

    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return float(str(value).strip())


def _coerce_str(value: object, _base_dir: Optional[Path] = None) -> str:
    """Return ``value`` coerced to string."""

    return str(value)


def _coerce_str_tuple(value: object, _base_dir: Optional[Path] = None) -> Tuple[str, ...]:
    """Return ``value`` as a tuple of strings."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        flattened: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                for sub_item in item:
                    if sub_item is None:
                        continue
                    text = str(sub_item).strip()
                    if text:
                        flattened.append(text)
            else:
                text = str(item).strip()
                if text:
                    flattened.append(text)
        return tuple(flattened)
    text = str(value).strip()
    if not text:
        return ()
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple, set)):
                return tuple(str(item) for item in parsed)
        except json.JSONDecodeError:
            pass
    separators = [",", ";"]
    parts = [text]
    for sep in separators:
        if sep in text:
            parts = [segment.strip() for segment in text.split(sep)]
            break
    return tuple(part for part in parts if part)


@dataclass
class StageConfigBase:
    """Base dataclass for stage configuration objects."""

    config: Optional[Path] = None
    overrides: Set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
        metadata={"manifest": False},
    )

    ENV_VARS: ClassVar[Dict[str, str]] = {}
    FIELD_PARSERS: ClassVar[Dict[str, Callable[[Any, Optional[Path]], Any]]] = {}

    def apply_env(self) -> None:
        """Overlay configuration from environment variables."""

        for field_name, env_name in self.ENV_VARS.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            new_value = self._coerce_field(field_name, raw, None)
            current = getattr(self, field_name, None)
            if new_value == current:
                continue
            setattr(self, field_name, new_value)
            self.overrides.add(field_name)

    def update_from_file(self, cfg_path: Path) -> None:
        """Overlay configuration from ``cfg_path``."""

        mapping = load_config_mapping(cfg_path)
        base_dir = cfg_path.parent
        for key, value in mapping.items():
            if not hasattr(self, key):
                continue
            new_value = self._coerce_field(key, value, base_dir)
            current = getattr(self, key, None)
            if new_value == current:
                continue
            setattr(self, key, new_value)
            self.overrides.add(key)
        self.config = _coerce_optional_path(cfg_path, None)
        self.overrides.add("config")

    def apply_args(self, args: Any) -> None:
        """Overlay configuration from an argparse namespace."""

        for field_def in fields(self):
            name = field_def.name
            if not hasattr(args, name):
                continue
            value = getattr(args, name)
            if value is None:
                continue
            new_value = self._coerce_field(name, value, None)
            current = getattr(self, name, None)
            if new_value == current:
                continue
            setattr(self, name, new_value)
            self.overrides.add(name)

    @classmethod
    def from_env(cls) -> "StageConfigBase":
        """Instantiate a configuration populated solely from environment variables."""

        cfg = cls()
        cfg.apply_env()
        cfg.finalize()
        return cfg

    def finalize(self) -> None:  # pragma: no cover - overridden by subclasses
        """Hook allowing subclasses to normalise derived fields."""

    def to_manifest(self) -> Dict[str, Any]:
        """Return a manifest-friendly snapshot of the configuration."""

        payload: Dict[str, Any] = {}
        for field_def in fields(self):
            if not field_def.metadata.get("manifest", True):
                continue
            payload[field_def.name] = _manifest_value(getattr(self, field_def.name))
        return payload

    def _coerce_field(self, name: str, value: Any, base_dir: Optional[Path]) -> Any:
        """Run field-specific coercion logic before manifest serialization."""

        parser = self.FIELD_PARSERS.get(name)
        if parser is None:
            return value
        return parser(value, base_dir)

    def is_overridden(self, field_name: str) -> bool:
        """Return ``True`` when ``field_name`` was explicitly overridden."""

        return field_name in self.overrides

    _coerce_path = staticmethod(_coerce_path)
    _coerce_optional_path = staticmethod(_coerce_optional_path)
    _coerce_bool = staticmethod(_coerce_bool)
    _coerce_int = staticmethod(_coerce_int)
    _coerce_float = staticmethod(_coerce_float)
    _coerce_str = staticmethod(_coerce_str)
    _coerce_str_tuple = staticmethod(_coerce_str_tuple)


__all__ = [
    "StageConfigBase",
    "load_config_mapping",
    "_coerce_bool",
    "_coerce_float",
    "_coerce_int",
    "_coerce_optional_path",
    "_coerce_path",
    "_coerce_str",
    "_coerce_str_tuple",
]
