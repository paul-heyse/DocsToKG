"""Configuration helpers for the ontology downloader."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import logging

import yaml

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .core import FetchSpec


class ConfigError(RuntimeError):
    """Raised when configuration files are invalid."""


PYTHON_MIN_VERSION = (3, 9)


def ensure_python_version() -> None:
    """Ensure the interpreter meets the minimum supported version."""

    if sys.version_info < PYTHON_MIN_VERSION:
        print("Error: Python 3.9+ required", file=sys.stderr)
        raise SystemExit(1)


@dataclass(slots=True)
class LoggingConfiguration:
    level: str = "INFO"
    max_log_size_mb: int = 100
    retention_days: int = 30


@dataclass(slots=True)
class ValidationConfiguration:
    skip_reasoning_if_size_mb: int = 500
    parser_timeout_sec: int = 60


@dataclass(slots=True)
class DownloadConfiguration:
    max_retries: int = 5
    timeout_sec: int = 30
    download_timeout_sec: int = 300
    backoff_factor: float = 0.5
    per_host_rate_limit: str = "4/second"
    max_download_size_gb: int = 5

    def rate_limit_per_second(self) -> float:
        value, _, unit = self.per_host_rate_limit.partition("/")
        try:
            tokens = float(value)
        except ValueError as exc:  # pragma: no cover - validated earlier
            raise ConfigError(f"Invalid per_host_rate_limit value: {self.per_host_rate_limit}") from exc
        if unit.strip() not in {"second", "sec", "s"}:
            raise ConfigError("per_host_rate_limit must be expressed per second")
        return tokens


@dataclass(slots=True)
class DefaultsConfiguration:
    accept_licenses: Sequence[str] = field(default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"])
    normalize_to: Sequence[str] = field(default_factory=lambda: ["ttl"])
    prefer_source: Sequence[str] = field(default_factory=lambda: ["obo", "ols", "bioportal", "direct"])
    http: DownloadConfiguration = field(default_factory=DownloadConfiguration)
    validation: ValidationConfiguration = field(default_factory=ValidationConfiguration)
    logging: LoggingConfiguration = field(default_factory=LoggingConfiguration)
    continue_on_error: bool = True
    concurrent_downloads: int = 1


DEFAULT_MAX_CONCURRENT_DOWNLOADS = 3


@dataclass(slots=True)
class ResolvedConfig:
    defaults: DefaultsConfiguration
    specs: Sequence["FetchSpec"]

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        return cls(defaults=DefaultsConfiguration(), specs=())


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def get_env_overrides() -> Dict[str, str]:
    prefix = "ONTOFETCH_"
    return {key[len(prefix) :].lower(): value for key, value in os.environ.items() if key.startswith(prefix)}


def merge_defaults(raw_config: Mapping[str, object]) -> ResolvedConfig:
    defaults_section = raw_config.get("defaults", {})
    if not isinstance(defaults_section, Mapping):
        raise ConfigError("'defaults' section must be a mapping")

    http_defaults = defaults_section.get("http", {})
    validation_defaults = defaults_section.get("validation", {})
    logging_defaults = defaults_section.get("logging", {})

    defaults = DefaultsConfiguration(
        accept_licenses=_coerce_sequence(defaults_section.get("accept_licenses")) or DefaultsConfiguration().accept_licenses,
        normalize_to=_coerce_sequence(defaults_section.get("normalize_to")) or DefaultsConfiguration().normalize_to,
        prefer_source=_coerce_sequence(defaults_section.get("prefer_source")) or DefaultsConfiguration().prefer_source,
        http=DownloadConfiguration(
            max_retries=int(http_defaults.get("max_retries", 5)),
            timeout_sec=int(http_defaults.get("timeout_sec", 30)),
            download_timeout_sec=int(http_defaults.get("download_timeout_sec", 300)),
            backoff_factor=float(http_defaults.get("backoff_factor", 0.5)),
            per_host_rate_limit=str(http_defaults.get("per_host_rate_limit", "4/second")),
            max_download_size_gb=int(http_defaults.get("max_download_size_gb", 5)),
        ),
        validation=ValidationConfiguration(
            skip_reasoning_if_size_mb=int(validation_defaults.get("skip_reasoning_if_size_mb", 500)),
            parser_timeout_sec=int(validation_defaults.get("parser_timeout_sec", 60)),
        ),
        logging=LoggingConfiguration(
            level=str(logging_defaults.get("level", "INFO")),
            max_log_size_mb=int(logging_defaults.get("max_log_size_mb", 100)),
            retention_days=int(logging_defaults.get("retention_days", 30)),
        ),
        continue_on_error=bool(defaults_section.get("continue_on_error", True)),
        concurrent_downloads=int(defaults_section.get("concurrent_downloads", 1)),
    )

    env_overrides = get_env_overrides()
    if "max_retries" in env_overrides:
        defaults.http.max_retries = int(env_overrides["max_retries"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_MAX_RETRIES", "value": env_overrides["max_retries"]},
        )
    if "timeout_sec" in env_overrides:
        defaults.http.timeout_sec = int(env_overrides["timeout_sec"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_TIMEOUT_SEC", "value": env_overrides["timeout_sec"]},
        )
    if "per_host_rate_limit" in env_overrides:
        defaults.http.per_host_rate_limit = env_overrides["per_host_rate_limit"]
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_PER_HOST_RATE_LIMIT", "value": env_overrides["per_host_rate_limit"]},
        )
    if "backoff_factor" in env_overrides:
        defaults.http.backoff_factor = float(env_overrides["backoff_factor"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_BACKOFF_FACTOR", "value": env_overrides["backoff_factor"]},
        )
    if "log_level" in env_overrides:
        defaults.logging.level = str(env_overrides["log_level"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_LOG_LEVEL", "value": env_overrides["log_level"]},
        )

    ontologies = raw_config.get("ontologies", [])
    if not isinstance(ontologies, list):
        raise ConfigError("'ontologies' must be a list")

    fetch_specs: List["FetchSpec"] = []
    for index, entry in enumerate(ontologies, start=1):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Ontology entry #{index} must be a mapping")
        ontology_id = str(entry.get("id"))
        resolver = str(entry.get("resolver", defaults.prefer_source[0]))
        extras = dict(entry.get("extras", {}))
        target_formats = _coerce_sequence(entry.get("target_formats") or defaults.normalize_to)
        fetch_specs.append(_make_fetch_spec(ontology_id, resolver, extras, target_formats))

    return ResolvedConfig(defaults=defaults, specs=fetch_specs)


def _make_fetch_spec(
    ontology_id: str,
    resolver: str,
    extras: Mapping[str, object],
    target_formats: Sequence[str],
) -> "FetchSpec":
    from .core import FetchSpec as _FetchSpec  # Local import avoids circular dependency

    return _FetchSpec(id=ontology_id, resolver=resolver, extras=dict(extras), target_formats=target_formats)


def validate_config(config_path: Path) -> ResolvedConfig:
    raw = load_raw_yaml(config_path)
    return merge_defaults(raw)


def load_raw_yaml(config_path: Path) -> MutableMapping[str, object]:
    try:
        text = config_path.read_text()
    except FileNotFoundError as exc:  # pragma: no cover - depends on filesystem state
        raise ConfigError(f"Configuration file not found: {config_path}") from exc

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            mark = exc.problem_mark
            raise ConfigError(f"Error in sources.yaml line {mark.line + 1}: {exc}") from exc
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(data, MutableMapping):
        raise ConfigError("Configuration root must be a mapping")
    return data


def load_config(config_path: Path) -> ResolvedConfig:
    raw = load_raw_yaml(config_path)
    return merge_defaults(raw)


__all__ = [
    "ConfigError",
    "DefaultsConfiguration",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ResolvedConfig",
    "ValidationConfiguration",
    "DEFAULT_MAX_CONCURRENT_DOWNLOADS",
    "ensure_python_version",
    "get_env_overrides",
    "load_config",
    "validate_config",
]
LOGGER = logging.getLogger("DocsToKG.OntologyDownload")

