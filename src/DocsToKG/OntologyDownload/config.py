"""
Ontology Download Configuration

This module centralizes configuration models, parsing helpers, and validation
logic for the ontology downloader service. It reads YAML configuration files,
merges default values, applies environment overrides, and ensures that the
resulting settings are ready for downstream document processing and download
pipelines.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight YAML fallback for tests

    class _YAMLError(Exception):
        pass

    def _parse_scalar(value: str):
        lowered = value.lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
        if lowered in {"null", "none", ""}:
            return None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part.strip()) for part in inner.split(",")]
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _peek_next(lines, start: int):
        for index in range(start, len(lines)):
            raw = lines[index]
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            return indent, stripped
        return None

    def _safe_load(text: str):
        lines = text.splitlines()
        stack: List[tuple[int, object]] = []
        root: Optional[object] = None
        index = 0
        while index < len(lines):
            raw = lines[index]
            stripped = raw.strip()
            index += 1
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            while stack and indent < stack[-1][0]:
                stack.pop()
            parent = stack[-1][1] if stack else None

            if stripped.startswith("- "):
                if parent is None or not isinstance(parent, list):
                    raise _YAMLError("List item without list context")
                entry_text = stripped[2:].strip()
                if not entry_text:
                    item: object = {}
                    parent.append(item)
                    stack.append((indent + 2, item))
                elif ":" in entry_text:
                    key, value = entry_text.split(":", 1)
                    item = {key.strip(): _parse_scalar(value.strip())}
                    parent.append(item)
                    stack.append((indent + 2, item))
                else:
                    parent.append(_parse_scalar(entry_text))
                continue

            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                if parent is None:
                    if root is None:
                        root = {}
                        stack.append((indent, root))
                        parent = root
                    elif isinstance(root, dict):
                        parent = root
                    else:
                        raise _YAMLError("Multiple root nodes not supported")
                if value == "":
                    next_info = _peek_next(lines, index)
                    if next_info and next_info[0] > indent and next_info[1].startswith("- "):
                        new_container: object = []
                    else:
                        new_container = {}
                    parent[key] = new_container  # type: ignore[index]
                    stack.append((indent + 2, new_container))
                else:
                    parent[key] = _parse_scalar(value)  # type: ignore[index]
                continue

            raise _YAMLError(f"Unsupported YAML syntax: {stripped}")

        return root or {}

    class _FallbackYAML:
        YAMLError = _YAMLError

        @staticmethod
        def safe_load(text: str):
            """Parse a YAML string using the lightweight fallback implementation.

            Args:
                text: Raw YAML content to be parsed into Python structures.

            Returns:
                Parsed Python object mirroring the YAML structure.

            Raises:
                _YAMLError: If the payload contains unsupported YAML constructs.
            """
            return _safe_load(text)

    yaml = _FallbackYAML()  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .core import FetchSpec


class ConfigError(RuntimeError):
    """Raised when ontology configuration files are invalid or inconsistent.

    Args:
        message: Description of the configuration error encountered.

    Examples:
        >>> raise ConfigError("Missing ontology id")
        Traceback (most recent call last):
        ...
        ConfigError: Missing ontology id
    """


PYTHON_MIN_VERSION = (3, 9)


def ensure_python_version() -> None:
    """Ensure the interpreter meets the minimum supported version.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: If Python is older than the minimum version supported by
            the ontology downloader tooling.
    """

    if sys.version_info < PYTHON_MIN_VERSION:
        print("Error: Python 3.9+ required", file=sys.stderr)
        raise SystemExit(1)


@dataclass(slots=True)
class LoggingConfiguration:
    """Structured logging options for ontology download operations.

    Attributes:
        level: Logging level name (e.g., `INFO`, `DEBUG`).
        max_log_size_mb: Maximum individual log file size before rotation.
        retention_days: Number of days to keep logs prior to compression.

    Examples:
        >>> logging_config = LoggingConfiguration(level="DEBUG", retention_days=7)
        >>> logging_config.level
        'DEBUG'
    """

    level: str = "INFO"
    max_log_size_mb: int = 100
    retention_days: int = 30


@dataclass(slots=True)
class ValidationConfig:
    """Validation limits governing parser execution.

    Attributes:
        parser_timeout_sec: Maximum runtime allowed per validator in seconds.
        max_memory_mb: Memory ceiling in megabytes for validation processes.
        skip_reasoning_if_size_mb: Threshold size that disables reasoning steps.

    Examples:
        >>> validation = ValidationConfig(parser_timeout_sec=120, max_memory_mb=1024)
        >>> validation.max_memory_mb
        1024
    """

    parser_timeout_sec: int = 60
    max_memory_mb: int = 2048
    skip_reasoning_if_size_mb: int = 500


@dataclass(slots=True)
class DownloadConfiguration:
    """HTTP download and retry settings for ontology sources.

    Attributes:
        max_retries: Number of retries for transient download failures.
        timeout_sec: Timeout for request connection and read operations.
        download_timeout_sec: Upper bound for whole file downloads.
        backoff_factor: Backoff multiplier used between retry attempts.
        per_host_rate_limit: Token bucket rate limit definition per host.
        max_download_size_gb: Maximum allowed ontology archive size.
        concurrent_downloads: Maximum concurrent download workers to run.

    Examples:
        >>> dl_config = DownloadConfiguration(max_retries=3, per_host_rate_limit="2/second")
        >>> dl_config.max_retries
        3
    """

    max_retries: int = 5
    timeout_sec: int = 30
    download_timeout_sec: int = 300
    backoff_factor: float = 0.5
    per_host_rate_limit: str = "4/second"
    max_download_size_gb: float = 5.0
    concurrent_downloads: int = 1

    def rate_limit_per_second(self) -> float:
        """Convert the textual rate limit to a per-second float value.

        Args:
            None

        Returns:
            Number of allowed download tokens per second for each host.

        Raises:
            ConfigError: If the configured rate limit does not use a per-second
            unit or the numeric portion cannot be parsed.
        """
        value, _, unit = self.per_host_rate_limit.partition("/")
        try:
            tokens = float(value)
        except ValueError as exc:  # pragma: no cover - validated earlier
            raise ConfigError(
                f"Invalid per_host_rate_limit value: {self.per_host_rate_limit}"
            ) from exc
        if unit.strip() not in {"second", "sec", "s"}:
            raise ConfigError("per_host_rate_limit must be expressed per second")
        return tokens


@dataclass(slots=True)
class DefaultsConfig:
    """Collection of default settings applied to ontology fetch specifications.

    Attributes:
        accept_licenses: License identifiers allowed for automated downloads.
        normalize_to: Output formats to produce after ontology normalization.
        prefer_source: Resolver preference order when one is not specified.
        http: Download configuration shared across ontology fetch operations.
        validation: Validation configuration for ontologies post-download.
        logging: Logging configuration applied to the downloader runtime.
        continue_on_error: Whether to proceed after non-fatal download errors.
        concurrent_downloads: Maximum concurrent download workers to run.

    Examples:
        >>> defaults = DefaultsConfig(prefer_source=["obo", "direct"])
        >>> list(defaults.prefer_source)
        ['obo', 'direct']
    """

    accept_licenses: Sequence[str] = field(
        default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
    )
    normalize_to: Sequence[str] = field(default_factory=lambda: ["ttl"])
    prefer_source: Sequence[str] = field(
        default_factory=lambda: ["obo", "ols", "bioportal", "direct"]
    )
    http: DownloadConfiguration = field(default_factory=DownloadConfiguration)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfiguration = field(default_factory=LoggingConfiguration)
    continue_on_error: bool = True


DEFAULT_MAX_CONCURRENT_DOWNLOADS = 3


@dataclass(slots=True)
class ResolvedConfig:
    """Container for merged configuration defaults and fetch specifications.

    Attributes:
        defaults: Finalized default settings applied to fetch specs.
        specs: Sequence of ontology fetch specifications to execute.

    Examples:
        >>> resolved = ResolvedConfig.from_defaults()
        >>> resolved.specs
        ()
    """

    defaults: DefaultsConfig
    specs: Sequence["FetchSpec"]

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        """Create an empty resolved configuration using library defaults.

        Args:
            None

        Returns:
            ResolvedConfig populated with default download parameters and no specs.
        """
        return cls(defaults=DefaultsConfig(), specs=())


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    """Normalize configuration entries into a list of strings.

    Args:
        value: Raw configuration value which may be None, a string, or an
            iterable of string-like items.

    Returns:
        List of string values that are safe for downstream processing.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def get_env_overrides() -> Dict[str, str]:
    """Extract ontology downloader overrides from environment variables.

    Args:
        None

    Returns:
        Mapping of normalized environment keys (lowercase without prefix) to
        their string values for supported overrides.
    """
    prefix = "ONTOFETCH_"
    return {
        key[len(prefix) :].lower(): value
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }


def _build_defaults(raw_defaults: Mapping[str, object]) -> DefaultsConfig:
    template = DefaultsConfig()

    http_defaults = (
        raw_defaults.get("http") if isinstance(raw_defaults.get("http"), Mapping) else {}
    )
    validation_defaults = (
        raw_defaults.get("validation")
        if isinstance(raw_defaults.get("validation"), Mapping)
        else {}
    )
    logging_defaults = (
        raw_defaults.get("logging") if isinstance(raw_defaults.get("logging"), Mapping) else {}
    )

    defaults = DefaultsConfig(
        accept_licenses=_coerce_sequence(raw_defaults.get("accept_licenses"))
        or list(template.accept_licenses),
        normalize_to=_coerce_sequence(raw_defaults.get("normalize_to"))
        or list(template.normalize_to),
        prefer_source=_coerce_sequence(raw_defaults.get("prefer_source"))
        or list(template.prefer_source),
        http=DownloadConfiguration(
            max_retries=int(http_defaults.get("max_retries", template.http.max_retries)),
            timeout_sec=int(http_defaults.get("timeout_sec", template.http.timeout_sec)),
            download_timeout_sec=int(
                http_defaults.get("download_timeout_sec", template.http.download_timeout_sec)
            ),
            backoff_factor=float(http_defaults.get("backoff_factor", template.http.backoff_factor)),
            per_host_rate_limit=str(
                http_defaults.get("per_host_rate_limit", template.http.per_host_rate_limit)
            ),
            max_download_size_gb=float(
                http_defaults.get("max_download_size_gb", template.http.max_download_size_gb)
            ),
            concurrent_downloads=int(
                http_defaults.get("concurrent_downloads", template.http.concurrent_downloads)
            ),
        ),
        validation=ValidationConfig(
            parser_timeout_sec=int(
                validation_defaults.get(
                    "parser_timeout_sec", template.validation.parser_timeout_sec
                )
            ),
            max_memory_mb=int(
                validation_defaults.get("max_memory_mb", template.validation.max_memory_mb)
            ),
            skip_reasoning_if_size_mb=int(
                validation_defaults.get(
                    "skip_reasoning_if_size_mb", template.validation.skip_reasoning_if_size_mb
                )
            ),
        ),
        logging=LoggingConfiguration(
            level=str(logging_defaults.get("level", template.logging.level)),
            max_log_size_mb=int(
                logging_defaults.get("max_log_size_mb", template.logging.max_log_size_mb)
            ),
            retention_days=int(
                logging_defaults.get("retention_days", template.logging.retention_days)
            ),
        ),
        continue_on_error=bool(raw_defaults.get("continue_on_error", template.continue_on_error)),
    )

    return defaults


def _apply_env_overrides(defaults: DefaultsConfig) -> None:
    overrides = get_env_overrides()
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    if "max_retries" in overrides:
        defaults.http.max_retries = int(overrides["max_retries"])
        logger.info(
            "Config overridden by env var: ONTOFETCH_MAX_RETRIES=%s",
            overrides["max_retries"],
            extra={"stage": "config"},
        )
    if "timeout_sec" in overrides:
        defaults.http.timeout_sec = int(overrides["timeout_sec"])
        logger.info(
            "Config overridden by env var: ONTOFETCH_TIMEOUT_SEC=%s",
            overrides["timeout_sec"],
            extra={"stage": "config"},
        )
    if "download_timeout_sec" in overrides:
        defaults.http.download_timeout_sec = int(overrides["download_timeout_sec"])
        logger.info(
            "Config overridden by env var: ONTOFETCH_DOWNLOAD_TIMEOUT_SEC=%s",
            overrides["download_timeout_sec"],
            extra={"stage": "config"},
        )
    if "per_host_rate_limit" in overrides:
        defaults.http.per_host_rate_limit = overrides["per_host_rate_limit"]
        logger.info(
            "Config overridden by env var: ONTOFETCH_PER_HOST_RATE_LIMIT=%s",
            overrides["per_host_rate_limit"],
            extra={"stage": "config"},
        )
    if "backoff_factor" in overrides:
        defaults.http.backoff_factor = float(overrides["backoff_factor"])
        logger.info(
            "Config overridden by env var: ONTOFETCH_BACKOFF_FACTOR=%s",
            overrides["backoff_factor"],
            extra={"stage": "config"},
        )
    if "log_level" in overrides:
        defaults.logging.level = str(overrides["log_level"])
        logger.info(
            "Config overridden by env var: ONTOFETCH_LOG_LEVEL=%s",
            overrides["log_level"],
            extra={"stage": "config"},
        )


def merge_defaults(ontology_spec: Mapping[str, object], defaults: Optional[DefaultsConfig] = None):
    """Merge an ontology specification with resolved default settings.

    Args:
        ontology_spec: Raw ontology specification loaded from configuration.
        defaults: Default configuration block to apply (optional).

    Returns:
        Fetch specification populated with normalized resolver settings.

    Raises:
        ConfigError: If required fields are missing or extras have invalid types.
    """
    if defaults is None:
        return build_resolved_config(ontology_spec)
    ontology_id = str(ontology_spec.get("id", "")).strip()
    if not ontology_id:
        raise ConfigError("Ontology specification missing required field 'id'")
    resolver = str(ontology_spec.get("resolver", defaults.prefer_source[0]))
    extras_value = ontology_spec.get("extras") or {}
    if extras_value is None:
        extras_value = {}
    if not isinstance(extras_value, Mapping):
        raise ConfigError(f"Ontology '{ontology_id}' extras must be a mapping if provided")
    target_formats = _coerce_sequence(ontology_spec.get("target_formats") or defaults.normalize_to)
    return _make_fetch_spec(ontology_id, resolver, extras_value, target_formats)


def build_resolved_config(raw_config: Mapping[str, object]) -> ResolvedConfig:
    """Construct a fully-resolved configuration from raw YAML contents.

    Args:
        raw_config: Mapping produced by YAML parsing containing defaults and ontologies.

    Returns:
        ResolvedConfig containing merged defaults and instantiated fetch specs.

    Raises:
        ConfigError: If the raw configuration violates schema expectations.
    """
    defaults_section = raw_config.get("defaults")
    if defaults_section and not isinstance(defaults_section, Mapping):
        raise ConfigError("'defaults' section must be a mapping")
    defaults = _build_defaults(defaults_section or {})
    _apply_env_overrides(defaults)

    ontologies = raw_config.get("ontologies", [])
    if not isinstance(ontologies, list):
        raise ConfigError("'ontologies' must be a list")

    fetch_specs: List["FetchSpec"] = []
    for index, entry in enumerate(ontologies, start=1):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Ontology entry #{index} must be a mapping")
        fetch_specs.append(merge_defaults(entry, defaults))

    return ResolvedConfig(defaults=defaults, specs=fetch_specs)


def _make_fetch_spec(
    ontology_id: str,
    resolver: str,
    extras: Mapping[str, object],
    target_formats: Sequence[str],
) -> "FetchSpec":
    """Instantiate a FetchSpec from raw YAML fields.

    Args:
        ontology_id: Identifier of the ontology to retrieve.
        resolver: Resolver backend to use when locating documents.
        extras: Additional resolver-specific settings.
        target_formats: Desired normalization output formats.

    Returns:
        Fetch specification ready for ontology download orchestration.
    """
    from .core import FetchSpec as _FetchSpec  # Local import avoids circular dependency

    return _FetchSpec(
        id=ontology_id, resolver=resolver, extras=dict(extras), target_formats=list(target_formats)
    )


def validate_config(config_path: Path) -> ResolvedConfig:
    """Validate a configuration file and return the resolved settings.

    Args:
        config_path: File path to a YAML config describing ontology downloads.

    Returns:
        Resolved configuration containing defaults and fetch specifications.

    Raises:
        ConfigError: If the configuration fails schema validation.
        SystemExit: If the file does not exist.
    """
    raw = load_raw_yaml(config_path)
    config = build_resolved_config(raw)
    _validate_schema(raw, config)
    return config


def _validate_schema(raw: Mapping[str, object], config: Optional[ResolvedConfig] = None) -> None:
    errors: List[str] = []

    defaults = raw.get("defaults")
    if defaults is not None and not isinstance(defaults, Mapping):
        errors.append("'defaults' section must be a mapping when provided")
    if isinstance(defaults, Mapping):
        for key in defaults:
            if key not in {
                "accept_licenses",
                "normalize_to",
                "prefer_source",
                "http",
                "validation",
                "logging",
                "continue_on_error",
            }:
                errors.append(f"Unknown key in defaults: {key}")

        http_section = defaults.get("http")
        if http_section is not None and not isinstance(http_section, Mapping):
            errors.append("'defaults.http' must be a mapping")

        validation_section = defaults.get("validation")
        if validation_section is not None and not isinstance(validation_section, Mapping):
            errors.append("'defaults.validation' must be a mapping")

        logging_section = defaults.get("logging")
        if logging_section is not None and not isinstance(logging_section, Mapping):
            errors.append("'defaults.logging' must be a mapping")
        if isinstance(http_section, Mapping):
            max_retries = http_section.get("max_retries")
            timeout_sec = http_section.get("timeout_sec")
            if max_retries is not None and int(max_retries) < 0:
                errors.append("defaults.http.max_retries must be >= 0")
            if timeout_sec is not None and int(timeout_sec) <= 0:
                errors.append("defaults.http.timeout_sec must be > 0")
        if isinstance(logging_section, Mapping):
            level = logging_section.get("level")
            if level is not None and str(level).upper() not in {
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
            }:
                errors.append("defaults.logging.level must be one of DEBUG, INFO, WARNING, ERROR")

    ontologies = raw.get("ontologies")
    if ontologies is None:
        errors.append("'ontologies' section is required")
    elif not isinstance(ontologies, list):
        errors.append("'ontologies' must be a list of ontology entries")
    else:
        for index, entry in enumerate(ontologies, start=1):
            if not isinstance(entry, Mapping):
                errors.append(f"Ontology entry #{index} must be a mapping")
                continue
            if "id" not in entry:
                errors.append(f"Ontology entry #{index} missing required 'id'")
            if "resolver" in entry and not isinstance(entry["resolver"], str):
                errors.append(
                    f"Ontology entry '{entry.get('id', index)}' field 'resolver' must be a string if provided"
                )
            extras_value = entry.get("extras")
            if extras_value is not None and not isinstance(extras_value, Mapping):
                errors.append(
                    f"Ontology entry '{entry.get('id', index)}' field 'extras' must be a mapping if provided"
                )
            if entry.get("resolver") == "bioportal":
                extras = entry.get("extras", {})
                if not isinstance(extras, Mapping) or "acronym" not in extras:
                    errors.append(
                        f"Ontology entry '{entry.get('id', index)}' using bioportal resolver requires extras.acronym"
                    )
            if "target_formats" in entry and not isinstance(entry["target_formats"], Iterable):
                errors.append(
                    f"Ontology entry '{entry.get('id', index)}' field 'target_formats' must be a list or iterable"
                )

    if errors:
        raise ConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))

    if config is not None:
        if config.defaults.http.max_retries < 0:
            raise ConfigError("defaults.http.max_retries must be >= 0")
        if config.defaults.http.timeout_sec <= 0:
            raise ConfigError("defaults.http.timeout_sec must be > 0")
        if config.defaults.logging.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError("defaults.logging.level must be one of DEBUG, INFO, WARNING, ERROR")


def load_raw_yaml(config_path: Path) -> MutableMapping[str, object]:
    """Load and parse a YAML configuration file into a mutable mapping.

    Args:
        config_path: Path to the YAML file to parse.

    Returns:
        Parsed mapping-compatible object suitable for validation.

    Raises:
        SystemExit: If the file cannot be located.
        ConfigError: If the YAML structure is invalid.
    """
    try:
        text = config_path.read_text()
    except FileNotFoundError:  # pragma: no cover - depends on filesystem state
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

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
    """Load configuration from disk without performing schema validation.

    Args:
        config_path: Path to the YAML file describing ontology downloads.

    Returns:
        Resolved configuration with defaults merged and fetch specs created.

    Raises:
        ConfigError: If the configuration contains invalid structures or values.
    """
    raw = load_raw_yaml(config_path)
    return build_resolved_config(raw)


__all__ = [
    "ConfigError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "LoggingConfig",
    "ResolvedConfig",
    "ValidationConfig",
    "DEFAULT_MAX_CONCURRENT_DOWNLOADS",
    "ensure_python_version",
    "get_env_overrides",
    "load_config",
    "merge_defaults",
    "validate_config",
]

# Backwards compatibility exports
DefaultsConfiguration = DefaultsConfig  # DefaultsConfig is now the canonical name
LoggingConfig = LoggingConfiguration
ValidationConfiguration = ValidationConfig
