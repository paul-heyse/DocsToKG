# 1. Module: config

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.config``.

Ontology Downloader Configuration

This module centralizes configuration schema definitions, environment
overrides, and YAML parsing for DocsToKG's ontology downloader. It builds on
Pydantic v2 models to provide strong validation, type-safe defaults, automatic
JSON Schema generation, and runtime mutability where operational overrides are
required.

Key Features:
- Declarative Pydantic models for HTTP, validation, and logging settings
- YAML loading with structural validation and friendly error messages
- Environment variable overrides merged via :class:`pydantic_settings.BaseSettings`
- JSON Schema export for documentation and validation tooling
- Utilities to merge defaults with ad-hoc fetch specifications

Dependencies:
- PyYAML for configuration parsing
- pydantic and pydantic-settings for model validation

Usage:
    from DocsToKG.OntologyDownload.config import load_config

    resolved = load_config(Path("sources.yaml"))
    for spec in resolved.specs:
        print(spec.id)

## 1. Functions

### `ensure_python_version()`

Ensure the interpreter meets the minimum supported Python version.

Args:
None

Returns:
None

Raises:
SystemExit: If the current interpreter version is below the minimum.

### `_coerce_sequence(value)`

Normalize configuration entries into a list of strings.

Args:
value: Input value that may be ``None``, a string, or an iterable of strings.

Returns:
List of strings suitable for downstream configuration processing.

### `get_env_overrides()`

Return raw environment override values for backwards compatibility.

Args:
None

Returns:
Mapping of configuration field name to override value sourced from the environment.

### `_apply_env_overrides(defaults)`

Apply environment variable overrides to default configuration.

Args:
defaults: Defaults configuration object that will be mutated in place.

Returns:
None

### `_make_fetch_spec(ontology_id, resolver, extras, target_formats)`

Instantiate a FetchSpec from raw YAML fields.

Args:
ontology_id: Identifier of the ontology being configured.
resolver: Name of the resolver responsible for fetching content.
extras: Additional resolver-specific configuration parameters.
target_formats: Desired output formats for the ontology artefact.

Returns:
Fully initialised ``FetchSpec`` object ready for execution.

### `merge_defaults(ontology_spec, defaults)`

Merge an ontology specification with resolved default settings.

Args:
ontology_spec: Raw ontology specification mapping loaded from YAML.
defaults: Optional resolved defaults to merge with the specification.

Returns:
FetchSpec instance describing the fully-merged ontology configuration.

Raises:
ConfigError: If required fields are missing or invalid in the specification.

### `build_resolved_config(raw_config)`

Construct fully-resolved configuration from raw YAML contents.

Args:
raw_config: Parsed YAML mapping defining defaults and ontologies.

Returns:
ResolvedConfig combining defaults and individual fetch specifications.

Raises:
ConfigError: If validation fails or required sections are missing.

### `_validate_schema(raw, config)`

Perform additional structural validation beyond Pydantic models.

Args:
raw: Raw configuration mapping used for structural checks.
config: Optional resolved configuration for cross-field validation.

Returns:
None

Raises:
ConfigError: When structural constraints are violated.

### `load_raw_yaml(config_path)`

Load and parse a YAML configuration file into a mutable mapping.

Args:
config_path: Path to the YAML configuration file.

Returns:
Mutable mapping representation of the YAML content.

Raises:
SystemExit: If the file is not found on disk.
ConfigError: If the YAML cannot be parsed or is structurally invalid.

### `load_config(config_path)`

Load configuration from disk without performing additional schema validation.

Args:
config_path: Path to the YAML configuration file.

Returns:
ResolvedConfig produced from the raw file contents.

Raises:
ConfigError: If the configuration cannot be parsed or validated.

### `validate_config(config_path)`

Validate a configuration file and return the resolved settings.

Args:
config_path: Path to the YAML configuration file.

Returns:
ResolvedConfig object after validation.

Raises:
ConfigError: If validation fails.

### `validate_level(cls, value)`

Validate logging level values.

Args:
value: Logging level provided in configuration.

Returns:
Uppercase logging level string accepted by :mod:`logging`.

Raises:
ValueError: If the supplied level is not among the supported options.

### `validate_rate_limits(cls, value)`

Ensure rate limit strings follow the expected pattern.

Args:
value: Mapping of service name to rate limit expression.

Returns:
Validated mapping preserving the original values.

Raises:
ValueError: If any rate limit fails to match the accepted pattern.

### `rate_limit_per_second(self)`

Convert ``per_host_rate_limit`` to a requests-per-second value.

Args:
None

Returns:
Floating-point requests-per-second value derived from configuration.

Raises:
ValueError: If the configured rate limit string is invalid.

### `parse_service_rate_limit(self, service)`

Parse a per-service rate limit to requests per second.

Args:
service: Logical service name to look up.

Returns:
Requests-per-second value when configured, otherwise ``None``.

Raises:
None.

### `normalized_allowed_hosts(self)`

Return allowlist entries normalized to lowercase punycode labels.

Args:
None

Returns:
Tuple of (exact hostnames, wildcard suffixes) when entries exist,
otherwise ``None`` if no valid allowlist entries are configured.

Raises:
ValueError: If any configured hostname cannot be converted to punycode.

### `polite_http_headers(self)`

Return polite HTTP headers suitable for resolver API calls.

The headers include a deterministic ``User-Agent`` string, propagate a
``From`` contact address when configured, and synthesize an ``X-Request-ID``
correlated with the current fetch so API providers can trace requests.

Args:
correlation_id: Identifier attached to the current batch for log correlation.
request_id: Optional override for ``X-Request-ID`` header; auto-generated when ``None``.
timestamp: Optional timestamp used when constructing the request identifier.

Returns:
Mapping of header names to polite values complying with provider guidelines.

### `validate_prefer_source(cls, value)`

Ensure resolver names are recognized.

Args:
value: Ordered list of resolver names provided in configuration.

Returns:
Validated list containing only supported resolver identifiers.

Raises:
ValueError: If any resolver name is not part of the supported set.

### `from_defaults(cls)`

Create an empty resolved configuration with library defaults.

Args:
None

Returns:
ResolvedConfig populated with default settings and no fetch specs.

## 2. Classes

### `ConfigError`

Raised when ontology configuration files are invalid or inconsistent.

Attributes:
message: Human-readable explanation of the configuration flaw.

Examples:
>>> try:
...     raise ConfigError("missing id")
... except ConfigError as exc:
...     assert "missing id" in str(exc)

### `LoggingConfiguration`

Structured logging options for ontology download operations.

Attributes:
level: Logging level for downloader telemetry (DEBUG, INFO, etc.).
max_log_size_mb: Maximum size of log files before rotation occurs.
retention_days: Number of days log files are retained on disk.

Examples:
>>> config = LoggingConfiguration(level="debug")
>>> config.level
'DEBUG'

### `ValidationConfig`

Validation limits governing parser execution.

Attributes:
parser_timeout_sec: Maximum runtime allowed for ontology parsing.
max_memory_mb: Memory ceiling allocated to validation routines.
skip_reasoning_if_size_mb: Threshold above which reasoning is skipped.

Examples:
>>> ValidationConfig(parser_timeout_sec=120).parser_timeout_sec
120

### `DownloadConfiguration`

HTTP download, throttling, and polite header settings for resolvers.

Attributes:
max_retries: Maximum retry attempts for failed download requests.
timeout_sec: Base timeout applied to metadata requests.
download_timeout_sec: Timeout applied to streaming download operations.
backoff_factor: Exponential backoff multiplier between retry attempts.
per_host_rate_limit: Token bucket rate limit string for host throttling.
max_download_size_gb: Maximum permitted download size in gigabytes.
concurrent_downloads: Allowed number of simultaneous downloads.
validate_media_type: Whether to enforce Content-Type validation.
rate_limits: Optional per-service rate limit overrides.
allowed_hosts: Optional allowlist restricting download hostnames.
polite_headers: Default polite HTTP headers applied to resolver API calls.

Examples:
>>> cfg = DownloadConfiguration(per_host_rate_limit="10/second")
>>> round(cfg.rate_limit_per_second(), 2)
10.0

### `DefaultsConfig`

Collection of default settings for ontology fetch specifications.

Attributes:
accept_licenses: Licenses accepted by default during downloads.
normalize_to: Preferred formats for normalized ontology output.
prefer_source: Ordered list of resolver priorities.
http: Download configuration defaults.
validation: Validation configuration defaults.
logging: Logging configuration defaults.
continue_on_error: Whether processing continues after failures.
resolver_fallback_enabled: Whether automatic resolver fallback is enabled.

Examples:
>>> defaults = DefaultsConfig()
>>> "obo" in defaults.prefer_source
True

### `ResolvedConfig`

Container for merged configuration defaults and fetch specifications.

Attributes:
defaults: Default configuration applied to all ontology fetch specs.
specs: List of individual fetch specifications after merging defaults.

Examples:
>>> config = ResolvedConfig.from_defaults()
>>> isinstance(config.defaults, DefaultsConfig)
True

### `EnvironmentOverrides`

Environment variable overrides for ontology downloader defaults.

Attributes:
max_retries: Override for retry count via ``ONTOFETCH_MAX_RETRIES``.
timeout_sec: Override for metadata timeout via ``ONTOFETCH_TIMEOUT_SEC``.
download_timeout_sec: Override for streaming timeout.
per_host_rate_limit: Override for per-host rate limit string.
backoff_factor: Override for exponential backoff multiplier.
log_level: Override for logging level.

Examples:
>>> overrides = EnvironmentOverrides()
>>> overrides.model_config['env_prefix']
'ONTOFETCH_'
