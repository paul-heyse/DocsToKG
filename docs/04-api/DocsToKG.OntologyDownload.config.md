# Module: config

Ontology Download Configuration

This module centralizes configuration models, parsing helpers, and validation
logic for the ontology downloader service. It reads YAML configuration files,
merges default values, applies environment overrides, and ensures that the
resulting settings are ready for downstream document processing and download
pipelines.

## Functions

### `ensure_python_version()`

Ensure the interpreter meets the minimum supported version.

Args:
None

Returns:
None

Raises:
SystemExit: If Python is older than the minimum version supported by
the ontology downloader tooling.

### `_coerce_sequence(value)`

Normalize configuration entries into a list of strings.

Args:
value: Raw configuration value which may be None, a string, or an
iterable of string-like items.

Returns:
List of string values that are safe for downstream processing.

### `get_env_overrides()`

Extract ontology downloader overrides from environment variables.

Args:
None

Returns:
Mapping of normalized environment keys (lowercase without prefix) to
their string values for supported overrides.

### `_build_defaults(raw_defaults)`

*No documentation available.*

### `_apply_env_overrides(defaults)`

*No documentation available.*

### `merge_defaults(ontology_spec, defaults)`

*No documentation available.*

### `build_resolved_config(raw_config)`

*No documentation available.*

### `_make_fetch_spec(ontology_id, resolver, extras, target_formats)`

Instantiate a FetchSpec from raw YAML fields.

Args:
ontology_id: Identifier of the ontology to retrieve.
resolver: Resolver backend to use when locating documents.
extras: Additional resolver-specific settings.
target_formats: Desired normalization output formats.

Returns:
Fetch specification ready for ontology download orchestration.

### `validate_config(config_path)`

Validate a configuration file and return the resolved settings.

Args:
config_path: File path to a YAML config describing ontology downloads.

Returns:
Resolved configuration containing defaults and fetch specifications.

Raises:
ConfigError: If the configuration fails schema validation.
SystemExit: If the file does not exist.

### `_validate_schema(raw, config)`

*No documentation available.*

### `load_raw_yaml(config_path)`

Load and parse a YAML configuration file into a mutable mapping.

Args:
config_path: Path to the YAML file to parse.

Returns:
Parsed mapping-compatible object suitable for validation.

Raises:
SystemExit: If the file cannot be located.
ConfigError: If the YAML structure is invalid.

### `load_config(config_path)`

Load configuration from disk without performing schema validation.

Args:
config_path: Path to the YAML file describing ontology downloads.

Returns:
Resolved configuration with defaults merged and fetch specs created.

### `rate_limit_per_second(self)`

Convert the textual rate limit to a per-second float value.

Args:
None

Returns:
Number of allowed download tokens per second for each host.

Raises:
ConfigError: If the configured rate limit does not use a per-second
unit or the numeric portion cannot be parsed.

### `from_defaults(cls)`

*No documentation available.*

### `_parse_scalar(value)`

*No documentation available.*

### `_peek_next(lines, start)`

*No documentation available.*

### `_safe_load(text)`

*No documentation available.*

### `safe_load(text)`

*No documentation available.*

## Classes

### `ConfigError`

Raised when ontology configuration files are invalid or inconsistent.

### `LoggingConfiguration`

Structured logging options for ontology download operations.

Attributes:
level: Logging level name (e.g., `INFO`, `DEBUG`).
max_log_size_mb: Maximum individual log file size before rotation.
retention_days: Number of days to keep logs prior to compression.

### `ValidationConfig`

Validation limits governing parser execution.

### `DownloadConfiguration`

HTTP download and retry settings for ontology sources.

Attributes:
max_retries: Number of retries for transient download failures.
timeout_sec: Timeout for request connection and read operations.
download_timeout_sec: Upper bound for whole file downloads.
backoff_factor: Backoff multiplier used between retry attempts.
per_host_rate_limit: Token bucket rate limit definition per host.
max_download_size_gb: Maximum allowed ontology archive size.

### `DefaultsConfiguration`

Collection of default settings applied to ontology fetch specifications.

Attributes:
accept_licenses: License identifiers allowed for automated downloads.
normalize_to: Output formats to produce after ontology normalization.
prefer_source: Resolver preference order when one is not specified.
http: Download configuration shared across ontology fetch operations.
validation: Validation configuration for ontologies post-download.
logging: Logging configuration applied to the downloader runtime.
continue_on_error: Whether to proceed after non-fatal download errors.
concurrent_downloads: Maximum concurrent download workers to run.

### `ResolvedConfig`

Container for merged configuration defaults and fetch specifications.

Attributes:
defaults: Finalized default settings applied to fetch specs.
specs: Sequence of ontology fetch specifications to execute.

### `_YAMLError`

*No documentation available.*

### `_FallbackYAML`

*No documentation available.*
