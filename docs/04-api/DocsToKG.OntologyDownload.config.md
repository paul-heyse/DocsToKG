# 1. Module: config

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.config``.

## 1. Overview

Configuration models, parsing, and validation helpers.

## 2. Functions

### `ensure_python_version()`

Ensure the interpreter meets the minimum supported Python version.

### `_coerce_sequence(value)`

*No documentation available.*

### `__getattr__(name)`

Lazily expose pipeline exceptions without introducing import cycles.

### `parse_rate_limit_to_rps(limit_str)`

Convert a rate limit expression into requests-per-second.

### `get_env_overrides()`

Return environment-derived overrides as stringified key/value pairs.

### `_apply_env_overrides(defaults)`

Mutate ``defaults`` in-place using values from :class:`EnvironmentOverrides`.

### `build_resolved_config(raw_config)`

Materialise a :class:`ResolvedConfig` from a raw mapping loaded from disk.

### `_validate_schema(raw, config)`

*No documentation available.*

### `load_raw_yaml(config_path)`

Read a YAML configuration file and return its top-level mapping.

### `load_config(config_path)`

Load, validate, and resolve configuration suitable for execution.

### `validate_config(config_path)`

Load a configuration solely for validation feedback.

### `validate_level(cls, value)`

Normalise logging levels and ensure they match the supported set.

### `validate_rate_limits(cls, value)`

Ensure per-resolver rate limits follow the supported syntax.

### `rate_limit_per_second(self)`

Return the configured per-host rate limit in requests per second.

### `parse_service_rate_limit(self, service)`

Return service-specific rate limits expressed as requests per second.

### `normalized_allowed_hosts(self)`

Split allowed host list into exact domains, wildcard suffixes, and associated per-host metadata (ports, IP literals).

### `polite_http_headers(self)`

Compute polite HTTP headers for outbound resolver requests.

### `validate_prefer_source(cls, value)`

Ensure preferred resolvers belong to the supported resolver set.

### `from_defaults(cls)`

Construct a resolved configuration populated with default values only.

## 3. Classes

### `LoggingConfiguration`

Logging-related configuration for ontology downloads.

### `ValidationConfig`

Settings controlling ontology validation throughput and limits.

### `DownloadConfiguration`

HTTP download, retry, and politeness settings for resolvers.

### `DefaultsConfig`

Composite configuration applied when no per-spec overrides exist.

### `ResolvedConfig`

Materialised configuration combining defaults and fetch specifications.

### `EnvironmentOverrides`

Pydantic settings model exposing environment-derived overrides.
