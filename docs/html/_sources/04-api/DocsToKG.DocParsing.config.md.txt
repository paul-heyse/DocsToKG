# 1. Module: config

This reference documents the DocsToKG module ``DocsToKG.DocParsing.config``.

## 1. Overview

Configuration helpers for DocParsing stages.

This module isolates the machinery required to hydrate stage configuration
objects from environment variables, CLI arguments, and on-disk configuration
files. Keeping these helpers separate from the CLI entry points makes unit
testing less cumbersome and prevents import-time side effects.

## 2. Functions

### `_load_yaml_markers(raw)`

Deserialize YAML configuration content, raising for missing dependencies.

### `_load_toml_markers(raw)`

Deserialize TOML configuration content with compatibility fallbacks.

### `load_config_mapping(path)`

Load a configuration mapping from JSON, YAML, or TOML.

### `_manifest_value(value)`

Convert values to manifest-friendly representations.

### `_coerce_path(value, base_dir)`

Convert ``value`` into an absolute :class:`Path`.

### `_coerce_optional_path(value, base_dir)`

Convert optional path-like values.

### `_coerce_bool(value, _base_dir)`

Convert supported string and numeric representations into booleans.

Raises:
ValueError: If ``value`` cannot be interpreted as a boolean literal.

### `_coerce_int(value, _base_dir)`

Convert ``value`` to ``int``.

### `_coerce_float(value, _base_dir)`

Convert ``value`` to ``float``.

### `_coerce_str(value, _base_dir)`

Return ``value`` coerced to string.

### `_coerce_str_tuple(value, _base_dir)`

Return ``value`` as a tuple of strings.

### `_namespace_setdefault(namespace, name, default)`

Return ``namespace.name`` or set it to ``default`` when absent.

### `annotate_cli_overrides(namespace)`

Attach CLI override metadata to ``namespace`` and return the explicit set.

### `parse_args_with_overrides(parser, argv)`

Parse CLI arguments while tracking which options were explicitly provided.

### `ensure_cli_metadata(namespace)`

Ensure ``namespace`` carries CLI metadata, defaulting to treating all fields as explicit.

### `apply_env(self)`

Overlay configuration from environment variables.

### `update_from_file(self, cfg_path)`

Overlay configuration from ``cfg_path``.

### `apply_args(self, args)`

Overlay configuration from an argparse namespace.

### `from_env(cls)`

Instantiate a configuration populated solely from environment variables.

### `finalize(self)`

Hook allowing subclasses to normalise derived fields.

### `to_manifest(self)`

Return a manifest-friendly snapshot of the configuration.

### `_coerce_field(self, name, value, base_dir)`

Run field-specific coercion logic before manifest serialization.

### `is_overridden(self, field_name)`

Return ``True`` when ``field_name`` was explicitly overridden.

## 3. Classes

### `StageConfigBase`

Base dataclass for stage configuration objects.
