# 1. Module: plugins

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.plugins``.

## 1. Overview

Plugin discovery helpers for ontology downloader components.

## 2. Functions

### `_describe_plugin(obj)`

*No documentation available.*

### `_detect_entry_version(entry)`

*No documentation available.*

### `_load_resolver_plugins_locked(registry)`

Populate ``registry`` with resolver plugins discovered via entry points.

### `load_resolver_plugins(registry)`

Discover resolver plugins registered via ``entry_points``.

### `ensure_resolver_plugins(registry)`

Load resolver plugins exactly once per interpreter.

### `_load_validator_plugins_locked(registry)`

Populate ``registry`` with validator plugins discovered via entry points.

### `load_validator_plugins(registry)`

Discover validator plugins registered via ``entry_points``.

### `ensure_plugins_loaded()`

Load resolver and validator plugins exactly once in a thread-safe manner.

### `get_resolver_registry()`

Return the resolver plugin registry, ensuring plugins are loaded once.

### `get_validator_registry()`

Return the validator plugin registry, ensuring plugins are loaded once.

### `register_plugin_registry(kind, registry)`

Register the in-memory plugin registry for discovery helpers.

### `get_plugin_registry(kind)`

Return the registered plugin registry for ``kind``.

### `list_registered_plugins(kind)`

Return mapping of plugin names to qualified identifiers for ``kind``.

### `get_registered_plugin_meta(kind)`

Return metadata captured for entry-point registered plugins.

### `plan(self)`

Plan a fetch for the provided ontology specification.

### `__call__(self)`

Execute the validator.

## 3. Classes

### `ResolverPlugin`

Protocol describing resolver plugins discovered via entry points.

### `ValidatorPlugin`

Protocol describing validator plugins discovered via entry points.
