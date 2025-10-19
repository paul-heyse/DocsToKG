# 1. Module: config_loaders

This reference documents the DocsToKG module ``DocsToKG.DocParsing.config_loaders``.

## 1. Overview

Public helpers for loading DocParsing configuration documents.

## 2. Functions

### `load_yaml_markers(raw)`

Deserialize structural marker configuration expressed as YAML.

### `load_toml_markers(raw)`

Deserialize structural marker configuration expressed as TOML.

### `__str__(self)`

Return the stored error message for human-facing output.

## 3. Classes

### `ConfigLoadError`

Raised when configuration documents cannot be deserialized.
