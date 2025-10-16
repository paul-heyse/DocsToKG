# 1. Module: plugins

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.plugins``.

## 1. Overview

Plugin discovery helpers for ontology downloader components.

## 2. Functions

### `load_resolver_plugins(registry)`

Discover resolver plugins registered via ``entry_points``.

### `ensure_resolver_plugins(registry)`

Load resolver plugins exactly once per interpreter.

### `load_validator_plugins(registry)`

Discover validator plugins registered via ``entry_points``.

### `plan(self)`

Plan a fetch for the provided ontology specification.

### `__call__(self)`

Execute the validator.

## 3. Classes

### `ResolverPlugin`

Protocol describing resolver plugins discovered via entry points.

### `ValidatorPlugin`

Protocol describing validator plugins discovered via entry points.
