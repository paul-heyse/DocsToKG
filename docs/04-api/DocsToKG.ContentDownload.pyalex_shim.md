# 1. Module: pyalex_shim

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.pyalex_shim``.

## 1. Overview

Helpers for interacting with the pyalex configuration safely.

Tests swap in fake ``pyalex`` modules (and configs) to avoid hitting the real
network client. Import-time aliases to ``pyalex.config`` become stale when that
swap happens later, so this module provides a small proxy and helper utilities
that always resolve the active configuration object at call time.

## 2. Functions

### `_load_pyalex_module()`

Return the currently registered ``pyalex`` module if available.

### `get_config()`

Return the live ``pyalex`` config object, handling late module swaps.

### `_set_config_attr(config, name, value)`

Assign an attribute on the config, falling back to mapping semantics.

### `apply_mailto(mailto)`

Update the pyalex config contact fields when a mailto is provided.

### `__getattr__(self, name)`

*No documentation available.*

### `__setattr__(self, name, value)`

*No documentation available.*

## 3. Classes

### `ConfigProxy`

Dynamic proxy that forwards attribute access to the live pyalex config.
