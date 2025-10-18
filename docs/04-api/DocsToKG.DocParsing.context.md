# 1. Module: context

This reference documents the DocsToKG module ``DocsToKG.DocParsing.context``.

## 1. Overview

Typed runtime context shared across DocParsing stages.

The :class:`ParsingContext` dataclass captures the run-scoped attributes that
the CLI and pipeline stages exchange. It replaces loosely typed dictionaries so
callers benefit from IDE completion, static analysis, and centralised default
management. The context can also serialise itself into manifest-friendly
payloads when stages record configuration snapshots.

## 2. Functions

### `__post_init__(self)`

Normalise paths eagerly to simplify downstream usage.

### `_resolve_path(value)`

Return ``value`` coerced to an absolute :class:`Path`.

### `field_names(cls)`

Expose recognised field names (excluding the ``extra`` payload).

### `apply_config(self, cfg)`

Populate context attributes from a stage configuration dataclass.

### `merge_extra(self, mapping)`

Merge arbitrary manifest-safe metadata into the context.

### `update_extra(self)`

Convenience helper mirroring :meth:`dict.update` with filtering.

### `to_manifest(self)`

Serialise the context to a manifest-friendly dictionary.

### `copy(self)`

Return a shallow copy suitable for isolated mutation.

## 3. Classes

### `ParsingContext`

Runtime metadata describing a DocParsing invocation.
