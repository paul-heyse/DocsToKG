# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.DocParsing.__init__``.

## 1. Overview

High-level facade exposing consolidated DocParsing modules.

## 2. Functions

### `_import_module(module_name)`

Wrapper around :func:`import_module` for monkeypatch-friendly indirection.

### `_load_module(name)`

Load a module by name, using cache for performance.

### `__getattr__(name)`

Dynamically import submodules only when they are requested.

### `__dir__()`

Ensure lazily exposed attributes appear in :func:`dir` results.

### `plan()`

Proxy to :func:`DocsToKG.DocParsing.core.plan` with lazy loading.

### `manifest()`

Proxy to :func:`DocsToKG.DocParsing.core.manifest` with lazy loading.

### `pdf_build_parser()`

Proxy to :func:`DocsToKG.DocParsing.doctags.pdf_build_parser`.

### `pdf_parse_args()`

Proxy to :func:`DocsToKG.DocParsing.doctags.pdf_parse_args`.

### `pdf_main()`

Proxy to :func:`DocsToKG.DocParsing.doctags.pdf_main`.

### `html_build_parser()`

Proxy to :func:`DocsToKG.DocParsing.doctags.html_build_parser`.

### `html_parse_args()`

Proxy to :func:`DocsToKG.DocParsing.doctags.html_parse_args`.

### `html_main()`

Proxy to :func:`DocsToKG.DocParsing.doctags.html_main`.
