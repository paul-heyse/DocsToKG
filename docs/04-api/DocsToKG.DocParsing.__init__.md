# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.DocParsing.__init__``.

## 1. Overview

High-level facade exposing consolidated DocParsing modules and compatibility shims.

## 2. Functions

### `_populate_forwarding_module(module, target)`

Populate ``module`` so that it forwards attributes to ``target``.

### `_populate_cli_module(module)`

Populate the legacy ``cli`` module shim.

### `_populate_pdf_pipeline_module(module)`

Populate the legacy ``pdf_pipeline`` module shim with a deprecation warning.

### `__getattr__(attr)`

*No documentation available.*

### `parse_args(argv)`

Legacy CLI argument parser for backwards-compatible imports.

### `main(args)`

Legacy entry point delegating to :mod:`DocsToKG.DocParsing.doctags`.

### `create_module(self, spec)`

Create a new module instance that will be populated by the shim.

### `exec_module(self, module)`

Execute the shim builder to populate ``module``.

### `find_spec(self, fullname, path, target)`

Return a module spec when ``fullname`` matches a supported shim.

## 3. Classes

### `_DocParsingShimLoader`

Loader that populates deprecated DocParsing modules on demand.

### `_DocParsingShimFinder`

Meta-path finder that serves the compatibility shims defined above.
