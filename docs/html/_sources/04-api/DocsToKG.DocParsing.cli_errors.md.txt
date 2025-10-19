# 1. Module: cli_errors

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli_errors``.

## 1. Overview

Shared CLI validation error types for DocParsing entry points.

## 2. Functions

### `format_cli_error(error)`

Return a consistent error string for CLI consumption.

### `__post_init__(self)`

Initialise the ``ValueError`` base with the human-readable message.

### `__str__(self)`

Return the underlying message for convenience.

### `__post_init__(self)`

Ensure the chunk stage marker is applied before chaining.

### `__post_init__(self)`

Ensure the doctags stage marker is applied before chaining.

### `__post_init__(self)`

Ensure the embed stage marker is applied before chaining.

## 3. Classes

### `CLIValidationError`

Base exception capturing option names and human-friendly messages.

### `ChunkingCLIValidationError`

Validation error raised by chunking CLI helpers.

### `DoctagsCLIValidationError`

Validation error raised by DocTags CLI helpers.

### `EmbeddingCLIValidationError`

Validation error raised by embedding CLI helpers.
