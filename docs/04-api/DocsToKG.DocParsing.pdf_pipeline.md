# 1. Module: pdf_pipeline

This reference documents the DocsToKG module ``DocsToKG.DocParsing.pdf_pipeline``.

## 1. Overview

Compatibility shims for the legacy ``pdf_pipeline`` module.

The PDF conversion logic now lives in :mod:`DocsToKG.DocParsing.pipelines`.
This module re-exports the historical surface so downstream tooling and tests
can continue importing ``DocsToKG.DocParsing.pdf_pipeline`` without changes.

Importing this module emits a :class:`DeprecationWarning`; migrate to
:mod:`DocsToKG.DocParsing.pipelines` (or the CLI entry points) before the shim is
removed in a future release.

## 2. Functions

### `parse_args(argv)`

Return parsed CLI arguments for the legacy PDF pipeline.

Args:
argv: Optional argument vector forwarded to :func:`argparse.ArgumentParser.parse_args`.

Returns:
argparse.Namespace: Parsed argument namespace compatible with the refactored pipeline.

### `main(args)`

Invoke the refactored PDF pipeline using the legacy facade.

Args:
args: Optional argument namespace or raw argument list compatible with :func:`parse_args`.

Returns:
int: Process exit code returned by :func:`DocsToKG.DocParsing.pipelines.pdf_main`.
