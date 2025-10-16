# 1. Module: pdf_pipeline

This reference documents the DocsToKG module ``DocsToKG.DocParsing.pdf_pipeline``.

## 1. Overview

Compatibility shims for the legacy ``pdf_pipeline`` module.

The PDF conversion logic now lives in :mod:`DocsToKG.DocParsing.pipelines`.
This module re-exports the historical surface so downstream tooling and tests
can continue importing ``DocsToKG.DocParsing.pdf_pipeline`` without changes.

## 2. Functions

### `parse_args(argv)`

Return parsed CLI arguments for the PDF pipeline.

### `main(args)`

Invoke the refactored PDF pipeline using legacy entrypoints.
