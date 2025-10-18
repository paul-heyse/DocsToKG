# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.DocParsing``.

## 1. Overview

High-level facade exposing consolidated DocParsing modules. The package now
provides direct imports for the canonical implementations without dynamically
registered compatibility shims.

## 2. Re-exported Modules

- ``core`` – unified helpers for DocTags conversion, chunking, embedding, and CLI tooling.
- ``formats`` – Pydantic models and validation helpers for chunk/vector payloads.
- ``doctags`` – PDF/HTML DocTags conversion pipeline and CLI parser builders.
- ``chunking`` – Runtime utilities for Docling chunk generation.
- ``embedding`` – Embedding runtime, configuration, and pooling helpers.
- ``token_profiles`` – Token profile analysis for DocTags corpora.

## 3. Convenience Functions

The package surfaces frequently used entry points from :mod:`DocsToKG.DocParsing.doctags`:

- ``pdf_build_parser`` / ``html_build_parser`` – construct CLI parsers.
- ``pdf_parse_args`` / ``html_parse_args`` – parse CLI arguments for DocTags conversion.
- ``pdf_main`` / ``html_main`` – run DocTags conversion in PDF or HTML mode.

The ``plan`` and ``manifest`` helpers from :mod:`DocsToKG.DocParsing.core` remain
available as top-level attributes for convenience.
