# 1. Module: markers

This reference documents the DocsToKG module ``DocsToKG.DocParsing.formats.markers``.

## 1. Overview

Utilities for structural markers used by DocParsing chunking.

## 2. Functions

### `dedupe_preserve_order(markers)`

Return markers without duplicates while preserving input order.

### `_ensure_str_list(value, label)`

Normalise configuration entries into string lists.

### `load_structural_marker_config(path)`

Load user-provided heading and caption markers from JSON or YAML.
