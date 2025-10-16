# 1. Module: classifier

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.core``.

## 1. Overview

Payload classification helpers shared across the content download toolkit.

## 2. Functions

### `classify_payload(head_bytes, content_type, url)`

Classify a payload as ``"pdf"`` or ``"html"`` when signals are present.

### `_extract_filename_from_disposition(disposition)`

Return the filename component from a Content-Disposition header.

### `_infer_suffix(url, content_type, disposition, classification, default_suffix)`

Infer a destination suffix from HTTP hints and classification heuristics.
