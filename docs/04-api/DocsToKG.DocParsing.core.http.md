# 1. Module: http

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.http``.

## 1. Overview

HTTP client helpers for DocParsing.

## 2. Functions

### `normalize_http_timeout(timeout)`

Normalize timeout inputs into a ``(connect, read)`` tuple of floats.

### `get_http_session()`

Return a shared :class:`requests.Session` configured with retries.

### `_clone_http_session(session)`

Create a shallow copy of a :class:`requests.Session` for transient headers.

### `_coerce_pair(values)`

Coerce arbitrary iterables into a two-element timeout tuple.
