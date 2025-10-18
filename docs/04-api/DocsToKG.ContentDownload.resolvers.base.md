# 1. Module: base

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.base``.

## 1. Overview

Shared resolver primitives and helpers for the content download pipeline.

## 2. Functions

### `_absolute_url(base, href)`

Resolve relative ``href`` values against ``base`` to obtain absolute URLs.

### `_collect_candidate_urls(node, results)`

Recursively collect HTTP(S) URLs from nested response payloads.

### `find_pdf_via_meta(soup, base_url)`

Return PDF URL declared via ``citation_pdf_url`` meta tags.

### `find_pdf_via_link(soup, base_url)`

Return PDF URL advertised through alternate link tags.

### `find_pdf_via_anchor(soup, base_url)`

Return PDF URL inferred from anchor elements mentioning PDFs.

### `_fetch_semantic_scholar_data(session, config, doi)`

Return Semantic Scholar metadata for ``doi`` using configured headers.

### `_fetch_unpaywall_data(session, config, doi)`

Return Unpaywall metadata for ``doi`` using configured headers.

### `__post_init__(self)`

*No documentation available.*

### `is_event(self)`

Return ``True`` when this result represents an informational event.

### `from_wire(cls, value)`

Coerce serialized event values into :class:`ResolverEvent` members.

### `from_wire(cls, value)`

Coerce serialized reason values into :class:`ResolverEventReason` members.

### `is_enabled(self, config, artifact)`

Return ``True`` if this resolver should run for the given artifact.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs or events for the given artifact.

### `register(cls, resolver_cls)`

Register a resolver class under its declared ``name`` attribute.

### `create_default(cls)`

Instantiate resolver instances in priority order.

### `__init_subclass__(cls, register)`

*No documentation available.*

### `_request_json(self, session, method, url)`

*No documentation available.*

## 3. Classes

### `ResolverResult`

Either a candidate download URL or an informational resolver event.

### `ResolverEvent`

Structured event taxonomy emitted by resolvers.

### `ResolverEventReason`

Structured reason taxonomy for resolver events.

### `Resolver`

Protocol that resolver implementations must follow.

### `ResolverRegistry`

Registry tracking resolver classes by their ``name`` attribute.

### `RegisteredResolver`

Mixin ensuring subclasses register with :class:`ResolverRegistry`.

### `ApiResolverBase`

Shared helper for resolvers interacting with JSON-based HTTP APIs.
