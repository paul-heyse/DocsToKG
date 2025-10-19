# 1. Module: resolvers

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.resolvers``.

## 1. Overview

Resolver implementations for DocsToKG ontology downloads.

## 2. Functions

### `normalize_license_to_spdx(value)`

Normalize common license strings to canonical SPDX identifiers.

### `plan(self, spec, config, logger)`

Produce a :class:`FetchPlan` describing how to retrieve ``spec``.

Args:
spec: The fetch specification to plan for.
config: The resolved configuration.
logger: Logger for this operation.
cancellation_token: Optional token for cooperative cancellation.

Returns:
A fetch plan describing how to retrieve the specification.

### `_normalize_media_type(self, media_type)`

*No documentation available.*

### `_preferred_media_type(self, spec)`

*No documentation available.*

### `_negotiate_media_type(self)`

*No documentation available.*

### `_execute_with_retry(self, func)`

Run a callable with retry semantics tailored for resolver APIs.

### `_extract_correlation_id(self, logger)`

Return the correlation id from a logger adapter when available.

### `_build_polite_headers(self, config, logger)`

Create polite headers derived from configuration and logger context.

### `_apply_headers_to_session(session, headers)`

Apply polite headers to a client session when supported.

### `_request_with_retry(self)`

Issue an HTTP request with polite headers and retry semantics.

### `_build_plan(self)`

Construct a ``FetchPlan`` from resolver components.

### `plan(self, spec, config, logger)`

Build a :class:`FetchPlan` by resolving OBO-hosted download URLs.

### `plan(self, spec, config, logger)`

Plan an OLS download by negotiating media type and authentication.

### `_load_api_key(self)`

*No documentation available.*

### `plan(self, spec, config, logger)`

Plan a BioPortal download by combining ontology and submission metadata.

### `_iter_dicts(payload)`

*No documentation available.*

### `plan(self, spec, config, logger)`

Plan a LOV download by inspecting linked metadata for URLs and licenses.

### `plan(self, spec, config, logger)`

Plan a SKOS vocabulary download from explicit configuration metadata.

### `plan(self, spec, config, logger)`

Plan a direct download from declarative extras without discovery.

### `plan(self, spec, config, logger)`

Plan an XBRL taxonomy download using declarative resolver extras.

### `plan(self, spec, config, logger)`

Plan an Ontobee download by composing the appropriate REST endpoint.

### `_retryable(exc)`

*No documentation available.*

### `_on_retry(attempt, exc, sleep_time)`

*No documentation available.*

### `_invoke()`

*No documentation available.*

### `_perform()`

*No documentation available.*

## 3. Classes

### `FetchPlan`

Concrete plan output from a resolver.

### `Resolver`

Protocol describing resolver planning behaviour.

### `ResolverCandidate`

Resolver plan captured for download-time fallback.

### `BaseResolver`

Shared helpers for resolver implementations.

### `OBOResolver`

Resolve ontologies hosted on the OBO Library using Bioregistry helpers.

### `OLSResolver`

Resolve ontologies from the Ontology Lookup Service (OLS4).

### `BioPortalResolver`

Resolve ontologies using the BioPortal (OntoPortal) API.

### `LOVResolver`

Resolve vocabularies from Linked Open Vocabularies (LOV).

### `SKOSResolver`

Resolve SKOS vocabularies specified directly via configuration.

### `DirectResolver`

Resolve direct download links specified in configuration extras.

### `XBRLResolver`

Resolve XBRL taxonomy downloads from regulator endpoints.

### `OntobeeResolver`

Resolve Ontobee-hosted ontologies via canonical PURLs.
