## Why

DocsToKG needs to ingest and process ontologies from diverse sources (OBO Library, OLS, BioPortal, SKOS registries, XBRL taxonomies) for knowledge graph construction. Currently, there is no systematic way to discover, download, validate, and normalize ontologies from these heterogeneous sources. Manual downloads are fragile, lack provenance tracking, and don't handle format conversions or caching, creating bottlenecks for downstream processing pipelines.

## What Changes

- Create a source-agnostic ontology download manager that resolves ontology identifiers (prefixes, acronyms, URLs) to downloadable artifacts using battle-tested Python libraries (Bioregistry, OAK, ols-client, ontoportal-client).
- Implement a resolver registry supporting OBO Library PURLs (via Bioregistry), OLS4 (via ols-client), BioPortal/OntoPortal (via ontoportal-client), direct SKOS/RDF URLs (via RDFLib), and XBRL taxonomies (via Arelle).
- Add robust download plumbing with ETag/Last-Modified conditional requests, SHA-256 verification, partial resume via Range headers, exponential backoff retry, and per-host rate limiting.
- Provide post-download validation and normalization using RDFLib (RDF/OWL parsing), Pronto (OBO/OWL validation), Owlready2 (OWL reasoning), optional ROBOT (conversions/QC), and Arelle (XBRL validation).
- Store original artifacts alongside normalized formats (Turtle, OBO Graph JSON) in versioned directories under pystow-managed paths (~/.data/ontology-fetcher) with comprehensive provenance manifests (URL, version, license, ETag, SHA-256, timestamps).
- Expose configuration via YAML (sources.yaml) for ontology specs, license allowlists, format preferences, resolver priorities, and HTTP parameters.
- Implement CLI (`ontofetch`) for pull/show/validate operations with structured logging (JSON) capturing status, retries, latencies, and validation outcomes.
- Cover resolver stack with unit tests (mocked resolvers), download tests (local HTTP server with resume/ETag support), parser tests (fixture ontologies), and integration smoke tests (nightly fetches of public ontologies).

## Impact

- Affected specs: new `ontology-downloader` capability
- Affected code: new modules under `src/DocsToKG/OntologyDownload/` (resolvers, download manager, validators, CLI)
- External dependencies: Bioregistry, oaklib, ols-client, ontoportal-client, RDFLib, Pronto, Owlready2, Arelle, pystow, pooch (all available via pip)
- Storage: pystow-managed directory at ~/.data/ontology-fetcher with cache/, ontologies/, logs/, configs/
- External services: OBO Library, OLS4, BioPortal/OntoPortal APIs, direct HTTP endpoints for SKOS/XBRL
