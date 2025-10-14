## 1. Setup and Dependencies
- [ ] 1.1 Add dependencies to requirements.txt or pyproject.toml: bioregistry, oaklib, ols-client, ontoportal-client, rdflib, pronto, owlready2, arelle, pystow, pooch with version pins
- [ ] 1.2 Create module structure: src/DocsToKG/OntologyDownload/ with __init__.py, core.py, resolvers.py, download.py, validators.py, cli.py
- [ ] 1.3 Set up pystow configuration for ~/.data/ontology-fetcher with subdirectories: configs/, cache/, logs/, ontologies/
- [ ] 1.4 Create example sources.yaml configuration file with defaults section and sample ontologies (HP, EFO, NCIT, EuroVoc)

## 2. Core Data Models
- [ ] 2.1 Implement FetchSpec dataclass with fields: id, resolver, extras, target_formats
- [ ] 2.2 Implement FetchPlan dataclass with fields: url, headers, filename_hint, version, license, media_type
- [ ] 2.3 Implement FetchResult dataclass with fields: local_path, status, etag, last_modified, sha256
- [ ] 2.4 Implement Manifest dataclass for provenance with fields: id, resolver, url, filename, version, license, status, sha256, etag, last_modified, downloaded_at, target_formats
- [ ] 2.5 Define Resolver protocol interface with plan() method signature

## 3. Resolver Implementations
- [ ] 3.1 Implement OBOResolver using Bioregistry (get_owl_download, get_obo_download, get_rdf_download) with format fallback logic
- [ ] 3.2 Implement OLSResolver using ols-client to query OLS4 API and retrieve ontology metadata/download URLs
- [ ] 3.3 Implement BioPortalResolver using ontoportal-client to fetch latest submission URLs with API key from pystow
- [ ] 3.4 Implement SKOSResolver for direct URL-based SKOS thesauri
- [ ] 3.5 Implement XBRLResolver for XBRL taxonomy ZIP packages
- [ ] 3.6 Create resolver registry dictionary mapping resolver names to instances
- [ ] 3.7 Add unit tests for each resolver with mocked library responses

## 4. Download Manager
- [ ] 4.1 Implement sha256_file() utility function for computing file hashes
- [ ] 4.2 Implement download_stream() using pooch with stream-to-.part pattern
- [ ] 4.3 Add ETag/Last-Modified conditional request headers from previous manifest
- [ ] 4.4 Implement HTTP 304 cache hit detection and early return
- [ ] 4.5 Add Range request support for partial resume on interruption
- [ ] 4.6 Implement exponential backoff retry with configurable max_retries (default 5) and backoff_factor (default 0.5)
- [ ] 4.7 Add per-host rate limiting using token bucket algorithm (default 4 req/sec)
- [ ] 4.8 Add SHA-256 verification on completed downloads
- [ ] 4.9 Add structured logging for HTTP status, elapsed time, retry count, cache hit/miss
- [ ] 4.10 Create test fixture: local HTTP server supporting ETag, Last-Modified, Range headers for download tests

## 5. Validation Pipeline
- [ ] 5.1 Implement RDFLib validator: Graph.parse() with triple count recording and error capture
- [ ] 5.2 Add RDFLib Turtle normalization (serialize to normalized/)
- [ ] 5.3 Implement Pronto validator: Ontology() load with term count recording and error capture
- [ ] 5.4 Add Pronto OBO Graph JSON export (to normalized/)
- [ ] 5.5 Implement Owlready2 validator: get_ontology().load() with error capture
- [ ] 5.6 Add optional ROBOT integration: runtime detection via shutil.which('robot')
- [ ] 5.7 Implement ROBOT convert subprocess call for format conversions
- [ ] 5.8 Implement ROBOT report subprocess call for SPARQL QC checks
- [ ] 5.9 Implement Arelle validator for XBRL taxonomy packages with JSON result export
- [ ] 5.10 Write validation results to validation/<parser>_parse.json
- [ ] 5.11 Create mini-ontology test fixtures (5-10 terms) in Turtle, OBO, OWL for parser tests

## 6. Storage and Manifests
- [ ] 6.1 Implement directory creation for ontologies/<id>/<version>/{original,normalized,validation}
- [ ] 6.2 Implement manifest.json write with comprehensive provenance fields
- [ ] 6.3 Implement manifest.json read for cache invalidation decisions (ETag/Last-Modified)
- [ ] 6.4 Add version resolution logic (use provided version or generate from timestamp)
- [ ] 6.5 Implement safe ZIP extraction with path validation for XBRL taxonomies
- [ ] 6.6 Add manifest validation and schema checks

## 7. Configuration
- [ ] 7.1 Implement YAML parser for sources.yaml with schema validation
- [ ] 7.2 Parse defaults section: accept_licenses, normalize_to, prefer_source, http settings
- [ ] 7.3 Parse ontologies list into FetchSpec objects with per-ontology overrides
- [ ] 7.4 Implement license allowlist checking with fail-closed behavior
- [ ] 7.5 Add configuration validation command (ontofetch config validate)
- [ ] 7.6 Document configuration schema with examples

## 8. Orchestration
- [ ] 8.1 Implement fetch_one() function orchestrating resolver plan → download → validation → manifest
- [ ] 8.2 Add error handling and structured logging at each pipeline stage
- [ ] 8.3 Implement batch fetch_all() for multiple ontologies from sources.yaml
- [ ] 8.4 Add progress reporting for batch operations
- [ ] 8.5 Implement graceful failure recording (continue on error, log failures)

## 9. CLI
- [ ] 9.1 Set up CLI framework (argparse or Click) with ontofetch entry point
- [ ] 9.2 Implement `ontofetch pull` subcommand with --spec FILE and positional ID arguments
- [ ] 9.3 Add --force flag to bypass cache and force redownload
- [ ] 9.4 Implement `ontofetch show ID` subcommand to display manifest
- [ ] 9.5 Add --versions flag to list all downloaded versions
- [ ] 9.6 Implement `ontofetch validate ID[@VERSION]` subcommand to re-run validators
- [ ] 9.7 Add validator selection flags: --robot, --rdflib, --pronto, --owlready2, --arelle
- [ ] 9.8 Add --json flag for machine-readable output across all subcommands
- [ ] 9.9 Implement human-friendly formatted output (tables, summaries)
- [ ] 9.10 Configure structured JSON logging with custom formatter

## 10. Testing
- [ ] 10.1 Write resolver unit tests with mocked library responses (one test suite per resolver)
- [ ] 10.2 Write download manager tests using local HTTP server fixture
- [ ] 10.3 Test ETag/Last-Modified conditional requests (304 response)
- [ ] 10.4 Test Range request resume on partial download
- [ ] 10.5 Test exponential backoff retry on transient failures (500, 503)
- [ ] 10.6 Test per-host rate limiting enforcement
- [ ] 10.7 Write parser tests using mini-ontology fixtures (RDFLib, Pronto, Owlready2)
- [ ] 10.8 Test ROBOT integration with optional runtime detection
- [ ] 10.9 Test Arelle XBRL validation with minimal taxonomy package
- [ ] 10.10 Write integration smoke test: fetch PATO (OBO), BFO (OLS), verify manifests and validation outputs
- [ ] 10.11 Add configuration validation tests (schema checks, invalid YAML)
- [ ] 10.12 Test safe ZIP extraction with malicious paths (zip-slip guard)

## 11. Documentation
- [ ] 11.1 Write installation instructions with dependency installation command
- [ ] 11.2 Document pystow configuration: env vars, credential paths for BioPortal API keys
- [ ] 11.3 Create sources.yaml schema documentation with field descriptions
- [ ] 11.4 Write CLI usage guide with examples for pull/show/validate subcommands
- [ ] 11.5 Document resolver-specific requirements (BioPortal API key, ROBOT Java runtime)
- [ ] 11.6 Add troubleshooting guide for common issues (rate limits, auth failures, validation errors)
- [ ] 11.7 Document storage layout and manifest schema
- [ ] 11.8 Add example workflows: batch download, incremental update, validation-only re-runs

## 12. Integration and Polish
- [ ] 12.1 Run integration smoke tests with real API calls (HP from OBO, EFO from OLS)
- [ ] 12.2 Test end-to-end workflow with example sources.yaml
- [ ] 12.3 Verify structured logs contain all required fields (resolver, status, timing, errors)
- [ ] 12.4 Test force-refresh with --force flag on cached ontologies
- [ ] 12.5 Verify SHA-256 checksums recorded correctly in manifests
- [ ] 12.6 Test multi-version handling (download same ontology multiple times, check version directories)
- [ ] 12.7 Verify normalized/ directory contains Turtle and OBO Graph JSON where applicable
- [ ] 12.8 Run validation on large ontology (NCIT, ChEBI) to check memory usage
- [ ] 12.9 Test CLI --json output is valid JSON and includes expected fields
- [ ] 12.10 Add performance benchmarks for download/validation on representative ontologies

