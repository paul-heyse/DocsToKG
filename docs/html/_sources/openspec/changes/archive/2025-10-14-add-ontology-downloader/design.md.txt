## Context

DocsToKG processes scientific documents and builds knowledge graphs from their content. Ontologies provide controlled vocabularies, taxonomies, and semantic relationships essential for entity recognition, concept linking, and knowledge graph quality. The project requires access to biomedical ontologies (GO, HP, UBERON, EFO, NCIT), legal/financial thesauri (EuroVoc, XBRL taxonomies), and domain-specific vocabularies from multiple sources (OBO Library, OLS, BioPortal, SKOS registries). Current workflows rely on manual downloads or ad-hoc scripts that lack caching, provenance, validation, and error recovery.

## Goals / Non-Goals

- Goals:
  - Automate discovery and download of ontologies from OBO Library, OLS, BioPortal, SKOS endpoints, and XBRL registries
  - Provide robust HTTP handling with conditional requests (ETag/Last-Modified), resume support, checksums, retries, and rate limiting
  - Validate downloaded artifacts using domain-appropriate parsers (RDFLib, Pronto, Owlready2, Arelle)
  - Normalize ontologies to consistent formats (Turtle, OBO Graph JSON) while preserving originals
  - Record comprehensive provenance (source URL, version, license, checksums, timestamps) in manifest files
  - Expose declarative configuration (YAML) for ontology lists, resolver preferences, and HTTP policies
  - Support idempotent operations (cached downloads, incremental updates)
  - Enable observability through structured logs and validation reports
- Non-Goals:
  - Building ontology editors or reasoners (use existing tools: Protégé, HermiT, ELK)
  - Implementing custom RDF stores or SPARQL endpoints (downstream systems handle storage)
  - Supporting proprietary ontologies with restricted access beyond credential configuration
  - Real-time ontology update notifications (polling-based updates suffice)
  - GUI for ontology browsing (CLI and programmatic API only)

## Decisions

- Decision: **Use battle-tested Python libraries instead of custom implementations** for discovery (Bioregistry, OAK), access (ols-client, ontoportal-client), parsing (RDFLib, Pronto, Owlready2), and domain-specific validation (Arelle for XBRL, optional ROBOT for conversions).
  Alternatives: custom implementations would duplicate well-maintained community tools and increase maintenance burden. Rationale: leverages expertise and testing from OBO, EMBL-EBI, NCBO, and RDF communities.

- Decision: **Adopt pystow for standardized data directories and secrets management**, aligning with OAK and ontoportal-client conventions.
  Alternatives: custom directory structures would diverge from ecosystem conventions. Rationale: users get predictable paths (~/.data/ontology-fetcher), env-var overrides, and unified configuration with related tools.

- Decision: **Use pooch (not cached_path) for download plumbing** with hash verification, caching, and registry support.
  Alternatives: custom HTTP logic or cached_path. Rationale: pooch provides built-in SHA-256 verification, registry patterns, and cleaner API for our use case; cached_path is simpler but lacks hash verification features.

- Decision: **Implement resolver registry pattern** with protocol-based interfaces (`Resolver.plan(spec) → FetchPlan`) allowing independent resolvers for OBO, OLS, BioPortal, SKOS, and XBRL.
  Alternatives: monolithic resolver with conditional logic would be harder to test and extend. Rationale: enables per-source unit testing, clean addition of new sources, and isolated credential handling.

- Decision: **Support format preferences and fallback chains** (e.g., prefer OWL → OBO → RDF) per ontology, with first-available-wins semantics.
  Alternatives: hardcode single format per source. Rationale: some ontologies publish multiple serializations with varying quality; flexibility improves success rates.

- Decision: **Store original artifacts and normalized versions separately** in `original/` and `normalized/` subdirectories with validation reports in `validation/`.
  Alternatives: only normalized versions or single mixed directory. Rationale: preserves bit-exact originals for reproducibility while providing tooling-friendly normalized formats (Turtle) for downstream pipelines.

- Decision: **Record comprehensive provenance in manifest.json** including URL, resolver, version, license, ETag, Last-Modified, SHA-256, timestamps, and validation status.
  Alternatives: minimal metadata or external database. Rationale: self-contained manifests enable auditing, cache invalidation decisions, and offline reproducibility without external dependencies.

- Decision: **Support ETag/Last-Modified conditional requests and HTTP Range resume** for idempotent operations and efficient incremental updates.
  Alternatives: always redownload. Rationale: reduces bandwidth, respects rate limits, and enables graceful recovery from network interruptions.

- Decision: **Implement exponential backoff retry (5 attempts, 0.5s base) and per-host rate limiting (4 req/sec default)** to avoid overwhelming source services.
  Alternatives: no retry or unlimited rate. Rationale: improves reliability against transient failures while being a respectful HTTP client (follows robots.txt spirit).

- Decision: **Use ols-client (not ebi-ols-client) for OLS4 access** as specified by user requirements.
  Alternatives: ebi-ols-client or direct REST calls. Rationale: user preference stated explicitly; ols-client is actively maintained.

- Decision: **Expose CLI via `ontofetch` command** with subcommands (pull, show, validate) for both single-ontology and batch operations.
  Alternatives: Python API only or separate scripts. Rationale: CLI provides immediate usability for operators while Python API supports programmatic integration.

- Decision: **Configure ontology lists and policies via sources.yaml** with defaults section for global settings and per-ontology overrides.
  Alternatives: JSON, TOML, or code-based configuration. Rationale: YAML is human-friendly for lists, supports comments, and aligns with common DevOps practices.

- Decision: **Validate using multiple parsers per format** (RDFLib + Pronto for OWL/OBO) to catch format-specific issues and gather richer metadata.
  Alternatives: single parser per format. Rationale: different parsers catch different issues (RDFLib for RDF syntax, Pronto for OBO semantics, Owlready2 for reasoning); redundancy improves quality assurance.

- Decision: **Make ROBOT integration optional** (subprocess-based, only when installed) for conversions and QC reports.
  Alternatives: mandatory ROBOT or skip entirely. Rationale: ROBOT requires Java runtime and separate install; optional support maximizes compatibility while enabling power users to leverage OBO community's QC tooling.

- Decision: **Treat XBRL and RF2 (SNOMED) as native formats** without forced conversion, using domain-specific tools (Arelle for XBRL, external Snowstorm/FHIR for RF2).
  Alternatives: force RDF conversion. Rationale: these formats have specialized semantics and tooling; premature conversion loses fidelity.

- Decision: **Implement comprehensive error handling with specific exception types** (OntologyDownloadError, ResolverError, ValidationError, ConfigurationError) and clear remediation messages.
  Alternatives: generic exceptions or error codes. Rationale: specific exceptions enable targeted catch blocks; remediation messages reduce support burden and enable self-service troubleshooting.

- Decision: **Enforce HTTPS-only downloads with URL validation to prevent SSRF attacks**, rejecting private IP ranges and validating schemes.
  Alternatives: allow all URLs, rely on external firewalls. Rationale: defense-in-depth security; prevents accidental or malicious downloads from internal services; aligns with security best practices.

- Decision: **Mask sensitive data (API keys, tokens) in all logs** using a centralized formatter that redacts before writing.
  Alternatives: trust developers to avoid logging secrets. Rationale: prevents credential leaks in log aggregators, bug reports, and stdout; complies with security auditing requirements.

- Decision: **Implement configurable performance limits** (timeouts, memory constraints, concurrent downloads, max file size) with sensible defaults.
  Alternatives: hardcode limits or unlimited. Rationale: different deployment environments have different constraints; configurability enables tuning without code changes; defaults prevent resource exhaustion.

- Decision: **Use Python 3.9+ as minimum version** for modern type hints (PEP 604 union syntax `str | None`), dict union operators, and improved asyncio.
  Alternatives: Python 3.7+ for broader compatibility. Rationale: DocsToKG codebase likely already uses 3.9+; modern syntax improves code readability; 3.7 reached EOL in June 2023.

- Decision: **Support both Unix signal-based and Windows thread-based timeouts** for parser operations to handle platform differences.
  Alternatives: Unix-only or no timeouts. Rationale: cross-platform support essential for development/deployment flexibility; timeouts prevent hung processes on malformed ontologies.

- Decision: **Implement structured JSON logging with correlation IDs** for batch operations to enable tracing and log aggregation.
  Alternatives: unstructured text logs. Rationale: structured logs parse easily in ELK/Splunk/CloudWatch; correlation IDs connect related operations across log entries; essential for debugging batch failures.

- Decision: **Provide comprehensive CLI help with examples and init subcommand** that generates annotated configuration templates.
  Alternatives: minimal help text, refer to external docs. Rationale: inline help reduces onboarding friction; examples show correct usage patterns; init command accelerates setup.

- Decision: **Implement log rotation with compression and retention policies** to prevent disk exhaustion from long-running deployments.
  Alternatives: unbounded log growth or external logrotate. Rationale: self-contained solution works across platforms; compression saves disk; retention policy prevents indefinite growth.

## Architecture Overview

- **Configuration Layer**: YAML parser loads `sources.yaml` into structured config objects (FetchSpec list, defaults) with validation of required fields and allowlist checks.

- **Resolver Registry**: Protocol-based registry maps resolver names (`obo`, `ols`, `bioportal`, `skos`, `xbrl`) to concrete implementations:
  - **OBOResolver**: uses Bioregistry functions (`get_owl_download`, `get_obo_download`, `get_rdf_download`) to resolve OBO Foundry prefixes to PURLs; falls back through format preferences.
  - **OLSResolver**: queries OLS4 API via ols-client to find ontology by ID/acronym and retrieve canonical artifact URL with version metadata.
  - **BioPortalResolver**: uses ontoportal-client to fetch ontology metadata and latest submission download URL; includes API key from pystow config in headers.
  - **SKOSResolver**: returns configured direct URL for SKOS thesauri (EuroVoc, ECB vocabularies); validation deferred to RDFLib.
  - **XBRLResolver**: returns configured ZIP URL for XBRL taxonomy packages; validation deferred to Arelle post-extraction.

- **Download Manager**: Wraps pooch for robust fetching with:
  - Stream-to-`.part` file pattern to avoid corrupted partial downloads.
  - ETag/Last-Modified conditional headers from previous manifest; on HTTP 304, returns cached status.
  - Range request support for resume on connection interruption.
  - SHA-256 computation on complete files; verification against known hashes when available.
  - Exponential backoff retry with per-host rate limiting (token bucket).
  - Structured logging of HTTP status, elapsed time, retry count, and final outcome.

- **Validation Pipeline**: Post-download, runs format-appropriate validators:
  - **RDFLib**: `Graph.parse()` for RDF/OWL/SKOS; records triple count or error; optionally serializes to normalized Turtle.
  - **Pronto**: `Ontology()` load for OBO/OWL; records term count, relationships, or error; optionally exports OBO Graph JSON.
  - **Owlready2**: `get_ontology().load()` for OWL; enables optional reasoning checks.
  - **ROBOT** (if installed): subprocess calls for `robot convert` (format conversions) and `robot report` (SPARQL-based QC); captures stdout/stderr and TSV reports.
  - **Arelle** (XBRL): loads taxonomy package, runs validation, records results as JSON.
  - Results written to `validation/<parser>_parse.json` or `validation/robot_report.tsv`.

- **Storage Layout**: pystow-managed root (~/.data/ontology-fetcher) contains:

  ```
  configs/sources.yaml        # ontology specs
  cache/                      # HTTP cache, ETags, partial downloads
  logs/                       # structured JSON logs per run
  ontologies/<id>/<version>/
    original/                 # bit-exact downloaded files
    normalized/               # Turtle, OBO Graph JSON
    validation/               # parse results, QC reports
    manifest.json             # provenance metadata
  ```

- **CLI (`ontofetch`)**: Command-line interface with subcommands:
  - `ontofetch pull [--spec FILE | ID...]`: download ontologies; updates existing or fetches new.
  - `ontofetch show ID [--versions]`: display manifest, versions, or validation status.
  - `ontofetch validate ID[@VERSION] [--robot] [--rdflib] [--pronto]`: re-run validators.
  - Structured output (JSON for machine, formatted for human) based on `--json` flag.

- **Observability**: Structured JSON logs capture:
  - Resolver: source, ID, resolver type, plan URL, planning time.
  - Download: URL, status code, ETag, content length, SHA-256, elapsed time, retries, cache hit/miss.
  - Validation: parser, outcome (success/failure), error messages, metrics (triples, terms).
  - Aggregation: per-run summary with success/failure/cached counts, total bandwidth, total time.

- **Error Handling**: Comprehensive exception handling at each layer:
  - **Custom Exceptions**: OntologyDownloadError, ResolverError, ValidationError, ConfigurationError with descriptive messages and remediation guidance.
  - **Retry Logic**: Exponential backoff for transient failures (ConnectionError, Timeout, HTTP 503/429); permanent failures (401, 404) don't retry.
  - **Graceful Degradation**: Batch operations continue after individual failures; partial progress preserved; failed items logged with details.
  - **Resource Protection**: Catch OSError for disk space, MemoryError for large files, PermissionError for directories; cleanup partial artifacts.

- **Security Layer**: Defense-in-depth protection against common threats:
  - **URL Validation**: validate_url_security() checks scheme (HTTPS preferred), resolves hostname to IP, rejects private ranges (RFC 1918, loopback, link-local).
  - **TLS Verification**: requests.get(verify=True) enforces certificate validation; catch SSLError with remediation message.
  - **Filename Sanitization**: sanitize_filename() strips path separators, leading dots, limits length to 255 chars; prevents directory traversal.
  - **ZIP Safety**: Check is_zipfile(), compressed/uncompressed ratio, member path validation before extraction; prevents ZIP bombs and path traversal.
  - **Size Limits**: Enforce max_download_size_gb before starting download; prevents disk exhaustion from unexpected large files.
  - **Credential Protection**: mask_sensitive_data() redacts API keys, tokens, passwords from all logs before writing.

- **Performance Management**: Configurable resource limits and optimization strategies:
  - **Timeouts**: Per-operation timeouts (30s API calls, 300s downloads, 60s parsers) prevent hung operations; platform-specific implementations.
  - **Concurrency Control**: ThreadPoolExecutor or asyncio with semaphore limits parallel downloads; respects per-host rate limiting.
  - **Memory Limits**: Skip memory-intensive operations (Owlready2 reasoning) for files >500MB; catch MemoryError gracefully.
  - **Streaming**: Use iter_content(chunk_size=1MB) for downloads >100MB; log progress every 10%; prevents memory exhaustion.
  - **Rate Limiting**: Token bucket algorithm per-host; configurable default 4 req/sec; prevents overwhelming source servers.

- **Configuration System**: Hierarchical configuration with validation and overrides:
  - **Schema**: Strongly-typed dataclasses (HTTPConfig, ValidationConfig, LoggingConfig, DefaultsConfig) with defaults.
  - **Loading**: Parse YAML, validate schema, merge per-ontology overrides with defaults.
  - **Environment Overrides**: ONTOFETCH_* env vars take precedence over file config; log overrides for transparency.
  - **Validation**: Check types, ranges, required fields; provide line numbers for YAML errors; fail fast on invalid config.

- **Logging System**: Multi-level structured logging with sensitive data protection:
  - **Levels**: DEBUG (verbose, HTTP details), INFO (operational events), WARNING (degraded but continuing), ERROR (failures).
  - **Formatters**: JSONFormatter for file (machine-readable), human-readable for console.
  - **Rotation**: RotatingFileHandler with size limit, compression of old logs, retention policy cleanup.
  - **Correlation**: generate_correlation_id() per batch operation; attach to all related log entries for tracing.
  - **Sanitization**: mask_sensitive_data() centrally redacts credentials before any output.

## Risks / Trade-offs

- Risk: External service rate limits or transient failures during batch downloads. Mitigation: exponential backoff, per-host rate limiting, graceful failure recording in logs; operators can resume failed items.

- Risk: OLS, BioPortal, or other APIs change response formats or authentication. Mitigation: use well-maintained client libraries (ols-client, ontoportal-client) that abstract API details; monitor upstream releases for breaking changes.

- Risk: Large ontologies (SNOMED, NCIT, ChEBI) exhaust memory during parsing/reasoning. Mitigation: stream-based download, optional validation (skip reasoning for large files), document memory requirements per ontology class.

- Risk: License restrictions on some ontologies (SNOMED, LOINC) require explicit credentials or acceptance. Mitigation: fail closed with clear error messages when restricted ontologies lack credentials; document credential configuration in pystow paths.

- Risk: XBRL ZIP files may contain malicious paths (zip-slip vulnerability). Mitigation: use safe extraction with path validation (Python's `zipfile` with `is_zipfile` check and member path filtering).

- Risk: ROBOT dependency on Java runtime may not be available. Mitigation: make ROBOT integration optional with runtime detection (`shutil.which('robot')`); skip conversion/QC steps if absent.

- Risk: Divergent tokenization between Pronto/RDFLib/Owlready2 on same OWL file. Mitigation: treat parsers as independent validators rather than requiring consensus; log discrepancies for manual review.

- Risk: Stale cached ontologies if ETag/Last-Modified not supported by source. Mitigation: allow force-refresh flag (`--force`) to bypass cache; document staleness detection limits.

- Risk: Configuration drift if sources.yaml manually edited without validation. Mitigation: validate YAML schema on load; provide `ontofetch config validate` command.

- Risk: Credential leakage via logs or error messages exposing API keys. Mitigation: implement centralized sensitive data masking in logging_config.py; apply to all log formatters before writing; test with grep for common secret patterns.

- Risk: SSRF attacks via malicious resolver responses pointing to internal services. Mitigation: validate all URLs before download with validate_url_security(); reject private IP ranges, localhost, link-local addresses; enforce HTTPS.

- Risk: ZIP bomb or path traversal attacks in XBRL taxonomy packages. Mitigation: check compressed/uncompressed size ratio before extraction; validate all ZIP member paths don't contain ".." or start with "/"; extract to temp directory first.

- Risk: Unbounded resource consumption from missing timeouts or memory limits. Mitigation: enforce timeouts on all HTTP requests (30s API, 300s download) and parser operations (60s default); skip reasoning for files >500MB; implement concurrent download limits.

- Risk: Parser hangs on malformed ontologies preventing batch completion. Mitigation: implement platform-specific timeouts (signal.alarm on Unix, threading.Timer on Windows); catch timeout exceptions, log error, continue with remaining ontologies.

- Risk: Log disk exhaustion in long-running deployments. Mitigation: implement RotatingFileHandler with size limits; compress old logs; delete after retention period (30 days default); make all thresholds configurable.

- Risk: Incompatible dependency versions causing runtime errors. Mitigation: pin minimum and maximum versions in requirements.txt (e.g., rdflib>=7.0.0,<8.0.0); test with both minimum and latest versions in CI; document known incompatibilities.

## Migration Plan

1. Scaffold module structure under `src/DocsToKG/OntologyDownload/`: `core.py`, `resolvers.py`, `download.py`, `validators.py`, `config.py`, `logging_config.py`, `cli.py`.
2. Install dependencies via `requirements.txt` or `pyproject.toml` with version pins: `bioregistry>=0.10.0`, `oaklib>=0.5.0`, `ols-client>=0.1.4`, `ontoportal-client>=0.0.3`, `rdflib>=7.0.0,<8.0.0`, `pronto>=2.5.0,<3.0.0`, `owlready2>=0.43`, `arelle>=2.20.0`, `pystow>=0.5.0`, `pooch>=1.7.0`, `pyyaml>=6.0,<7.0`, `requests>=2.28.0,<3.0.0`.
3. Implement core data models (FetchSpec, FetchPlan, FetchResult, Manifest) and resolver protocol; add custom exception classes.
4. Implement configuration system with dataclass schemas (HTTPConfig, ValidationConfig, LoggingConfig) and YAML loading/validation.
5. Implement security utilities: validate_url_security(), sanitize_filename(), mask_sensitive_data().
6. Implement logging system: setup_logging(), JSONFormatter, generate_correlation_id(), rotation cleanup.
7. Implement resolvers one at a time (OBO → OLS → BioPortal → SKOS → XBRL) with unit tests using mocked library responses; add error handling for 401/403/503.
8. Build download manager with security validation, ETag/Range/SHA-256 logic, timeout handling, max file size enforcement; test with local HTTP server fixture.
9. Implement validation pipeline with RDFLib, Pronto, Owlready2, ROBOT, Arelle; add platform-specific timeouts, memory limit checks, error handling; create test fixtures.
10. Build pystow-based storage layer with directory creation, safe ZIP extraction, manifest read/write with validation results.
11. Implement orchestration (fetch_one, fetch_all) with correlation IDs, graceful error handling, concurrency control.
12. Implement CLI using argparse/Click with pull/show/validate/init subcommands; add --log-level, --json, --force flags; comprehensive --help text with examples.
13. Add Python version check (3.9+), platform-specific timeout implementations, pathlib usage throughout.
14. Create comprehensive test suite: resolver mocks, download manager with HTTP server fixture, security tests (SSRF, filename sanitization), configuration validation, error handling, concurrent downloads, log masking, platform compatibility.
15. Create example `sources.yaml` with all resolver types and extensive inline comments; add `ontofetch init` command to generate it.
16. Add integration smoke tests with real API calls (HP from OBO, EFO from OLS) and performance benchmarks.
17. Document installation, configuration (all config fields, env var overrides, pystow paths), CLI usage, troubleshooting (with remediation steps), security considerations.
18. Optional: Add `openspec archive` after deployment to move change to archive/ and create canonical spec in `openspec/specs/ontology-downloader/`.

## Open Questions

- Should we support parallel downloads for batch operations, or serialize to respect global rate limits? *Proposal: support via concurrent_downloads config (default 1); add per-host tracking to rate limiter.*
- What is the preferred handling for ontologies with multiple versions (keep all, prune old, configurable retention)? *Proposal: keep all by default; add optional version_retention_count config.*
- Should validation failures block downstream processing, or just log warnings? *Proposal: log warnings, continue; add --strict mode that exits on validation failure.*
- Do we need integration with OAK's adapter system for post-download ontology querying, or keep downloader focused solely on acquisition? *Proposal: downloader only; users can use OAK separately on downloaded files.*
- Should we expose programmatic Python API alongside CLI, or CLI-first with API added later? *Proposal: expose fetch_one/fetch_all in **init**.py from start; minimal overhead.*
- What authentication patterns should we support for restricted ontologies (API keys in pystow, OAuth, certificate-based)? *Proposal: API keys via pystow initially; OAuth/certs if user demand emerges.*
- What level of log detail should default INFO include (vs DEBUG)? *Proposal: INFO = operational milestones (download started/completed, validation outcome); DEBUG = HTTP headers, cache decisions, retry attempts.*
- Should we implement automatic retry on checksum mismatch, or require manual intervention? *Proposal: automatic retry once (treat as cache corruption); alert if persists.*
- What should be the behavior when disk space is low but not exhausted? *Proposal: check available space before large downloads; warn if <10% free; block if <1GB.*
- Should concurrent downloads share bandwidth fairly, or first-come-first-served? *Proposal: ThreadPoolExecutor default (FCFS); document asyncio option for fairness-aware scheduling.*
