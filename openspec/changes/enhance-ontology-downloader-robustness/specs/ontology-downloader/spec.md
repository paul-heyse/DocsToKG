# Ontology Downloader Enhancement Specification

## ADDED Requirements

### Requirement: Centralized Optional Dependency Management

The system SHALL provide a centralized module for managing optional dependencies with consistent fallback behavior across all components.

#### Scenario: Import pystow via optdeps when available

- **WHEN** a module imports pystow via `optdeps.get_pystow()` and pystow is installed
- **THEN** the function returns the real pystow module with full functionality

#### Scenario: Use pystow fallback stub when not installed

- **WHEN** a module imports pystow via `optdeps.get_pystow()` and pystow is not installed
- **THEN** the function returns a lightweight stub that provides Path-based fallback using PYSTOW_HOME environment variable

#### Scenario: Import rdflib via optdeps with stub fallback

- **WHEN** validators import rdflib via `optdeps.get_rdflib()` and rdflib is not installed
- **THEN** the function returns a stub with Graph class that records parse source and serializes to destination

#### Scenario: Import pronto via optdeps with stub fallback

- **WHEN** validators import pronto via `optdeps.get_pronto()` and pronto is not installed
- **THEN** the function returns a stub with Ontology class that provides placeholder terms and minimal dump functionality

#### Scenario: Import owlready2 via optdeps with stub fallback

- **WHEN** validators import owlready2 via `optdeps.get_owlready2()` and owlready2 is not installed
- **THEN** the function returns a stub with get_ontology function returning placeholder loaded ontology

#### Scenario: Single source of truth for all modules

- **WHEN** core.py, resolvers.py, and validators.py all need pystow
- **THEN** all modules import from optdeps.get_pystow() without local stub duplication

### Requirement: CLI Formatting Utility Module

The system SHALL provide reusable CLI formatting utilities for consistent table and summary output across all subcommands.

#### Scenario: Format table with headers and rows

- **WHEN** CLI needs to display tabular data with headers ["id", "status"] and rows [["hp", "success"], ["efo", "cached"]]
- **THEN** cli_utils.format_table() returns properly aligned table with column separators and header underline

#### Scenario: Format validation summary with status indicators

- **WHEN** CLI needs to display validation results {"rdflib": {"ok": true}, "pronto": {"ok": false, "error": "timeout"}}
- **THEN** cli_utils.format_validation_summary() returns table with validator, status, and details columns

#### Scenario: Reuse formatting across multiple subcommands

- **WHEN** show, validate, and pull subcommands all need table formatting
- **THEN** all subcommands import and use cli_utils functions without duplication

### Requirement: Pydantic v2 Configuration Models

The system SHALL use Pydantic v2 BaseModel classes for all configuration parsing with built-in validation, environment variable merging, and JSON Schema generation.

#### Scenario: Parse configuration with Pydantic validation

- **WHEN** sources.yaml contains defaults.http.max_retries=-1 (invalid negative value)
- **THEN** Pydantic field validator raises ValidationError with clear message "max_retries must be >= 0"

#### Scenario: Apply default values via Pydantic Field definitions

- **WHEN** sources.yaml omits defaults.http.timeout_sec field
- **THEN** Pydantic model applies default value of 30 from Field(default=30) definition

#### Scenario: Validate rate limit format with Pydantic pattern

- **WHEN** sources.yaml contains defaults.http.per_host_rate_limit="invalid"
- **THEN** Pydantic pattern validator raises ValidationError with expected format message

#### Scenario: Environment variable override via Pydantic settings

- **WHEN** ONTOFETCH_MAX_RETRIES=10 environment variable is set
- **THEN** Pydantic settings merge applies env value overriding config file value

#### Scenario: Generate JSON Schema from Pydantic models

- **WHEN** documentation generator requests configuration schema
- **THEN** Pydantic model.model_json_schema() returns complete JSON Schema with types, defaults, and descriptions

#### Scenario: Nested model validation for HTTP configuration

- **WHEN** sources.yaml contains defaults.http section with nested fields
- **THEN** Pydantic parses into nested DownloadConfiguration model with independent field validation

### Requirement: Per-Service Rate Limiting

The system SHALL enforce rate limits per service identifier (obo, ols, bioportal) in addition to per-host limits to prevent service-specific throttling.

#### Scenario: Apply service-specific rate limit for OLS

- **WHEN** downloading from OLS API with rate_limits.ols="5/second" configured
- **THEN** token bucket enforces 5 requests per second for all OLS API calls regardless of host

#### Scenario: Apply service-specific rate limit for BioPortal

- **WHEN** downloading from BioPortal with rate_limits.bioportal="1/second" configured
- **THEN** token bucket enforces 1 request per second for all BioPortal API calls

#### Scenario: Fall back to per-host limit when service not configured

- **WHEN** downloading from service without specific rate_limits entry
- **THEN** token bucket uses per_host_rate_limit default of 4 requests per second

#### Scenario: Independent buckets for different services on same host

- **WHEN** OLS and BioPortal both resolve to same host but have different service rate limits
- **THEN** system maintains separate token buckets with keys "ols:host" and "bioportal:host"

#### Scenario: Parse rate limit with per-minute unit

- **WHEN** rate_limits.ols="60/min" is configured
- **THEN** rate limit parser converts to 1.0 requests per second for token bucket

#### Scenario: Parse rate limit with per-hour unit

- **WHEN** rate_limits.bioportal="100/hour" is configured
- **THEN** rate limit parser converts to approximately 0.0278 requests per second

### Requirement: HEAD Request with Media Type Validation

The system SHALL perform preliminary HEAD request before GET to validate Content-Type and Content-Length against plan expectations.

#### Scenario: HEAD request confirms media type matches plan

- **WHEN** FetchPlan specifies media_type="application/rdf+xml" and HEAD response returns Content-Type "application/rdf+xml"
- **THEN** system proceeds with full GET request without warning

#### Scenario: HEAD request detects media type mismatch

- **WHEN** FetchPlan specifies media_type="application/rdf+xml" but HEAD response returns Content-Type "text/html"
- **THEN** system logs warning "Expected media type application/rdf+xml but received text/html" and proceeds with GET if override enabled

#### Scenario: HEAD request retrieves Content-Length early

- **WHEN** HEAD request returns Content-Length: 6442450944 (6GB)
- **THEN** system detects file exceeds max_download_size_gb=5 and aborts before full GET request

#### Scenario: HEAD request failure does not block download

- **WHEN** HEAD request times out or returns 405 Method Not Allowed
- **THEN** system logs warning and proceeds with GET request without HEAD validation

#### Scenario: Configuration flag disables media type validation

- **WHEN** config contains http.validate_media_type=false
- **THEN** system skips HEAD request media type check and proceeds directly to GET

### Requirement: Hardened URL Validation with Host Allowlist and IDN Support

The system SHALL validate URLs against optional host allowlist and normalize Internationalized Domain Names to prevent SSRF attacks in multi-tenant scenarios.

#### Scenario: Punycode normalization for IDN domain

- **WHEN** resolver returns URL with IDN host "münchen.example.org"
- **THEN** validate_url_security normalizes to punycode "xn--mnchen-3ya.example.org" before DNS resolution

#### Scenario: Host allowlist permits approved domain

- **WHEN** config contains http.allowed_hosts=["example.org", "purl.obolibrary.org"] and URL host is "purl.obolibrary.org"
- **THEN** validate_url_security accepts URL and proceeds with download

#### Scenario: Host allowlist rejects unapproved domain

- **WHEN** config contains http.allowed_hosts=["example.org"] and URL host is "malicious.com"
- **THEN** validate_url_security raises ConfigError "Host malicious.com not in allowlist"

#### Scenario: No allowlist allows any non-private host

- **WHEN** config does not specify http.allowed_hosts
- **THEN** validate_url_security accepts any HTTPS URL that resolves to public IP address

#### Scenario: IDN normalization prevents homograph attack

- **WHEN** resolver returns URL with visually similar but different Unicode domain
- **THEN** punycode normalization detects and rejects suspicious patterns

### Requirement: Safe Tar Archive Extraction

The system SHALL safely extract tar.gz and tar.xz archives with path validation to prevent tar-slip attacks and compression bomb detection.

#### Scenario: Extract tar.gz with valid paths

- **WHEN** tar.gz archive contains members with paths "schemas/core.xsd" and "taxonomies/base.xml"
- **THEN** extract_tar_safe extracts files to destination/schemas/core.xsd and destination/taxonomies/base.xml

#### Scenario: Reject tar member with path traversal

- **WHEN** tar.gz archive contains member with path "../../etc/shadow"
- **THEN** extract_tar_safe detects traversal, raises ConfigError, and aborts extraction

#### Scenario: Detect tar compression bomb

- **WHEN** tar.gz archive has compressed size 1MB but uncompressed size 10GB (ratio >10:1)
- **THEN** extract_tar_safe detects potential bomb, logs error, and aborts extraction

#### Scenario: Extract tar.xz with valid paths

- **WHEN** tar.xz archive contains taxonomy files
- **THEN** extract_tar_safe handles xz compression and extracts safely with same path validation as tar.gz

#### Scenario: Handle absolute paths in tar members

- **WHEN** tar.gz contains member with absolute path "/tmp/file.xml"
- **THEN** extract_tar_safe detects absolute path, rejects it, and aborts extraction

### Requirement: Deterministic TTL Normalization with Canonical Hashing

The system SHALL canonicalize Turtle output by sorting prefixes and triples before serialization and compute stable hash of canonical form.

#### Scenario: Sort prefixes alphabetically

- **WHEN** RDFLib graph contains prefixes @prefix rdf, @prefix owl, @prefix rdfs in arbitrary order
- **THEN** canonicalize_ttl sorts to @prefix owl, @prefix rdf, @prefix rdfs in output

#### Scenario: Sort triples by subject, predicate, object

- **WHEN** RDFLib graph contains triples in arbitrary insertion order
- **THEN** canonicalize_ttl sorts triples lexicographically by (subject, predicate, object) tuple

#### Scenario: Compute stable hash of canonical TTL

- **WHEN** same ontology is downloaded twice and canonicalized
- **THEN** SHA-256 hash of canonical TTL is identical both times

#### Scenario: Different parse orders yield same normalized hash

- **WHEN** same ontology is parsed with different RDF parsers producing different triple orders
- **THEN** canonicalize_ttl produces identical normalized output and hash

#### Scenario: Store normalized hash in manifest

- **WHEN** validation completes and normalized TTL is written
- **THEN** manifest.json includes normalized_sha256 field with canonical hash

### Requirement: Subprocess Isolation for Memory-Intensive Validators

The system SHALL execute pronto and owlready2 validators in short-lived subprocesses to prevent memory fragmentation in long-running batch operations.

#### Scenario: Execute pronto validation in subprocess

- **WHEN** validate_pronto is called with large OBO file
- **THEN** system spawns subprocess with Python interpreter, executes pronto parsing, writes results to JSON, and subprocess terminates

#### Scenario: Read validation results from subprocess

- **WHEN** pronto subprocess completes successfully and writes validation/pronto_result.json
- **THEN** parent process reads JSON, parses results, and returns ValidationResult

#### Scenario: Subprocess timeout enforcement

- **WHEN** pronto subprocess exceeds parser_timeout_sec=60 configuration
- **THEN** parent process terminates subprocess and records timeout in ValidationResult

#### Scenario: Execute owlready2 validation in subprocess

- **WHEN** validate_owlready2 is called with OWL file
- **THEN** system spawns subprocess for owlready2 loading, captures entity counts, and terminates subprocess

#### Scenario: Memory isolation prevents parent process leak

- **WHEN** batch pull processes 100 ontologies with pronto validation
- **THEN** parent process memory usage remains stable because each subprocess releases memory on termination

#### Scenario: Subprocess error captured and logged

- **WHEN** pronto subprocess raises exception and exits with non-zero code
- **THEN** parent process captures stderr, logs error, and returns ValidationResult with ok=false

### Requirement: SPDX License Normalization

The system SHALL normalize resolver-reported licenses to SPDX identifiers before enforcing allowlist to handle common variants and aliases.

#### Scenario: Normalize CC-BY variant to SPDX

- **WHEN** resolver reports license="CC-BY" or "CC BY 4.0"
- **THEN** normalize_license_to_spdx returns "CC-BY-4.0"

#### Scenario: Normalize CC0 variant to SPDX

- **WHEN** resolver reports license="CC0" or "Public Domain"
- **THEN** normalize_license_to_spdx returns "CC0-1.0"

#### Scenario: Normalize Apache variants to SPDX

- **WHEN** resolver reports license="Apache" or "Apache License 2.0"
- **THEN** normalize_license_to_spdx returns "Apache-2.0"

#### Scenario: Compare normalized licenses against allowlist

- **WHEN** accept_licenses=["CC-BY-4.0"] and resolver reports "CC BY 4.0"
- **THEN** _ensure_license_allowed normalizes to "CC-BY-4.0", finds match in allowlist, and allows download

#### Scenario: Unknown license returns original string

- **WHEN** resolver reports license="Proprietary Custom License v1"
- **THEN** normalize_license_to_spdx returns original string unchanged for allowlist comparison

### Requirement: Automatic Multi-Resolver Fallback

The system SHALL automatically try resolvers in prefer_source order when a resolver fails, increasing download success rates without configuration changes.

#### Scenario: First resolver succeeds without fallback

- **WHEN** FetchSpec uses resolver="obo" and OBOResolver successfully returns FetchPlan
- **THEN** no fallback is attempted and FetchPlan is used for download

#### Scenario: First resolver fails, fallback to second

- **WHEN** prefer_source=["obo", "ols", "bioportal"], OBOResolver raises ResolverError "No download URL found"
- **THEN** FallbackResolver logs "OBOResolver failed, trying OLSResolver" and attempts OLS resolution

#### Scenario: Second resolver succeeds

- **WHEN** OBOResolver fails and OLSResolver returns valid FetchPlan
- **THEN** FallbackResolver returns OLS FetchPlan and fetch proceeds with OLS source

#### Scenario: All resolvers exhausted

- **WHEN** prefer_source=["obo", "ols", "bioportal"] and all three resolvers raise errors
- **THEN** FallbackResolver logs all attempts and raises ResolverError "All resolvers exhausted for ontology <id>"

#### Scenario: Log fallback sequence for observability

- **WHEN** fallback occurs through multiple resolvers
- **THEN** each attempt is logged with fields: resolver_name, attempt_number, outcome, error_message

#### Scenario: Configuration flag disables fallback

- **WHEN** config contains resolver_fallback_enabled=false
- **THEN** system uses only the specified resolver without fallback attempts

### Requirement: Polite API Headers for Resolver Requests

The system SHALL include polite HTTP headers (User-Agent, From, X-Request-ID) in resolver API requests to reduce throttling and improve reproducibility.

#### Scenario: Include User-Agent in OLS API request

- **WHEN** OLSResolver queries OLS API for ontology metadata
- **THEN** request includes User-Agent: "DocsToKG-OntologyDownloader/1.0 (<contact@example.org>)"

#### Scenario: Include From header in BioPortal API request

- **WHEN** BioPortalResolver queries BioPortal API
- **THEN** request includes From: "<contact@example.org>" header from polite_headers config

#### Scenario: Include X-Request-ID for request tracing

- **WHEN** any resolver makes API request
- **THEN** request includes X-Request-ID: "<correlation_id>-<timestamp>" for distributed tracing

#### Scenario: Configure polite headers in defaults

- **WHEN** sources.yaml contains defaults.http.polite_headers with User-Agent and From
- **THEN** all resolver API requests use configured headers

#### Scenario: Default polite headers when not configured

- **WHEN** sources.yaml omits polite_headers section
- **THEN** system uses default headers with generic User-Agent and no From header

### Requirement: Linked Open Vocabularies (LOV) Resolver

The system SHALL provide LOVResolver for fetching SKOS vocabularies and RDF schemas from Linked Open Vocabularies repository.

#### Scenario: Resolve vocabulary from LOV by URI

- **WHEN** FetchSpec has resolver="lov" and extras.uri="<http://purl.org/vocommons/voaf>"
- **THEN** LOVResolver queries LOV API, retrieves vocabulary metadata, and returns FetchPlan with Turtle download URL

#### Scenario: LOV resolver sets appropriate media type

- **WHEN** LOVResolver constructs FetchPlan for SKOS vocabulary
- **THEN** FetchPlan includes media_type="text/turtle" for content negotiation

#### Scenario: LOV API unavailable fallback

- **WHEN** LOV API returns HTTP 503 Service Unavailable
- **THEN** LOVResolver raises ResolverError allowing fallback to other resolvers

#### Scenario: LOV vocabulary metadata included in plan

- **WHEN** LOV API returns vocabulary version and license
- **THEN** FetchPlan includes version and license fields for provenance

### Requirement: Ontobee PURL Resolver

The system SHALL provide OntobeeResolver for constructing PURLs for OBO Foundry ontologies when primary resolvers are unavailable.

#### Scenario: Construct Ontobee PURL for OBO ontology

- **WHEN** FetchSpec has resolver="ontobee" and id="hp"
- **THEN** OntobeeResolver returns FetchPlan with url="<http://purl.obolibrary.org/obo/hp.owl>"

#### Scenario: Support multiple OBO format variants

- **WHEN** FetchSpec has target_formats=["obo", "owl"] preference
- **THEN** OntobeeResolver constructs PURL for preferred format (hp.obo or hp.owl)

#### Scenario: Ontobee as fallback resolver

- **WHEN** prefer_source=["obo", "ols", "ontobee"] and OBO/OLS fail
- **THEN** fallback tries Ontobee PURL as last resort

#### Scenario: Ontobee PURL format validation

- **WHEN** Ontobee constructs PURL for ontology id
- **THEN** resolver validates id format matches OBO prefix pattern before returning FetchPlan

### Requirement: Remote Storage Backend via fsspec

The system SHALL support remote storage backends (S3, GCS, Azure, HTTP) via fsspec when ONTOFETCH_STORAGE_URL environment variable is set.

#### Scenario: Use local storage when no storage URL configured

- **WHEN** ONTOFETCH_STORAGE_URL environment variable is not set
- **THEN** system uses LocalStorageBackend with pystow-based paths

#### Scenario: Use S3 storage when S3 URL configured

- **WHEN** ONTOFETCH_STORAGE_URL="s3://my-bucket/ontologies" is set
- **THEN** system uses FsspecStorageBackend with s3fs filesystem for manifest and artifact operations

#### Scenario: Read manifest from remote storage

- **WHEN** using S3 storage and previous manifest exists
- **THEN** storage backend reads manifest.json from s3://my-bucket/ontologies/<id>/<version>/manifest.json

#### Scenario: Write artifacts to remote storage

- **WHEN** download completes and using remote storage
- **THEN** storage backend writes downloaded file to s3://my-bucket/ontologies/<id>/<version>/original/<filename>

#### Scenario: Missing fsspec dependency with storage URL

- **WHEN** ONTOFETCH_STORAGE_URL is set but fsspec not installed
- **THEN** system raises ConfigError "fsspec required for remote storage. Install: pip install fsspec s3fs"

#### Scenario: Cache directory remains local with remote storage

- **WHEN** using remote storage for ontologies
- **THEN** pooch cache directory remains local for download performance, only final artifacts stored remotely

### Requirement: CLI Plan Command for Download Preview

The system SHALL provide `ontofetch plan` subcommand to preview FetchPlan without executing download for workflow development and debugging.

#### Scenario: Plan command resolves and displays FetchPlan

- **WHEN** user runs `ontofetch plan hp --resolver obo`
- **THEN** CLI resolves FetchPlan for HP ontology and prints JSON with url, headers, version, license, media_type fields

#### Scenario: Plan command with human-readable output

- **WHEN** user runs `ontofetch plan hp` without --json flag
- **THEN** CLI displays FetchPlan fields in formatted table with field names and values

#### Scenario: Plan command respects resolver fallback

- **WHEN** user runs `ontofetch plan hp` with prefer_source=["obo", "ols"] and OBO fails
- **THEN** plan command attempts fallback and displays which resolver succeeded

#### Scenario: Plan command shows headers for debugging

- **WHEN** FetchPlan includes Authorization or other headers
- **THEN** plan output masks sensitive values but shows header presence for debugging

#### Scenario: Plan command fails on resolver error

- **WHEN** all resolvers fail for requested ontology
- **THEN** plan command exits with code 1 and displays resolver errors

### Requirement: CLI Doctor Command for Environment Diagnostics

The system SHALL provide `ontofetch doctor` subcommand to diagnose environment issues and provide actionable remediation guidance.

#### Scenario: Doctor checks filesystem permissions

- **WHEN** user runs `ontofetch doctor`
- **THEN** command checks write permissions for data directories (configs, cache, logs, ontologies) and reports status

#### Scenario: Doctor checks BioPortal API key

- **WHEN** doctor runs and BioPortal API key file exists at expected location
- **THEN** command reports "BioPortal API key: configured at <path>" and validates key format

#### Scenario: Doctor checks OLS API accessibility

- **WHEN** doctor runs and attempts to connect to OLS API
- **THEN** command reports "OLS API: accessible" or "OLS API: unreachable" with HTTP status

#### Scenario: Doctor checks disk space

- **WHEN** doctor runs and queries available disk space
- **THEN** command reports free space in data directory and warns if <5GB available

#### Scenario: Doctor provides remediation hints

- **WHEN** doctor detects missing API key
- **THEN** output includes "To configure: echo 'YOUR_API_KEY' > <path>/bioportal_api_key.txt"

#### Scenario: Doctor checks optional dependencies

- **WHEN** doctor runs and checks for rdflib, pronto, owlready2, robot installations
- **THEN** command reports which validators are available and which require installation

#### Scenario: Doctor JSON output for automation

- **WHEN** user runs `ontofetch doctor --json`
- **THEN** command outputs diagnostic results as JSON with boolean status fields for scripting

### Requirement: CLI Dry-Run Mode for Pull Command

The system SHALL provide `--dry-run` flag for pull subcommand to preview planned actions without executing downloads or validation.

#### Scenario: Dry-run logs planned downloads

- **WHEN** user runs `ontofetch pull --spec sources.yaml --dry-run`
- **THEN** CLI resolves all FetchPlans, logs planned downloads, and exits without downloading

#### Scenario: Dry-run shows which ontologies would be cached

- **WHEN** user runs `ontofetch pull hp efo --dry-run` and previous manifests exist with ETags
- **THEN** CLI indicates "hp: would check cache (ETag present)" and "efo: would download (no cache)"

#### Scenario: Dry-run reports total planned download size

- **WHEN** dry-run resolves all FetchPlans with Content-Length available
- **THEN** CLI summary shows "Total download size: 1.2 GB (3 ontologies, 2 cached)"

#### Scenario: Dry-run respects resolver fallback

- **WHEN** dry-run encounters resolver that would fail
- **THEN** CLI shows fallback sequence that would be attempted in real run

#### Scenario: Dry-run validates configuration

- **WHEN** user runs pull with --dry-run and sources.yaml has validation errors
- **THEN** CLI detects errors and reports them without attempting downloads

## MODIFIED Requirements

### Requirement: Declarative YAML Configuration with Pydantic Models

The system SHALL support configuration via sources.yaml with Pydantic v2 models providing validation, defaults, environment variable merging, and JSON Schema generation.

#### Scenario: Parse defaults section with Pydantic validation

- **WHEN** sources.yaml contains defaults.accept_licenses=["CC-BY-4.0", "CC0-1.0"]
- **THEN** Pydantic DefaultsConfig model parses licenses into validated list with type checking

#### Scenario: Parse ontology list into FetchSpec with Pydantic

- **WHEN** sources.yaml contains ontologies list with id, resolver, target_formats per entry
- **THEN** Pydantic parser creates validated FetchSpec objects with merged defaults

#### Scenario: Pydantic field validation catches configuration errors

- **WHEN** sources.yaml contains defaults.http.timeout_sec=-10 (invalid negative)
- **THEN** Pydantic raises ValidationError with message "timeout_sec must be greater than 0"

#### Scenario: Pydantic provides clear error messages

- **WHEN** sources.yaml has ontology entry missing required field 'id'
- **THEN** Pydantic error shows "Field required: id at ontologies[2]" with location context

#### Scenario: Environment variables override via Pydantic settings

- **WHEN** ONTOFETCH_MAX_RETRIES=10 env var is set and sources.yaml has max_retries=5
- **THEN** Pydantic settings merge applies max_retries=10 with priority to environment

#### Scenario: Generate JSON Schema from Pydantic models

- **WHEN** documentation tooling requests configuration schema
- **THEN** Pydantic model.model_json_schema() generates complete JSON Schema with all fields, types, constraints, and descriptions

#### Scenario: Nested Pydantic models for configuration sections

- **WHEN** sources.yaml has defaults.http, defaults.validation, defaults.logging sections
- **THEN** Pydantic parses into nested DownloadConfiguration, ValidationConfig, LoggingConfiguration models with independent validation

### Requirement: Robust HTTP Download with Caching, HEAD Validation, and Service Rate Limits

The system SHALL implement robust HTTP download logic with conditional requests, resume support, checksums, retry, per-service rate limiting, and preliminary HEAD validation using pooch as the underlying cache mechanism.

#### Scenario: HEAD request validates media type before GET

- **WHEN** FetchPlan specifies media_type and http.validate_media_type=true
- **THEN** StreamingDownloader issues HEAD request, validates Content-Type matches expected media type before full GET

#### Scenario: HEAD request retrieves Content-Length for early size check

- **WHEN** HEAD request returns Content-Length exceeding max_download_size_gb
- **THEN** download manager aborts before full GET with error "File size exceeds limit"

#### Scenario: Conditional GET with ETag returns 304 Not Modified

- **WHEN** previous manifest exists with ETag and server responds with HTTP 304
- **THEN** download manager returns FetchResult with status='cached' without re-downloading

#### Scenario: Resume partial download with Range header

- **WHEN** .part file exists from interrupted download and server supports Range requests
- **THEN** download manager sends Range header starting from partial file size and appends remaining bytes

#### Scenario: SHA-256 verification after complete download

- **WHEN** file download completes successfully
- **THEN** download manager computes SHA-256 hash and records it in FetchResult

#### Scenario: Exponential backoff retry on transient failure

- **WHEN** server responds with HTTP 503 or network timeout occurs
- **THEN** download manager retries up to max_retries times with exponential backoff

#### Scenario: Per-service rate limiting enforces configured limits

- **WHEN** multiple downloads from OLS service with rate_limits.ols="5/second"
- **THEN** download manager enforces token bucket limit of 5 req/sec for OLS service

#### Scenario: Per-host rate limiting as fallback

- **WHEN** downloading from service without specific rate_limits entry
- **THEN** download manager enforces per_host_rate_limit default of 4 req/sec

### Requirement: Multi-Parser Validation Pipeline with Subprocess Isolation

The system SHALL validate downloaded ontologies using multiple parsers with subprocess isolation for memory-intensive validators (pronto, owlready2) and deterministic normalization for RDFLib output.

#### Scenario: RDFLib parses and produces canonical Turtle

- **WHEN** Turtle file is downloaded and RDFLib validation runs
- **THEN** validator parses graph, canonicalizes triples and prefixes, computes normalized SHA-256, writes validation/rdflib_parse.json with {"ok": true, "triples": N, "normalized_sha256": "<hash>"}

#### Scenario: Pronto runs in subprocess for memory isolation

- **WHEN** OBO file validation with Pronto is requested
- **THEN** validate_pronto spawns subprocess, executes pronto parsing, writes results to validation/pronto_parse.json, subprocess terminates releasing memory

#### Scenario: Owlready2 runs in subprocess for reasoning

- **WHEN** OWL file validation with Owlready2 is requested and file size < skip_reasoning_if_size_mb
- **THEN** validate_owlready2 spawns subprocess, loads ontology, counts entities, writes results, subprocess terminates

#### Scenario: Subprocess timeout enforced for validators

- **WHEN** pronto subprocess exceeds parser_timeout_sec configuration
- **THEN** parent process terminates subprocess and writes validation result with timeout error

#### Scenario: RDFLib remains in-process for performance

- **WHEN** RDFLib validation runs on typical OWL/Turtle file
- **THEN** validation executes in parent process (not subprocess) for fast execution

#### Scenario: ROBOT integration skipped when not installed

- **WHEN** ROBOT validation requested but `robot` command not found via shutil.which()
- **THEN** validator logs warning and skips ROBOT steps without failing entire validation

#### Scenario: Arelle validates XBRL taxonomy package

- **WHEN** XBRL ZIP file downloaded and Arelle validation runs
- **THEN** validator extracts taxonomy, runs Arelle validation, writes validation/arelle_validation.json

### Requirement: Comprehensive Provenance Manifests with Normalized Hashes

The system SHALL record comprehensive provenance metadata for each downloaded ontology in manifest.json including source URL, resolver, version, license, ETag, Last-Modified, SHA-256, normalized SHA-256, fingerprint, timestamps, and validation status.

#### Scenario: Manifest records all standard provenance fields

- **WHEN** ontology successfully downloaded and validated
- **THEN** manifest.json contains fields: id, resolver, url, filename, version, license, status, sha256, etag, last_modified, downloaded_at, target_formats

#### Scenario: Manifest includes normalized SHA-256 hash

- **WHEN** RDFLib validator produces canonical TTL normalization
- **THEN** manifest.json includes normalized_sha256 field with SHA-256 hash of canonical TTL content

#### Scenario: Manifest includes composite fingerprint

- **WHEN** manifest is written after validation completes
- **THEN** manifest.json includes fingerprint field combining original SHA-256, normalized SHA-256, and metadata for unique identification

#### Scenario: Manifest used for cache invalidation decision

- **WHEN** subsequent download request occurs for same ontology
- **THEN** system reads previous manifest.json and uses etag/last_modified values in conditional request headers

#### Scenario: Manifest records validation outcomes

- **WHEN** validation pipeline runs on downloaded ontology
- **THEN** manifest.json includes validation section with results from all validators

#### Scenario: Manifest validation enforces required fields

- **WHEN** _write_manifest attempts to write manifest with missing sha256
- **THEN** validation raises ConfigurationError "Manifest field 'sha256' must be populated"

### Requirement: License Compliance Enforcement with SPDX Normalization

The system SHALL check ontology licenses against configurable allowlist using SPDX-normalized identifiers and fail closed when encountering restricted licenses without explicit acceptance.

#### Scenario: Normalize license before allowlist check

- **WHEN** resolver reports license="CC BY 4.0" and accept_licenses=["CC-BY-4.0"]
- **THEN** system normalizes "CC BY 4.0" to "CC-BY-4.0", finds match in allowlist, allows download

#### Scenario: License allowlist permits download after normalization

- **WHEN** ontology has license="CC0" (normalized to "CC0-1.0") and accept_licenses includes "CC0-1.0"
- **THEN** download proceeds normally

#### Scenario: License not in allowlist blocks download

- **WHEN** ontology has license="Proprietary" and it is not in accept_licenses
- **THEN** system logs error and raises ConfigurationError indicating license restriction

#### Scenario: Unknown license variant returns original

- **WHEN** resolver reports license="Custom Open License v2"
- **THEN** normalize_license_to_spdx returns original string for allowlist comparison

#### Scenario: Missing license treated as restricted

- **WHEN** ontology resolver returns license=None
- **THEN** system skips license check if allowlist empty, otherwise treats as restricted unless bypass configured

### Requirement: Security and Integrity Validation with Enhanced SSRF Protection

The system SHALL enforce security best practices including HTTPS-only downloads, certificate verification, enhanced URL validation with IDN normalization and optional allowlist, credential protection, safe archive extraction for ZIP and tar formats.

#### Scenario: HTTPS enforced for all downloads

- **WHEN** FetchPlan contains url with http:// scheme (not https://)
- **THEN** validate_url_security logs warning "Upgrading insecure HTTP to HTTPS" and upgrades scheme

#### Scenario: IDN punycode normalization prevents homograph attacks

- **WHEN** URL contains Internationalized Domain Name with Unicode characters
- **THEN** validate_url_security converts to punycode before DNS resolution

#### Scenario: Host allowlist prevents unauthorized domains

- **WHEN** http.allowed_hosts is configured and URL host not in allowlist
- **THEN** validate_url_security raises ConfigError "Host not in allowlist" before DNS resolution

#### Scenario: URL validation prevents SSRF to private IPs

- **WHEN** resolver returns FetchPlan with url pointing to private IP range
- **THEN** validate_url_security rejects URL, logs error "Rejected download from private IP address"

#### Scenario: TLS certificate verification enabled

- **WHEN** downloading from HTTPS URL
- **THEN** requests library uses verify=True for certificate validation and fails on invalid certificates

#### Scenario: API keys masked in logs

- **WHEN** logging BioPortal request with Authorization header containing API key
- **THEN** log formatter replaces key with "***masked***" before writing

#### Scenario: Safe filename sanitization

- **WHEN** Content-Disposition header contains path separators (e.g., "../evil.owl")
- **THEN** sanitize_filename removes separators and logs warning

#### Scenario: Maximum file size limit enforced early

- **WHEN** HEAD request Content-Length indicates file >max_download_size_gb
- **THEN** download manager aborts before full GET with clear error message

#### Scenario: ZIP extraction prevents path traversal

- **WHEN** extracting XBRL ZIP with member path "../../etc/passwd"
- **THEN** extract_zip_safe detects traversal, raises ConfigError, aborts extraction

#### Scenario: Tar extraction prevents path traversal

- **WHEN** extracting tar.gz with member path containing ".." components
- **THEN** extract_tar_safe detects traversal, raises ConfigError, aborts extraction

#### Scenario: Tar compression bomb detection

- **WHEN** tar.gz has compression ratio >10:1 (potential bomb)
- **THEN** extract_tar_safe detects threat, logs error, aborts extraction

### Requirement: CLI with Pull/Show/Validate/Plan/Doctor Operations

The system SHALL provide command-line interface (ontofetch) with subcommands for downloading (pull), inspecting (show), validating (validate), planning (plan), and diagnosing (doctor) with both batch and single-ontology modes.

#### Scenario: Pull ontologies from sources.yaml

- **WHEN** user runs `ontofetch pull --spec sources.yaml`
- **THEN** CLI reads sources.yaml, iterates through ontology list, downloads each with progress reporting

#### Scenario: Pull with dry-run shows planned actions

- **WHEN** user runs `ontofetch pull --spec sources.yaml --dry-run`
- **THEN** CLI resolves all FetchPlans, logs planned downloads with size estimates, exits without downloading

#### Scenario: Pull single ontology by ID

- **WHEN** user runs `ontofetch pull hp --resolver obo --target-formats owl,obo`
- **THEN** CLI constructs FetchSpec for HP and downloads using OBOResolver with format fallback

#### Scenario: Force refresh bypasses cache

- **WHEN** user runs `ontofetch pull hp --force`
- **THEN** CLI ignores ETag/Last-Modified from previous manifest and forces fresh download

#### Scenario: Show ontology manifest

- **WHEN** user runs `ontofetch show hp`
- **THEN** CLI reads ontologies/hp/<latest-version>/manifest.json and displays provenance fields

#### Scenario: Re-run validation on existing ontology

- **WHEN** user runs `ontofetch validate hp@2024-10-31 --rdflib --pronto`
- **THEN** CLI locates original file and re-runs RDFLib and Pronto validators (pronto in subprocess), updates validation outputs

#### Scenario: Plan command previews FetchPlan

- **WHEN** user runs `ontofetch plan hp --resolver obo --json`
- **THEN** CLI resolves FetchPlan and outputs JSON with url, headers, version, license, media_type without downloading

#### Scenario: Doctor diagnoses environment

- **WHEN** user runs `ontofetch doctor`
- **THEN** CLI checks filesystem permissions, API credentials, disk space, network connectivity, and reports status with remediation hints

#### Scenario: Doctor JSON output for automation

- **WHEN** user runs `ontofetch doctor --json`
- **THEN** CLI outputs diagnostic results as JSON with boolean status fields for automated monitoring

#### Scenario: Machine-readable JSON output

- **WHEN** user runs `ontofetch show hp --json`
- **THEN** CLI outputs complete manifest.json content as valid JSON to stdout

### Requirement: Source-Agnostic Resolver Registry with Extended Coverage

The system SHALL provide resolver registry that maps ontology source identifiers (OBO, OLS, BioPortal, SKOS, XBRL, LOV, Ontobee) to concrete resolver implementations with automatic fallback through preferred source order.

#### Scenario: Resolve OBO Foundry ontology by prefix

- **WHEN** FetchSpec with resolver="obo" and id="hp" is provided
- **THEN** OBOResolver uses Bioregistry functions to return FetchPlan with PURL for hp.owl or hp.obo

#### Scenario: Resolve ontology from OLS4

- **WHEN** FetchSpec with resolver="ols" and id="efo" is provided
- **THEN** OLSResolver queries OLS4 API via ols-client and returns FetchPlan with canonical OWL download URL

#### Scenario: Resolve ontology from BioPortal with API key

- **WHEN** FetchSpec with resolver="bioportal" and extras containing acronym="NCIT" is provided
- **THEN** BioPortalResolver uses ontoportal-client to fetch latest submission and returns FetchPlan with Authorization header

#### Scenario: Resolve SKOS thesaurus from direct URL

- **WHEN** FetchSpec with resolver="skos" and extras containing direct URL is provided
- **THEN** SKOSResolver returns FetchPlan with provided URL and RDF content negotiation headers

#### Scenario: Resolve XBRL taxonomy package

- **WHEN** FetchSpec with resolver="xbrl" and extras containing taxonomy ZIP URL is provided
- **THEN** XBRLResolver returns FetchPlan with ZIP URL for download and Arelle validation

#### Scenario: Resolve vocabulary from LOV

- **WHEN** FetchSpec with resolver="lov" and extras.uri is provided
- **THEN** LOVResolver queries LOV API and returns FetchPlan with Turtle download URL

#### Scenario: Resolve ontology from Ontobee PURL

- **WHEN** FetchSpec with resolver="ontobee" and id="hp" is provided
- **THEN** OntobeeResolver constructs PURL and returns FetchPlan with purl.obolibrary.org URL

#### Scenario: Automatic fallback through prefer_source

- **WHEN** prefer_source=["obo", "ols", "ontobee"] and OBOResolver fails
- **THEN** FallbackResolver automatically tries OLSResolver, then OntobeeResolver if OLS also fails

#### Scenario: Fallback logging for observability

- **WHEN** fallback occurs during resolution
- **THEN** each resolver attempt logged with outcome for troubleshooting

### Requirement: Structured Logging for Observability with Sensitive Data Masking

The system SHALL emit structured JSON logs capturing resolver actions, HTTP operations, validation outcomes, fallback attempts, and aggregate metrics with automatic masking of sensitive fields.

#### Scenario: Log resolver plan with timing

- **WHEN** resolver generates FetchPlan
- **THEN** log entry written with fields: timestamp, level="info", stage="resolver", id, resolver_type, plan_url, planning_time_ms

#### Scenario: Log fallback resolver attempts

- **WHEN** FallbackResolver tries multiple resolvers
- **THEN** log entries written for each attempt with resolver_name, attempt_number, outcome, error_message

#### Scenario: Log HTTP download with status and timing

- **WHEN** download completes
- **THEN** log entry written with fields: timestamp, stage="download", url, status_code, etag, sha256, elapsed_time_ms, retries, cache_hit, service

#### Scenario: Log HEAD request validation

- **WHEN** HEAD request executes before GET
- **THEN** log entry written with stage="head", media_type_match, content_length, elapsed_time_ms

#### Scenario: Log validation outcome with subprocess indicator

- **WHEN** validator runs (subprocess or in-process)
- **THEN** log entry written with fields: stage="validation", parser, outcome, error_message, metrics, subprocess=true/false

#### Scenario: Log subprocess validator lifecycle

- **WHEN** pronto or owlready2 validation spawns subprocess
- **THEN** log entries written for subprocess start, completion, and memory isolation benefit

#### Scenario: Mask sensitive headers in logs

- **WHEN** logging API request with Authorization header
- **THEN** log formatter detects sensitive keys and replaces values with "***masked***"

#### Scenario: Log batch summary on completion

- **WHEN** batch pull operation completes
- **THEN** log entry written with fields: stage="summary", total_ontologies, success_count, cached_count, failure_count, fallback_count, total_bandwidth_bytes

### Requirement: Normalized Format Generation with Deterministic Output

The system SHALL optionally generate normalized formats (canonical Turtle from RDF/OWL via RDFLib, OBO Graph JSON from OBO/OWL via Pronto) with deterministic output and stable hashing for cache correctness.

#### Scenario: RDFLib serializes OWL to canonical Turtle

- **WHEN** OWL file parses successfully with RDFLib and normalize_to includes "ttl"
- **THEN** RDFLib canonicalizes graph (sorted prefixes and triples), serializes to normalized/<filename>.ttl

#### Scenario: Canonical TTL produces deterministic hash

- **WHEN** same ontology downloaded twice and normalized
- **THEN** both normalized TTL files have identical SHA-256 hash

#### Scenario: Pronto exports OBO Graph JSON via subprocess

- **WHEN** OBO file parses successfully with Pronto and normalize_to includes "obographs"
- **THEN** Pronto subprocess exports ontology to normalized/<filename>.json in OBO Graph JSON format

#### Scenario: Original file preserved bit-exact

- **WHEN** any ontology is downloaded
- **THEN** exact downloaded bytes saved in original/<filename> without modification

#### Scenario: Normalized hash stored in manifest

- **WHEN** RDFLib produces canonical TTL and computes hash
- **THEN** manifest.json includes normalized_sha256 field for cache validation

## REMOVED Requirements

None. All existing requirements are preserved with enhancements.

## RENAMED Requirements

None. All requirements maintain their original names with "(with <enhancement>)" clarifications in headings where modified.
