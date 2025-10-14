## ADDED Requirements

### Requirement: Source-Agnostic Resolver Registry

The system SHALL provide a resolver registry that maps ontology source identifiers (OBO, OLS, BioPortal, SKOS, XBRL) to concrete resolver implementations that can locate downloadable artifacts.

#### Scenario: Resolve OBO Foundry ontology by prefix

- **WHEN** a FetchSpec with resolver="obo" and id="hp" is provided
- **THEN** the OBOResolver uses Bioregistry functions to return a FetchPlan with the PURL for hp.owl or hp.obo based on target_formats preference

#### Scenario: Resolve ontology from OLS4

- **WHEN** a FetchSpec with resolver="ols" and id="efo" is provided
- **THEN** the OLSResolver queries the OLS4 API via ols-client and returns a FetchPlan with the canonical OWL download URL and version metadata

#### Scenario: Resolve ontology from BioPortal with API key

- **WHEN** a FetchSpec with resolver="bioportal" and extras containing acronym="NCIT" is provided
- **THEN** the BioPortalResolver uses ontoportal-client to fetch the latest submission and returns a FetchPlan with the download URL and Authorization header containing the API key from pystow

#### Scenario: Resolve SKOS thesaurus from direct URL

- **WHEN** a FetchSpec with resolver="skos" and extras containing a direct URL is provided
- **THEN** the SKOSResolver returns a FetchPlan with the provided URL and appropriate headers for RDF content negotiation

#### Scenario: Resolve XBRL taxonomy package

- **WHEN** a FetchSpec with resolver="xbrl" and extras containing a taxonomy ZIP URL is provided
- **THEN** the XBRLResolver returns a FetchPlan with the ZIP URL for subsequent download and Arelle validation

### Requirement: Robust HTTP Download with Caching

The system SHALL implement robust HTTP download logic with conditional requests, resume support, checksums, retry, and rate limiting using pooch as the underlying cache and fetch mechanism.

#### Scenario: Conditional GET with ETag returns 304 Not Modified

- **WHEN** a previous manifest exists with an ETag and the server responds with HTTP 304
- **THEN** the download manager returns a FetchResult with status='cached' and the existing local_path without re-downloading

#### Scenario: Resume partial download with Range header

- **WHEN** a .part file exists from an interrupted download and the server supports Range requests
- **THEN** the download manager sends a Range header starting from the partial file size and appends the remaining bytes

#### Scenario: SHA-256 verification after complete download

- **WHEN** a file download completes successfully
- **THEN** the download manager computes the SHA-256 hash and records it in the FetchResult

#### Scenario: Exponential backoff retry on transient failure

- **WHEN** the server responds with HTTP 503 or a network timeout occurs
- **THEN** the download manager retries up to 5 times with exponential backoff (0.5s base factor)

#### Scenario: Per-host rate limiting enforces 4 requests per second

- **WHEN** multiple downloads are requested from the same host
- **THEN** the download manager enforces a token bucket rate limit (default 4 req/sec) to avoid overwhelming the server

### Requirement: Multi-Parser Validation Pipeline

The system SHALL validate downloaded ontologies using multiple parsers appropriate to their format (RDFLib for RDF/OWL/SKOS, Pronto for OBO/OWL, Owlready2 for OWL reasoning, ROBOT for conversions/QC, Arelle for XBRL) and record validation results in structured JSON files.

#### Scenario: RDFLib parses Turtle ontology successfully

- **WHEN** a Turtle file is downloaded and RDFLib validation runs
- **THEN** the validator writes validation/rdflib_parse.json with {"ok": true, "triples": N} where N is the triple count

#### Scenario: RDFLib fails to parse malformed RDF

- **WHEN** a downloaded file claims to be RDF but contains syntax errors
- **THEN** the validator writes validation/rdflib_parse.json with {"ok": false, "error": "<message>"}

#### Scenario: Pronto parses OBO file and counts terms

- **WHEN** an OBO file is downloaded and Pronto validation runs
- **THEN** the validator writes validation/pronto_parse.json with {"ok": true, "terms": N} where N is the number of terms

#### Scenario: ROBOT integration skipped when not installed

- **WHEN** ROBOT validation is requested but `robot` command is not found via shutil.which()
- **THEN** the validator logs a warning and skips ROBOT steps without failing the entire validation

#### Scenario: Arelle validates XBRL taxonomy package

- **WHEN** an XBRL ZIP file is downloaded and Arelle validation runs
- **THEN** the validator extracts the taxonomy, runs Arelle validation, and writes validation/arelle_validation.json with validation results

### Requirement: Normalized Format Generation

The system SHALL optionally generate normalized formats (Turtle from RDF/OWL via RDFLib, OBO Graph JSON from OBO/OWL via Pronto) and store them in the normalized/ subdirectory while preserving original files in original/.

#### Scenario: RDFLib serializes OWL to normalized Turtle

- **WHEN** an OWL file parses successfully with RDFLib and normalize_to includes "ttl"
- **THEN** RDFLib serializes the graph to normalized/<filename>.ttl in canonical Turtle format

#### Scenario: Pronto exports OBO Graph JSON

- **WHEN** an OBO file parses successfully with Pronto and normalize_to includes "obographs"
- **THEN** Pronto exports the ontology to normalized/<filename>.json in OBO Graph JSON format

#### Scenario: Original file preserved bit-exact

- **WHEN** any ontology is downloaded
- **THEN** the exact downloaded bytes are saved in original/<filename> without modification

### Requirement: Comprehensive Provenance Manifests

The system SHALL record comprehensive provenance metadata for each downloaded ontology in a manifest.json file including source URL, resolver, version, license, ETag, Last-Modified, SHA-256, timestamps, and validation status.

#### Scenario: Manifest records all provenance fields

- **WHEN** an ontology is successfully downloaded and validated
- **THEN** manifest.json contains fields: id, resolver, url, filename, version, license, status, sha256, etag, last_modified, downloaded_at, target_formats

#### Scenario: Manifest used for cache invalidation decision

- **WHEN** a subsequent download request occurs for the same ontology
- **THEN** the system reads the previous manifest.json and uses etag/last_modified values in conditional request headers

#### Scenario: Manifest records validation outcomes

- **WHEN** validation pipeline runs on a downloaded ontology
- **THEN** manifest.json includes references or summaries of validation results from validation/ directory

### Requirement: Declarative YAML Configuration

The system SHALL support configuration via sources.yaml with a defaults section for global settings (license allowlists, format preferences, HTTP parameters) and per-ontology specifications (id, resolver, target formats, resolver-specific extras).

#### Scenario: Parse defaults section with license allowlist

- **WHEN** sources.yaml contains defaults.accept_licenses=["CC-BY-4.0", "CC0-1.0"]
- **THEN** the system loads these licenses into a global allowlist used for all ontologies unless overridden

#### Scenario: Parse ontology list into FetchSpec objects

- **WHEN** sources.yaml contains an ontologies list with id, resolver, and target_formats per entry
- **THEN** the configuration parser creates a FetchSpec object for each ontology with merged defaults

#### Scenario: Per-ontology resolver override

- **WHEN** an ontology specifies resolver="bioportal" and extras.acronym="NCIT"
- **THEN** the FetchSpec includes these resolver-specific parameters for BioPortalResolver

#### Scenario: Configuration validation detects invalid YAML

- **WHEN** sources.yaml contains syntax errors or missing required fields
- **THEN** the configuration loader raises a validation error with clear message indicating the issue

### Requirement: License Compliance Enforcement

The system SHALL check ontology licenses against a configurable allowlist and fail closed (refuse to download) when encountering restricted licenses without explicit acceptance.

#### Scenario: License allowlist permits download

- **WHEN** an ontology has license="CC0-1.0" and defaults.accept_licenses includes "CC0-1.0"
- **THEN** the download proceeds normally

#### Scenario: License not in allowlist blocks download

- **WHEN** an ontology has license="Proprietary" and it is not in defaults.accept_licenses
- **THEN** the system logs an error and skips the download with clear message indicating license restriction

#### Scenario: Missing license treated as restricted

- **WHEN** an ontology resolver returns license=None
- **THEN** the system logs a warning and treats it as restricted unless a bypass flag is configured

### Requirement: pystow-Based Storage Management

The system SHALL use pystow to manage data directories at ~/.data/ontology-fetcher (or overridden via env vars) with subdirectories for configs/, cache/, logs/, and ontologies/<id>/<version>/{original,normalized,validation}.

#### Scenario: pystow creates directory structure on first run

- **WHEN** ontofetch is invoked for the first time
- **THEN** pystow.join('ontology-fetcher') creates ~/.data/ontology-fetcher with necessary subdirectories

#### Scenario: Version-specific subdirectory created

- **WHEN** an ontology with id="hp" and version="2024-10-31" is downloaded
- **THEN** the system creates ontologies/hp/2024-10-31/{original,normalized,validation}

#### Scenario: pystow env var overrides base path

- **WHEN** environment variable PYSTOW_HOME is set to /custom/path
- **THEN** ontology data is stored under /custom/path/ontology-fetcher instead of ~/.data

### Requirement: CLI with Pull/Show/Validate Operations

The system SHALL provide a command-line interface (ontofetch) with subcommands for downloading (pull), inspecting (show), and validating (validate) ontologies with both batch and single-ontology modes.

#### Scenario: Pull ontologies from sources.yaml

- **WHEN** user runs `ontofetch pull --spec sources.yaml`
- **THEN** the CLI reads sources.yaml, iterates through ontology list, and downloads each one with progress reporting

#### Scenario: Pull single ontology by ID

- **WHEN** user runs `ontofetch pull hp --resolver obo --target-formats owl,obo`
- **THEN** the CLI constructs a FetchSpec for HP and downloads it using OBOResolver with format fallback

#### Scenario: Force refresh bypasses cache

- **WHEN** user runs `ontofetch pull hp --force`
- **THEN** the CLI ignores ETag/Last-Modified from previous manifest and forces a fresh download

#### Scenario: Show ontology manifest

- **WHEN** user runs `ontofetch show hp`
- **THEN** the CLI reads ontologies/hp/<latest-version>/manifest.json and displays provenance fields in human-readable format

#### Scenario: List all versions of ontology

- **WHEN** user runs `ontofetch show hp --versions`
- **THEN** the CLI lists all version directories under ontologies/hp/ with timestamps and sizes

#### Scenario: Re-run validation on existing ontology

- **WHEN** user runs `ontofetch validate hp@2024-10-31 --rdflib --pronto`
- **THEN** the CLI locates ontologies/hp/2024-10-31/original/<file> and re-runs RDFLib and Pronto validators, updating validation/ outputs

#### Scenario: Machine-readable JSON output

- **WHEN** user runs `ontofetch show hp --json`
- **THEN** the CLI outputs the complete manifest.json content as valid JSON to stdout

### Requirement: Structured Logging for Observability

The system SHALL emit structured JSON logs capturing resolver actions, HTTP operations, validation outcomes, and aggregate metrics with fields for source, status, timing, errors, and retry counts.

#### Scenario: Log resolver plan with timing

- **WHEN** a resolver generates a FetchPlan
- **THEN** a log entry is written with fields: timestamp, level="info", stage="resolver", id, resolver_type, plan_url, planning_time_ms

#### Scenario: Log HTTP download with status and timing

- **WHEN** a download completes
- **THEN** a log entry is written with fields: timestamp, level="info", stage="download", url, status_code, etag, sha256, elapsed_time_ms, retries, cache_hit

#### Scenario: Log validation outcome

- **WHEN** a validator runs
- **THEN** a log entry is written with fields: timestamp, level="info"|"error", stage="validation", parser, outcome="success"|"failure", error_message, metrics (triples, terms)

#### Scenario: Log batch summary on completion

- **WHEN** a batch pull operation completes
- **THEN** a log entry is written with fields: timestamp, level="info", stage="summary", total_ontologies, success_count, cached_count, failure_count, total_bandwidth_bytes, total_elapsed_time_ms

### Requirement: Safe ZIP Extraction for XBRL

The system SHALL safely extract XBRL taxonomy ZIP files with path validation to prevent zip-slip attacks, ensuring extracted files remain within the target directory.

#### Scenario: Extract XBRL ZIP with valid paths

- **WHEN** an XBRL ZIP file contains taxonomy files with paths like "schemas/ifrs.xsd"
- **THEN** the system extracts them to ontologies/<id>/<version>/original/schemas/ifrs.xsd within the designated directory

#### Scenario: Reject ZIP with path traversal attack

- **WHEN** an XBRL ZIP file contains a member with path "../../etc/passwd"
- **THEN** the system detects the path traversal, logs an error, and refuses to extract the file

#### Scenario: Validate ZIP before extraction

- **WHEN** a downloaded XBRL file is claimed to be a ZIP
- **THEN** the system calls zipfile.is_zipfile() before extraction and fails safely if not a valid ZIP

### Requirement: Batch Operation with Graceful Failure Handling

The system SHALL support batch operations over multiple ontologies with graceful failure handling that continues processing remaining items and records failures in structured logs without aborting the entire batch.

#### Scenario: Batch download continues after single failure

- **WHEN** a batch pull includes 5 ontologies and ontology #2 fails due to network error
- **THEN** the system logs the failure for #2, continues with #3-5, and reports aggregate success/failure counts at completion

#### Scenario: Failed items reported in summary

- **WHEN** a batch pull completes with some failures
- **THEN** the CLI summary output lists failed ontology IDs with error reasons

#### Scenario: Partial results usable despite failures

- **WHEN** some ontologies in a batch download successfully while others fail
- **THEN** manifests and validation outputs are written for successful items and can be used independently

### Requirement: Optional ROBOT Integration for Conversions and QC

The system SHALL optionally integrate ROBOT (when installed and detected via runtime check) for ontology format conversions and SPARQL-based quality control reports without requiring ROBOT as a hard dependency.

#### Scenario: ROBOT detected and used for conversion

- **WHEN** ROBOT is installed and `ontofetch validate hp --robot` is run
- **THEN** the system calls `robot convert -i hp.owl -o hp.obo` via subprocess and stores output in normalized/

#### Scenario: ROBOT used for QC report

- **WHEN** ROBOT is installed and `ontofetch validate hp --robot` is run
- **THEN** the system calls `robot report -i hp.owl -o validation/robot_report.tsv` and captures SPARQL check results

#### Scenario: ROBOT not installed, validation continues

- **WHEN** ROBOT is not found via shutil.which('robot') and `ontofetch validate hp --robot` is run
- **THEN** the system logs "ROBOT not found, skipping conversion/QC steps" and proceeds with other validators without error

#### Scenario: ROBOT subprocess error captured

- **WHEN** ROBOT command fails with non-zero exit code
- **THEN** the system captures stdout/stderr, logs the error, and writes validation/robot_error.txt without crashing

