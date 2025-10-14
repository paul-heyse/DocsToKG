## 1. Setup and Dependencies

- [x] 1.1 Add dependencies to requirements.txt or pyproject.toml with these EXACT version constraints:

  ```
  bioregistry>=0.10.0
  oaklib>=0.5.0
  ols-client>=0.1.4
  ontoportal-client>=0.0.3
  rdflib>=7.0.0
  pronto>=2.5.0
  owlready2>=0.43
  arelle>=2.20.0
  pystow>=0.5.0
  pooch>=1.7.0
  pyyaml>=6.0
  requests>=2.28.0
  ```

- [x] 1.2 Create module structure under src/DocsToKG/OntologyDownload/ with these EXACT files:
  - `__init__.py` (exports: fetch_one, fetch_all, FetchSpec, FetchResult)
  - `core.py` (dataclasses: FetchSpec, FetchPlan, FetchResult, Manifest; protocol: Resolver)
  - `resolvers.py` (classes: OBOResolver, OLSResolver, BioPortalResolver, SKOSResolver, XBRLResolver, RESOLVERS dict)
  - `download.py` (functions: download_stream, sha256_file, validate_url_security, sanitize_filename)
  - `validators.py` (functions: validate_rdflib, validate_pronto, validate_owlready2, validate_robot, validate_arelle)
  - `config.py` (functions: load_config, validate_config, merge_defaults, get_env_overrides)
  - `logging_config.py` (functions: setup_logging, mask_sensitive_data, generate_correlation_id)
  - `cli.py` (main entry point with argparse/Click: pull, show, validate subcommands)
- [x] 1.3 Create pystow directory structure initialization in core.py using pystow.join('ontology-fetcher') with subdirs: configs/, cache/, logs/, ontologies/; use .mkdir(parents=True, exist_ok=True)
- [x] 1.4 Create example sources.yaml with COMPLETE commented examples:

  ```yaml
  defaults:
    accept_licenses: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
    normalize_to: ["ttl"]
    prefer_source: ["obo", "ols", "bioportal", "direct"]
    http:
      max_retries: 5
      timeout_sec: 30
      backoff_factor: 0.5
      per_host_rate_limit: "4/second"
      max_download_size_gb: 5
    validation:
      skip_reasoning_if_size_mb: 500
      parser_timeout_sec: 60
    logging:
      level: "INFO"
      max_log_size_mb: 100
      retention_days: 30

  ontologies:
    - id: hp
      resolver: obo
      target_formats: [owl, obo]
    - id: efo
      resolver: ols
    - id: ncit
      resolver: bioportal
      extras:
        acronym: NCIT
    - id: eurovoc
      resolver: skos
      extras:
        url: https://op.europa.eu/o/opportal-service/euvoc-download-handler?cellarURI=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Fauthority%2Feurovoc
  ```

## 2. Core Data Models

- [x] 2.1 Implement FetchSpec dataclass with fields: id, resolver, extras, target_formats
- [x] 2.2 Implement FetchPlan dataclass with fields: url, headers, filename_hint, version, license, media_type
- [x] 2.3 Implement FetchResult dataclass with fields: local_path, status, etag, last_modified, sha256
- [x] 2.4 Implement Manifest dataclass for provenance with fields: id, resolver, url, filename, version, license, status, sha256, etag, last_modified, downloaded_at, target_formats
- [x] 2.5 Define Resolver protocol interface with plan() method signature

## 3. Resolver Implementations

- [x] 3.1 Implement OBOResolver using Bioregistry (get_owl_download, get_obo_download, get_rdf_download) with format fallback logic
- [x] 3.2 Implement OLSResolver using ols-client to query OLS4 API and retrieve ontology metadata/download URLs
- [x] 3.3 Implement BioPortalResolver using ontoportal-client to fetch latest submission URLs with API key from pystow
- [x] 3.4 Implement SKOSResolver for direct URL-based SKOS thesauri
- [x] 3.5 Implement XBRLResolver for XBRL taxonomy ZIP packages
- [x] 3.6 Create resolver registry dictionary mapping resolver names to instances
- [x] 3.7 Add unit tests for each resolver with mocked library responses

## 4. Download Manager

- [x] 4.1 Implement sha256_file() utility function for computing file hashes
- [x] 4.2 Implement download_stream() using pooch with stream-to-.part pattern
- [x] 4.3 Add ETag/Last-Modified conditional request headers from previous manifest
- [x] 4.4 Implement HTTP 304 cache hit detection and early return
- [x] 4.5 Add Range request support for partial resume on interruption
- [x] 4.6 Implement exponential backoff retry with configurable max_retries (default 5) and backoff_factor (default 0.5)
- [x] 4.7 Add per-host rate limiting using token bucket algorithm (default 4 req/sec)
- [x] 4.8 Add SHA-256 verification on completed downloads
- [x] 4.9 Add structured logging for HTTP status, elapsed time, retry count, cache hit/miss
- [x] 4.10 Create test fixture: local HTTP server supporting ETag, Last-Modified, Range headers for download tests

## 5. Validation Pipeline

- [x] 5.1 Implement RDFLib validator: Graph.parse() with triple count recording and error capture
- [x] 5.2 Add RDFLib Turtle normalization (serialize to normalized/)
- [x] 5.3 Implement Pronto validator: Ontology() load with term count recording and error capture
- [x] 5.4 Add Pronto OBO Graph JSON export (to normalized/)
- [x] 5.5 Implement Owlready2 validator: get_ontology().load() with error capture
- [x] 5.6 Add optional ROBOT integration: runtime detection via shutil.which('robot')
- [x] 5.7 Implement ROBOT convert subprocess call for format conversions
- [x] 5.8 Implement ROBOT report subprocess call for SPARQL QC checks
- [x] 5.9 Implement Arelle validator for XBRL taxonomy packages with JSON result export
- [x] 5.10 Write validation results to validation/<parser>_parse.json
- [x] 5.11 Create mini-ontology test fixtures (5-10 terms) in Turtle, OBO, OWL for parser tests

## 6. Storage and Manifests

- [x] 6.1 Implement directory creation for ontologies/<id>/<version>/{original,normalized,validation}
- [x] 6.2 Implement manifest.json write with comprehensive provenance fields
- [x] 6.3 Implement manifest.json read for cache invalidation decisions (ETag/Last-Modified)
- [x] 6.4 Add version resolution logic (use provided version or generate from timestamp)
- [x] 6.5 Implement safe ZIP extraction with path validation for XBRL taxonomies
- [x] 6.6 Add manifest validation and schema checks

## 7. Configuration

- [x] 7.1 Implement YAML parser for sources.yaml with schema validation
- [x] 7.2 Parse defaults section: accept_licenses, normalize_to, prefer_source, http settings
- [x] 7.3 Parse ontologies list into FetchSpec objects with per-ontology overrides
- [x] 7.4 Implement license allowlist checking with fail-closed behavior
- [x] 7.5 Add configuration validation command (ontofetch config validate)
- [x] 7.6 Document configuration schema with examples

## 8. Orchestration

- [x] 8.1 Implement fetch_one() function orchestrating resolver plan → download → validation → manifest
- [x] 8.2 Add error handling and structured logging at each pipeline stage
- [x] 8.3 Implement batch fetch_all() for multiple ontologies from sources.yaml
- [x] 8.4 Add progress reporting for batch operations
- [x] 8.5 Implement graceful failure recording (continue on error, log failures)

## 9. CLI

- [x] 9.1 Set up CLI framework (argparse or Click) with ontofetch entry point
- [x] 9.2 Implement `ontofetch pull` subcommand with --spec FILE and positional ID arguments
- [x] 9.3 Add --force flag to bypass cache and force redownload
- [x] 9.4 Implement `ontofetch show ID` subcommand to display manifest
- [x] 9.5 Add --versions flag to list all downloaded versions
- [x] 9.6 Implement `ontofetch validate ID[@VERSION]` subcommand to re-run validators
- [x] 9.7 Add validator selection flags: --robot, --rdflib, --pronto, --owlready2, --arelle
- [x] 9.8 Add --json flag for machine-readable output across all subcommands
- [x] 9.9 Implement human-friendly formatted output (tables, summaries)
- [x] 9.10 Configure structured JSON logging with custom formatter

## 10. Testing

- [x] 10.1 Write resolver unit tests with mocked library responses (one test suite per resolver)
- [x] 10.2 Write download manager tests using local HTTP server fixture
- [x] 10.3 Test ETag/Last-Modified conditional requests (304 response)
- [x] 10.4 Test Range request resume on partial download
- [x] 10.5 Test exponential backoff retry on transient failures (500, 503)
- [x] 10.6 Test per-host rate limiting enforcement
- [x] 10.7 Write parser tests using mini-ontology fixtures (RDFLib, Pronto, Owlready2)
- [x] 10.8 Test ROBOT integration with optional runtime detection
- [x] 10.9 Test Arelle XBRL validation with minimal taxonomy package
- [x] 10.10 Write integration smoke test: fetch PATO (OBO), BFO (OLS), verify manifests and validation outputs
- [x] 10.11 Add configuration validation tests (schema checks, invalid YAML)
- [x] 10.12 Test safe ZIP extraction with malicious paths (zip-slip guard)

## 11. Documentation

- [x] 11.1 Write installation instructions with dependency installation command
- [x] 11.2 Document pystow configuration: env vars, credential paths for BioPortal API keys
- [x] 11.3 Create sources.yaml schema documentation with field descriptions
- [x] 11.4 Write CLI usage guide with examples for pull/show/validate subcommands
- [x] 11.5 Document resolver-specific requirements (BioPortal API key, ROBOT Java runtime)
- [x] 11.6 Add troubleshooting guide for common issues (rate limits, auth failures, validation errors)
- [x] 11.7 Document storage layout and manifest schema
- [x] 11.8 Add example workflows: batch download, incremental update, validation-only re-runs

## 12. Integration and Polish

- [x] 12.1 Run integration smoke tests with real API calls (HP from OBO, EFO from OLS)
- [x] 12.2 Test end-to-end workflow with example sources.yaml
- [x] 12.3 Verify structured logs contain all required fields (resolver, status, timing, errors)
- [x] 12.4 Test force-refresh with --force flag on cached ontologies
- [x] 12.5 Verify SHA-256 checksums recorded correctly in manifests
- [x] 12.6 Test multi-version handling (download same ontology multiple times, check version directories)
- [x] 12.7 Verify normalized/ directory contains Turtle and OBO Graph JSON where applicable
- [x] 12.8 Run validation on large ontology (NCIT, ChEBI) to check memory usage
- [x] 12.9 Test CLI --json output is valid JSON and includes expected fields
- [x] 12.10 Add performance benchmarks for download/validation on representative ontologies

## 13. Error Handling and Recovery

- [x] 13.1 In download.py, wrap download_stream() with try/except for requests.exceptions (ConnectionError, Timeout, HTTPError) and implement retry logic using tenacity library or custom decorator
- [x] 13.2 Add specific exception classes: OntologyDownloadError, ResolverError, ValidationError, ConfigurationError in core.py with clear error messages
- [x] 13.3 In resolvers.py, handle HTTP 401/403 for BioPortal/OLS with error message including credential config path (e.g., "Configure API key at ~/.config/pystow/ontology-fetcher/bioportal_api_key.txt")
- [x] 13.4 In download.py, catch OSError during file write and check for "No space left on device"; log error with space requirements estimate, cleanup .part file, raise with exit code 1
- [x] 13.5 In validators.py, wrap each parser call (RDFLib, Pronto, Owlready2) in try/except for ParserError, MemoryError; write {"ok": false, "error": "<details>"} to validation JSON; continue with remaining validators
- [x] 13.6 In config.py, catch FileNotFoundError for missing sources.yaml; exit with code 2 and stderr message "Configuration file not found: <path>"
- [x] 13.7 In download.py, implement checksum verification: after download, if sha256_file(path) != expected_hash (from manifest or registry), log error with both hashes, delete file, retry as cache miss
- [x] 13.8 Add timeout handling: wrap resolver API calls with requests.get(..., timeout=30); catch Timeout exception, log "API timeout after 30s", retry with exponential backoff
- [x] 13.9 In validators.py, catch MemoryError during Owlready2 parsing of large files; log "Memory limit exceeded parsing <id>. Consider skipping reasoning"; write error to validation JSON without crashing
- [ ] 13.10 Add permission error handling: catch PermissionError when creating pystow directories; log "Permission denied writing to <path>. Set PYSTOW_HOME env var"; exit code 1

## 14. Security Implementation

- [x] 14.1 In download.py, create validate_url_security(url: str) function that:
  - Parses URL with urllib.parse
  - Checks scheme is https:// (warn if http://, upgrade to https://)
  - Extracts hostname, resolves to IP with socket.gethostbyname()
  - Rejects private IP ranges: 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, ::1, fc00::/7
  - Returns validated URL or raises SecurityError
- [x] 14.2 In download.py, enable TLS certificate verification: use requests.get(url, verify=True) always; catch requests.exceptions.SSLError with clear message about invalid certificate
- [x] 14.3 In download.py, create sanitize_filename(filename: str) -> str that:
  - Removes path separators (os.sep, os.altsep, '/', '\\')
  - Removes leading dots and special chars
  - Limits length to 255 chars
  - Logs warning if sanitization occurred: "Sanitized unsafe filename: <original> -> <safe>"
- [x] 14.4 In logging_config.py, create mask_sensitive_data(log_record: dict) -> dict that:
  - Searches for 'Authorization', 'apikey', 'api_key', 'password', 'token' keys (case-insensitive)
  - Replaces values with "***masked***"
  - Returns sanitized record
- [x] 14.5 In download.py, enforce max file size: check Content-Length header before download; if > config.http.max_download_size_gb * 1024³, log error "File exceeds max_download_size: <size> > <limit>", skip download
- [ ] 14.6 In validators.py for XBRL, before ZIP extraction:
  - Call zipfile.is_zipfile() to verify
  - Check compressed vs uncompressed size ratio (<10x)
  - Validate all member paths don't contain ".." or start with "/"
  - Extract to temp dir first, then move to final location
- [x] 14.7 Call validate_url_security() before every download in download_stream()
- [x] 14.8 Apply mask_sensitive_data() to all log records in logging_config.py formatter

## 15. Performance and Resource Management

- [ ] 15.1 In config.py, add performance-related config schema fields:

  ```python
  http:
    timeout_sec: int = 30  # API calls
    download_timeout_sec: int = 300  # file downloads
    concurrent_downloads: int = 1  # parallel limit
  validation:
    parser_timeout_sec: int = 60
    max_memory_mb: int = 2048
    skip_reasoning_if_size_mb: int = 500
  ```

- [ ] 15.2 In download.py, use requests.get(url, stream=True, timeout=config.http.download_timeout_sec); iterate with iter_content(chunk_size=1024*1024); log progress every 10%
- [ ] 15.3 In orchestration (core.py fetch_all()), implement concurrency:
  - Use ThreadPoolExecutor(max_workers=config.http.concurrent_downloads)
  - Or asyncio with semaphore for async HTTP
  - Ensure rate limiting still applies per-host
- [ ] 15.4 In validators.py, before running Owlready2 reasoning:
  - Check file size with path.stat().st_size
  - If > config.validation.skip_reasoning_if_size_mb * 1024², log "Skipping reasoning for large file", set reasoning=False
- [ ] 15.5 In validators.py, implement parser timeout using:
  - On Unix: signal.alarm(config.validation.parser_timeout_sec) before parse, signal.alarm(0) after, handle signal.SIGALRM
  - On Windows: threading.Timer to interrupt after timeout
  - Catch timeout, log "Parser timeout after <N>s", write error to validation JSON
- [ ] 15.6 Add memory profiling imports: import psutil; log current process memory usage before/after large operations if --debug flag set

## 16. Enhanced Configuration

- [ ] 16.1 In config.py, define complete ConfigSchema dataclass with all fields:

  ```python
  @dataclass
  class HTTPConfig:
      max_retries: int = 5
      timeout_sec: int = 30
      download_timeout_sec: int = 300
      backoff_factor: float = 0.5
      per_host_rate_limit: str = "4/second"
      max_download_size_gb: float = 5.0
      concurrent_downloads: int = 1

  @dataclass
  class ValidationConfig:
      parser_timeout_sec: int = 60
      max_memory_mb: int = 2048
      skip_reasoning_if_size_mb: int = 500

  @dataclass
  class LoggingConfig:
      level: str = "INFO"
      max_log_size_mb: int = 100
      retention_days: int = 30

  @dataclass
  class DefaultsConfig:
      accept_licenses: list[str] = field(default_factory=list)
      normalize_to: list[str] = field(default_factory=lambda: ["ttl"])
      prefer_source: list[str] = field(default_factory=lambda: ["obo", "ols", "bioportal"])
      http: HTTPConfig = field(default_factory=HTTPConfig)
      validation: ValidationConfig = field(default_factory=ValidationConfig)
      logging: LoggingConfig = field(default_factory=LoggingConfig)
  ```

- [x] 16.2 In config.py, implement get_env_overrides() that checks for ONTOFETCH_* env vars:
  - ONTOFETCH_MAX_RETRIES -> defaults.http.max_retries
  - ONTOFETCH_TIMEOUT_SEC -> defaults.http.timeout_sec
  - ONTOFETCH_LOG_LEVEL -> defaults.logging.level
  - Log "Config overridden by env var: <key>=<value>" for each override
- [ ] 16.3 In config.py validate_config(), check:
  - defaults.http.max_retries is int >= 0
  - defaults.http.timeout_sec is int > 0
  - defaults.logging.level in ["DEBUG", "INFO", "WARNING", "ERROR"]
  - Each ontology has required fields: id, resolver
  - BioPortal resolver has extras.acronym
  - Raise ValueError with descriptive message for violations
- [ ] 16.4 In config.py, implement merge_defaults(ontology_spec, defaults) that:
  - Copies defaults as base
  - Overlays ontology-specific extras.timeout_sec, etc.
  - Returns merged FetchSpec
- [ ] 16.5 Add --config-validate subcommand in CLI that loads sources.yaml, runs validate_config(), prints "Configuration valid" or errors

## 17. Enhanced Logging System

- [ ] 17.1 In logging_config.py, create setup_logging(level: str, log_dir: Path) that:
  - Creates RotatingFileHandler for logs/ontofetch_<date>.jsonl with maxBytes=config.logging.max_log_size_mb * 1024²
  - Creates console StreamHandler for human-readable output
  - Sets file handler to JSON formatter, console to human-readable
  - Sets log level from config (DEBUG/INFO/WARNING/ERROR)
- [ ] 17.2 In logging_config.py, create JSONFormatter(logging.Formatter):

  ```python
  def format(self, record: logging.LogRecord) -> str:
      log_obj = {
          "timestamp": datetime.utcnow().isoformat() + "Z",
          "level": record.levelname,
          "message": record.getMessage(),
          "correlation_id": getattr(record, 'correlation_id', None),
          "ontology_id": getattr(record, 'ontology_id', None),
          "stage": getattr(record, 'stage', None),
          **record.__dict__.get('extra_fields', {})
      }
      log_obj = mask_sensitive_data(log_obj)
      return json.dumps(log_obj)
  ```

- [ ] 17.3 In logging_config.py, implement generate_correlation_id() -> str using uuid.uuid4().hex[:12]
- [ ] 17.4 In orchestration fetch_all(), generate correlation_id once, attach to all log records via logging.LoggerAdapter or extra={'correlation_id': ...}
- [ ] 17.5 In logging_config.py, implement log rotation cleanup: on startup, scan logs/ for files older than config.logging.retention_days, compress with gzip, delete if compressed >retention_days
- [ ] 17.6 Ensure all log calls include structured fields: logger.info("message", extra={'stage': 'download', 'ontology_id': id, 'elapsed_ms': elapsed})

## 18. Cross-Platform Compatibility

- [ ] 18.1 In core.py or cli.py startup, add Python version check:

  ```python
  import sys
  if sys.version_info < (3, 9):
      print("Error: Python 3.9+ required", file=sys.stderr)
      sys.exit(1)
  ```

- [ ] 18.2 Use pathlib.Path throughout for all path operations (no os.path.join or string concatenation)
- [ ] 18.3 In validators.py, implement platform-specific timeout:

  ```python
  import platform
  if platform.system() in ('Linux', 'Darwin'):
      # Use signal.alarm() for Unix
      import signal
      signal.signal(signal.SIGALRM, timeout_handler)
      signal.alarm(timeout_sec)
  else:
      # Use threading.Timer for Windows
      import threading
      timer = threading.Timer(timeout_sec, timeout_handler)
      timer.start()
  ```

- [ ] 18.4 In requirements.txt, specify minimum versions explicitly:

  ```
  rdflib>=7.0.0,<8.0.0
  pronto>=2.5.0,<3.0.0
  pystow>=0.5.0
  pooch>=1.7.0
  requests>=2.28.0,<3.0.0
  pyyaml>=6.0,<7.0
  ```

- [ ] 18.5 Test on all three platforms (Linux, macOS, Windows) in CI if available, or document platform-specific quirks in docs/

## 19. User Documentation and Help

- [ ] 19.1 In cli.py, add comprehensive --help text for main command:

  ```
  Usage: ontofetch [OPTIONS] COMMAND [ARGS]...

  Ontology downloader for DocsToKG supporting OBO, OLS, BioPortal, SKOS, XBRL sources.

  Options:
    --log-level [DEBUG|INFO|WARNING|ERROR]  Set logging verbosity
    --help                                  Show this message and exit

  Commands:
    pull      Download ontologies
    show      Display ontology metadata
    validate  Re-run validation on downloaded ontologies
    init      Create example sources.yaml configuration
  ```

- [ ] 19.2 Add --help for each subcommand with examples:

  ```
  ontofetch pull --help
    Options:
      --spec FILE              Path to sources.yaml (default: configs/sources.yaml)
      --force                  Force redownload bypassing cache
      --resolver TEXT          Resolver type for single ontology
      --target-formats LIST    Comma-separated formats (e.g., owl,obo)

    Examples:
      ontofetch pull --spec sources.yaml          # Batch download from config
      ontofetch pull hp --resolver obo            # Download single ontology
      ontofetch pull ncit --resolver bioportal --force  # Force refresh
  ```

- [ ] 19.3 Enhance error messages with remediation in all exception handlers:
  - BioPortal 401: "BioPortal API key required. Get key from <https://bioportal.bioontology.org/account>, then configure: echo 'YOUR_KEY' > ~/.config/pystow/ontology-fetcher/bioportal_api_key.txt"
  - Permission error: "Cannot write to <path>. Try: export PYSTOW_HOME=/path/to/writable/directory"
  - Network error: "Network connection failed. Check internet connection and firewall settings."
- [ ] 19.4 Create `ontofetch init` subcommand that writes example sources.yaml to current directory with extensive inline comments explaining each field
- [ ] 19.5 In README.md or docs/, add troubleshooting section:
  - "Rate limit exceeded" -> increase backoff_factor or reduce concurrent_downloads
  - "Memory error on large ontology" -> increase max_memory_mb or skip reasoning
  - "ROBOT not found" -> install with: brew install robot (macOS) or download from <https://github.com/ontodev/robot/releases>
- [ ] 19.6 Add configuration schema documentation showing all fields, defaults, and examples for each resolver type
- [ ] 19.7 For YAML parse errors, use python-yaml's line number from exception: "Error in sources.yaml line <N>: <message>"

## 20. Additional Testing for New Requirements

- [ ] 20.1 Test error handling: write tests that mock ConnectionError, Timeout, HTTP 503, verify retry logic and exponential backoff timing
- [ ] 20.2 Test security: attempt downloads from http://, private IPs (127.0.0.1), verify rejection; test filename sanitization with "../evil.owl"
- [ ] 20.3 Test configuration: load sources.yaml with invalid values (max_retries="many"), verify ValueError with line number
- [ ] 20.4 Test environment overrides: set ONTOFETCH_MAX_RETRIES=10, verify config uses 10 not file value
- [ ] 20.5 Test logging: verify mask_sensitive_data() replaces "Authorization: apikey ABC123" with "***masked***"
- [ ] 20.6 Test correlation IDs: run batch with 3 ontologies, verify all log entries for one ontology share same correlation_id
- [ ] 20.7 Test platform compatibility: run on Windows VM if available, verify pathlib paths work, threading.Timer timeout works
- [ ] 20.8 Test CLI help: capture stdout from `ontofetch --help`, verify contains all subcommands and options
- [ ] 20.9 Test concurrent downloads: configure concurrent_downloads=3, batch 10 ontologies, verify only 3 run in parallel
- [ ] 20.10 Test performance limits: mock file with Content-Length > max_download_size_gb, verify rejection before download starts
