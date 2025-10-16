# 1. Module: ontology_download

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.ontology_download``.

## 1. Overview

Unified ontology downloader orchestrating settings, retries, and logging.

## 2. Functions

### `retry_with_backoff(func)`

Execute ``func`` with exponential backoff until it succeeds.

Args:
func: Zero-argument callable to invoke.
retryable: Predicate returning ``True`` when the raised exception should
trigger another attempt.
max_attempts: Maximum number of attempts including the initial call.
backoff_base: Base delay in seconds used for the exponential schedule.
jitter: Maximum random jitter (uniform) added to each delay.
callback: Optional hook invoked before sleeping with
``(attempt_number, error, delay_seconds)``.
sleep: Sleep function, overridable for deterministic tests.

Returns:
The result produced by ``func`` when it succeeds.

Raises:
ValueError: If ``max_attempts`` is less than one.
BaseException: Re-raises the last exception from ``func`` when retries
are exhausted or the predicate indicates it is not retryable.

### `sanitize_filename(filename)`

Sanitize filenames to prevent directory traversal and unsafe characters.

Args:
filename: Candidate filename provided by an upstream service.

Returns:
Safe filename compatible with local filesystem storage.

### `generate_correlation_id()`

Create a short-lived identifier that links related log entries.

Args:
None.

Returns:
Twelve-character hexadecimal correlation identifier.

Raises:
None.

### `mask_sensitive_data(payload)`

Remove secrets from structured payloads prior to logging.

Args:
payload: Structured data that may contain sensitive values.

Returns:
Copy of ``payload`` with common secret fields masked.

### `ensure_python_version()`

Ensure the interpreter meets the minimum supported Python version.

Args:
None.

Returns:
None.

Raises:
SystemExit: If the current interpreter version is below the minimum.

### `_coerce_sequence(value)`

Normalize configuration entries into a list of strings.

Args:
value: Input value that may be ``None``, a string, or an iterable of strings.

Returns:
List of strings suitable for downstream configuration processing.

### `parse_rate_limit_to_rps(limit_str)`

Convert a rate limit expression into requests-per-second.

The accepted format is ``<value>/<unit>`` where ``unit`` is one of
``second``, ``minute``, or ``hour`` (including short aliases such as ``s``
or ``min``). The function returns ``None`` when the value cannot be parsed.

Args:
limit_str: Rate limit expression in ``value/unit`` form.

Returns:
Optional[float]: Requests-per-second value when parsing succeeds.

Raises:
ValueError: If the numeric component cannot be converted to ``float``.

### `get_env_overrides()`

Return raw environment override values for backwards compatibility.

Args:
None

Returns:
Mapping of configuration field name to override value sourced from the environment.

### `_apply_env_overrides(defaults)`

Apply environment variable overrides to default configuration.

Args:
defaults: Defaults configuration object that will be mutated in place.

Returns:
None

### `_make_fetch_spec(ontology_id, resolver, extras, target_formats)`

Instantiate a FetchSpec from raw YAML fields.

Args:
ontology_id: Identifier of the ontology being configured.
resolver: Name of the resolver responsible for fetching content.
extras: Additional resolver-specific configuration parameters.
target_formats: Desired output formats for the ontology artefact.

Returns:
Fully initialised ``FetchSpec`` object ready for execution.

### `merge_defaults(ontology_spec, defaults)`

Merge an ontology specification with resolved default settings.

Args:
ontology_spec: Raw ontology specification mapping loaded from YAML.
defaults: Optional resolved defaults to merge with the specification.

Returns:
FetchSpec instance describing the fully-merged ontology configuration.

Raises:
ConfigError: If required fields are missing or invalid in the specification.

### `build_resolved_config(raw_config)`

Construct fully-resolved configuration from raw YAML contents.

Args:
raw_config: Parsed YAML mapping defining defaults and ontologies.

Returns:
ResolvedConfig combining defaults and individual fetch specifications.

Raises:
ConfigError: If validation fails or required sections are missing.

### `_validate_schema(raw, config)`

Perform additional structural validation beyond Pydantic models.

Args:
raw: Raw configuration mapping used for structural checks.
config: Optional resolved configuration for cross-field validation.

Raises:
ConfigError: When structural constraints are violated.

### `load_raw_yaml(config_path)`

Load and parse a YAML configuration file into a mutable mapping.

Args:
config_path: Path to the YAML configuration file.

Returns:
Mutable mapping representation of the YAML content.

Raises:
SystemExit: If the file is not found on disk.
ConfigError: If the YAML cannot be parsed or is structurally invalid.

### `load_config(config_path)`

Load configuration from disk without performing additional schema validation.

Args:
config_path: Path to the YAML configuration file.

Returns:
ResolvedConfig produced from the raw file contents.

Raises:
ConfigError: If the configuration cannot be parsed or validated.

### `validate_config(config_path)`

Validate a configuration file and return the resolved settings.

Args:
config_path: Path to the YAML configuration file.

Returns:
ResolvedConfig object after validation.

Raises:
ConfigError: If validation fails.

### `_compress_old_log(path)`

Compress a log file in-place using gzip to reclaim disk space.

Args:
path: Filesystem location of the log file to compress.

### `_cleanup_logs(log_dir, retention_days)`

Apply rotation and retention policy to the log directory.

Args:
log_dir: Directory that stores structured log files.
retention_days: Number of days to keep uncompressed log files.

### `setup_logging()`

Configure structured logging handlers for ontology downloads.

Args:
level: Logging level applied to the ontology downloader logger.
retention_days: Number of days to retain uncompressed log files before archival.
max_log_size_mb: Maximum size of each log file before rotation occurs.
log_dir: Optional override for the log directory; falls back to defaults when omitted.

Returns:
Logger instance configured with console and rotating JSON handlers.

### `_create_stub_module(name, attrs)`

Create a stub module populated with the provided attributes.

Args:
name: Dotted module path that should appear in :data:`sys.modules`.
attrs: Mapping of attribute names to objects exposed by the stub.

Returns:
Module instance that mimics the requested package for test isolation.

### `_create_stub_bnode(value)`

Create a deterministic blank node identifier for rdflib stubs.

Args:
value: Optional explicit identifier to reuse instead of auto-incrementing.

Returns:
RDF blank node identifier anchored by the ``_:`` prefix.

### `_create_stub_literal(value)`

Represent literals as simple string values for stub graphs.

Args:
value: Python value to coerce into an rdflib-style literal.

Returns:
String literal representation suitable for Turtle serialization.

### `_create_stub_uri(value)`

Create a URI reference that matches rdflib serialization expectations.

Args:
value: URI string, optionally already wrapped in angle brackets.

Returns:
URI reference wrapped in angle brackets for Turtle compatibility.

### `_import_module(name)`

Import a module by name using :mod:`importlib`.

The indirection makes it trivial to monkeypatch the import logic in unit
tests without modifying global interpreter state.

Args:
name: Fully qualified module name to load.

Returns:
Imported module, falling back to the real implementation when present.

Raises:
ModuleNotFoundError: If the module cannot be located.

### `_create_pystow_stub(root)`

Return a stub module implementing ``join`` similar to pystow.

Args:
root: Filesystem directory that acts as the backing storage root.

Returns:
Module object exposing a ``join`` helper compatible with pystow usage.

### `_create_rdflib_stub()`

Create a stub implementation compatible with rdflib usage in tests.

Returns:
Module object that mirrors the small subset of rdflib used by validators.

### `_create_pronto_stub()`

Create a stub module mimicking pronto interfaces.

Returns:
Module object exposing a lightweight :class:`Ontology` implementation.

### `_create_owlready_stub()`

Create a stub module mimicking owlready2 key behaviour.

Returns:
Module object providing ``get_ontology`` compatible with owlready2.

### `get_pystow()`

Return the real :mod:`pystow` module or a fallback stub.

Args:
None.

Returns:
Real pystow module when installed, otherwise a deterministic stub.

### `get_rdflib()`

Return :mod:`rdflib` or a stub supporting limited graph operations.

Args:
None.

Returns:
Real rdflib module when available, else a stub graph implementation.

### `get_pronto()`

Return :mod:`pronto` or a stub with minimal ontology behaviour.

Args:
None.

Returns:
Real pronto module when installed, else a stub ontology wrapper.

### `get_owlready2()`

Return :mod:`owlready2` or a stub matching the API used in validators.

Args:
None.

Returns:
Real owlready2 module when available, else a limited stub replacement.

### `_safe_identifiers(ontology_id, version)`

Return identifiers sanitized for filesystem usage.

Args:
ontology_id: Raw ontology identifier that may contain unsafe characters.
version: Version label that should be filesystem-friendly.

Returns:
Tuple ``(safe_id, safe_version)`` containing sanitised values.

### `_directory_size(path)`

Return the total size in bytes for all regular files under ``path``.

Args:
path: Root directory whose files should be measured.

Returns:
Cumulative size in bytes of every regular file within ``path``.

### `get_storage_backend()`

Instantiate the storage backend based on environment configuration.

Args:
None.

Returns:
Storage backend instance selected according to ``ONTOFETCH_STORAGE_URL``.

### `_log_download_memory(logger, event)`

Emit debug-level memory usage snapshots when enabled.

Args:
logger: Logger instance controlling verbosity for download telemetry.
event: Short label describing the lifecycle point (e.g., ``before``).

Returns:
None

### `_is_retryable_status(status_code)`

*No documentation available.*

### `_enforce_idn_safety(host)`

Validate internationalized hostnames and reject suspicious patterns.

Args:
host: Hostname component extracted from the download URL.

Returns:
None

Raises:
ConfigError: If the hostname mixes multiple scripts or contains invisible characters.

### `_rebuild_netloc(parsed, ascii_host)`

Reconstruct URL netloc with a normalized hostname.

Args:
parsed: Parsed URL components produced by :func:`urllib.parse.urlparse`.
ascii_host: ASCII-normalized hostname (potentially IPv6).

Returns:
String suitable for use as the netloc portion of a URL.

### `validate_url_security(url, http_config)`

Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

Hostnames are converted to punycode before resolution, and both direct IP
addresses and DNS results are rejected when they target private or loopback
ranges to prevent server-side request forgery.

Args:
url: URL returned by a resolver for ontology download.
http_config: Download configuration providing optional host allowlist.

Returns:
HTTPS URL safe for downstream download operations.

Raises:
ConfigError: If the URL violates security requirements or allowlists.

### `sha256_file(path)`

Compute the SHA-256 digest for the provided file.

Args:
path: Path to the file whose digest should be calculated.

Returns:
Hexadecimal SHA-256 checksum string.

### `_validate_member_path(member_name)`

Validate archive member paths to prevent traversal attacks.

Args:
member_name: Path declared within the archive.

Returns:
Sanitised relative path safe for extraction on the local filesystem.

Raises:
ConfigError: If the member path is absolute or contains traversal segments.

### `_check_compression_ratio()`

Ensure compressed archives do not expand beyond the permitted ratio.

Args:
total_uncompressed: Sum of file sizes within the archive.
compressed_size: Archive file size on disk (or sum of compressed entries).
archive: Path to the archive on disk.
logger: Optional logger for emitting diagnostic messages.
archive_type: Human readable label for error messages (ZIP/TAR).

Raises:
ConfigError: If the archive exceeds the allowed expansion ratio.

### `_is_zip_symlink(info)`

Return True when a ZIP member encodes a symbolic link.

### `extract_zip_safe(zip_path, destination)`

Extract a ZIP archive while preventing traversal and compression bombs.

Args:
zip_path: Path to the ZIP file to extract.
destination: Directory where extracted files should be stored.
logger: Optional logger for emitting extraction telemetry.

Returns:
List of extracted file paths.

Raises:
ConfigError: If the archive contains unsafe paths, compression bombs, or is missing.

### `extract_tar_safe(tar_path, destination)`

Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

Args:
tar_path: Path to the tar archive (tar, tar.gz, tar.xz).
destination: Directory where extracted files should be stored.
logger: Optional logger for emitting extraction telemetry.

Returns:
List of extracted file paths.

Raises:
ConfigError: If the archive is missing, unsafe, or exceeds compression limits.

### `extract_archive_safe(archive_path, destination)`

Extract archives by dispatching to the appropriate safe handler.

Args:
archive_path: Path to the archive on disk.
destination: Directory where files should be extracted.
logger: Optional logger receiving structured extraction telemetry.

Returns:
List of paths extracted from the archive in the order processed.

Raises:
ConfigError: If the archive format is unsupported or extraction fails.

### `_get_bucket(host, http_config, service)`

Return a token bucket keyed by host and optional service name.

Args:
host: Hostname extracted from the download URL.
http_config: Download configuration providing base rate limits.
service: Logical service identifier enabling per-service overrides.

Returns:
TokenBucket instance shared across downloads for throttling, seeded
with either per-host defaults or service-specific overrides.

### `download_stream()`

Download ontology content with HEAD validation, rate limiting, caching, retries, and hash checks.

Args:
url: URL of the ontology document to download.
destination: Target file path for the downloaded content.
headers: HTTP headers forwarded to the download request.
previous_manifest: Manifest metadata from a prior run, used for caching.
http_config: Download configuration containing timeouts, limits, and rate controls.
cache_dir: Directory where intermediary cached files are stored.
logger: Logger adapter for structured download telemetry.
expected_media_type: Expected Content-Type for validation, if known.
service: Logical service identifier for per-service rate limiting.

Returns:
DownloadResult describing the final artifact and metadata.

Raises:
ConfigError: If validation fails, limits are exceeded, or HTTP errors occur.

### `_log_validation_memory(logger, validator, event)`

Emit memory usage diagnostics for a validator when debug logging is enabled.

Args:
logger: Logger responsible for validator telemetry.
validator: Name of the validator emitting the event.
event: Lifecycle label describing when the measurement is captured.

### `_write_validation_json(path, payload)`

Persist structured validation metadata to disk as JSON.

Args:
path: Destination path for the JSON payload.
payload: Mapping containing validation results.

### `_python_merge_sort(source, destination)`

Sort an N-Triples file using a disk-backed merge strategy.

Args:
source: Path to the unsorted triple file.
destination: Output path that receives sorted triples.
chunk_size: Number of lines loaded into memory per chunk before flushing.

### `_term_to_string(term, namespace_manager)`

Render an RDF term using the provided namespace manager.

Args:
term: RDF term such as a URIRef, BNode, or Literal.
namespace_manager: Namespace manager responsible for prefix resolution.

Returns:
Term rendered in N3 form, falling back to :func:`str` when unavailable.

### `_canonicalize_turtle(graph)`

Return canonical Turtle output with sorted prefixes and triples.

The canonical form mirrors the ontology downloader specification by sorting
prefixes lexicographically and emitting triples ordered by subject,
predicate, and object so downstream hashing yields deterministic values.

Args:
graph: RDF graph containing triples to canonicalize.

Returns:
Canonical Turtle serialization as a string.

### `_canonicalize_blank_nodes_line(line, mapping)`

Replace blank node identifiers with deterministic sequential labels.

Args:
line: Serialized triple line containing blank node identifiers.
mapping: Mutable mapping preserving deterministic blank node assignments.

Returns:
Triple line with normalized blank node identifiers.

### `_sort_triple_file(source, destination)`

Sort serialized triple lines using platform sort when available.

Args:
source: Path to the unsorted triple file.
destination: Output path that receives sorted triples.

### `normalize_streaming(source, output_path)`

Normalize ontologies using streaming canonical Turtle serialization.

The streaming path serializes triples to a temporary file, leverages the
platform ``sort`` command (when available) to order triples lexicographically,
and streams the canonical Turtle output while computing a SHA-256 digest.
When ``output_path`` is provided the canonical form is persisted without
retaining the entire content in memory.

Args:
source: Path to the ontology document providing triples.
output_path: Optional destination for the normalized Turtle document.
graph: Optional pre-loaded RDF graph re-used instead of reparsing.
chunk_bytes: Threshold controlling how frequently buffered bytes are flushed.

Returns:
SHA-256 hex digest of the canonical Turtle content.

### `_worker_pronto(payload)`

Execute Pronto validation logic and emit JSON-friendly results.

### `_worker_owlready2(payload)`

Execute Owlready2 validation logic and emit JSON-friendly results.

### `_run_validator_subprocess(name, payload)`

Execute a validator worker module within a subprocess.

The subprocess workflow enforces parser timeouts, returns JSON payloads,
and helps release memory held by heavy libraries such as Pronto and
Owlready2 after each validation completes.

### `_run_with_timeout(func, timeout_sec)`

Execute a callable and raise :class:`ValidationTimeout` on deadline expiry.

Args:
func: Callable invoked without arguments.
timeout_sec: Number of seconds allowed for execution.

Returns:
None

Raises:
ValidationTimeout: When the callable exceeds the allotted runtime.

### `_prepare_xbrl_package(request, logger)`

Extract XBRL taxonomy ZIP archives for downstream validation.

Args:
request: Validation request describing the ontology package under test.
logger: Logger used to record extraction telemetry.

Returns:
Tuple containing the entrypoint path passed to Arelle and a list of artifacts.

Raises:
ValueError: If the archive is malformed or contains unsafe paths.

### `validate_rdflib(request, logger)`

Parse ontologies with rdflib, canonicalize Turtle output, and emit hashes.

Args:
request: Validation request describing the target ontology and output directories.
logger: Logger adapter used for structured validation events.

Returns:
ValidationResult capturing success state, metadata, canonical hash,
and generated files.

Raises:
ValidationTimeout: Propagated when parsing exceeds configured timeout.

### `validate_pronto(request, logger)`

Execute Pronto validation in an isolated subprocess and emit OBO Graphs when requested.

Args:
request: Validation request describing ontology inputs and output directories.
logger: Structured logger for recording warnings and failures.

Returns:
ValidationResult with parsed ontology statistics, subprocess output,
and any generated artifacts.

Raises:
ValidationTimeout: Propagated when Pronto takes longer than allowed.

### `validate_owlready2(request, logger)`

Inspect ontologies with Owlready2 in a subprocess to count entities and catch parsing errors.

Args:
request: Validation request referencing the ontology to parse.
logger: Logger for reporting failures or memory warnings.

Returns:
ValidationResult summarizing entity counts or failure details.

Raises:
None

### `validate_robot(request, logger)`

Run ROBOT CLI validation and conversion workflows when available.

Args:
request: Validation request detailing ontology paths and output locations.
logger: Logger adapter for reporting warnings and CLI errors.

Returns:
ValidationResult describing generated outputs or encountered issues.

Raises:
None

### `validate_arelle(request, logger)`

Validate XBRL ontologies with Arelle CLI if installed.

Args:
request: Validation request referencing the ontology under test.
logger: Logger used to communicate validation progress and failures.

Returns:
ValidationResult indicating whether the validation completed and
referencing any produced log files.

Raises:
None

### `_load_validator_plugins(logger)`

Discover validator plugins registered via entry points.

### `_run_validator_task(validator, request, logger)`

Execute a single validator with exception guards.

### `run_validators(requests, logger)`

Execute registered validators and aggregate their results.

Args:
requests: Iterable of validation requests that specify validators to run.
logger: Logger adapter shared across validation executions.

Returns:
Mapping from validator name to the corresponding ValidationResult.

### `_run_worker_cli(name, stdin_payload)`

Execute a validator worker handler and emit JSON to stdout.

### `main()`

Entry point for module execution providing validator worker dispatch.

Args:
None.

Returns:
None.

### `get_manifest_schema()`

Return a deep copy of the manifest JSON Schema definition.

Args:
None

Returns:
Dictionary describing the manifest JSON Schema.

### `validate_manifest_dict(payload)`

Validate manifest payload against the JSON Schema definition.

Args:
payload: Manifest dictionary loaded from JSON.
source: Optional filesystem path for contextual error reporting.

Returns:
None

Raises:
ConfigurationError: If validation fails.

### `parse_http_datetime(value)`

Parse HTTP ``Last-Modified`` style timestamps into UTC datetimes.

Args:
value: Timestamp string from HTTP headers such as ``Last-Modified``.

Returns:
Optional[datetime]: Normalized UTC datetime when parsing succeeds.

Raises:
None: Parsing failures are converted into a ``None`` return value.

### `parse_iso_datetime(value)`

Parse ISO-8601 timestamps into timezone-aware UTC datetimes.

Args:
value: ISO-8601 formatted timestamp string.

Returns:
Optional[datetime]: Normalized UTC datetime when parsing succeeds.

Raises:
None: Invalid values return ``None`` instead of raising.

### `parse_version_timestamp(value)`

Parse version strings or manifest timestamps into UTC datetimes.

Args:
value: Version identifier or timestamp string to normalize.

Returns:
Optional[datetime]: Parsed UTC datetime when the input matches supported formats.

Raises:
None: All parsing failures result in ``None``.

### `infer_version_timestamp(value)`

Infer a timestamp from resolver version identifiers.

Args:
value: Resolver version string containing date-like fragments.

Returns:
Optional[datetime]: Parsed UTC datetime when the value contains recoverable dates.

Raises:
None: Returns ``None`` instead of raising on unparseable inputs.

### `_coerce_datetime(value)`

Return timezone-aware datetime parsed from HTTP or ISO timestamp.

### `_normalize_timestamp(value)`

Return canonical ISO8601 string for HTTP timestamp headers.

### `_populate_plan_metadata(planned, config, adapter)`

Augment planned fetch with HTTP metadata when available.

### `_migrate_manifest_inplace(payload)`

Upgrade manifests created with older schema versions in place.

### `_read_manifest(manifest_path)`

Return previously recorded manifest data if a valid JSON file exists.

Args:
manifest_path: Filesystem path where the manifest is stored.

Returns:
Parsed manifest dictionary when available and valid, otherwise ``None``.

### `_validate_manifest(manifest)`

Check that a manifest instance satisfies structural and type requirements.

Args:
manifest: Manifest produced after a download completes.

Raises:
ConfigurationError: If required fields are missing or contain invalid types.

### `_parse_last_modified(value)`

Return a timezone-aware datetime parsed from HTTP date headers.

### `_fetch_last_modified(plan, config, logger)`

Probe the upstream plan URL for a Last-Modified header.

### `_write_manifest(manifest_path, manifest)`

Persist a validated manifest to disk as JSON.

Args:
manifest_path: Destination path for the manifest file.
manifest: Manifest describing the downloaded ontology artifact.

### `_build_destination(spec, plan, config)`

Determine the output directory and filename for a download.

Args:
spec: Fetch specification identifying the ontology.
plan: Resolver plan containing URL metadata and optional hints.
config: Resolved configuration with storage layout parameters.

Returns:
Tuple containing the target file path, resolved version, and base directory.

### `_ensure_license_allowed(plan, config, spec)`

Confirm the ontology license is present in the configured allow list.

Args:
plan: Resolver plan returned for the ontology.
config: Resolved configuration containing accepted licenses.
spec: Fetch specification for contextual error reporting.

Raises:
ConfigurationError: If the plan's license is not permitted.

### `_resolver_candidates(spec, config)`

*No documentation available.*

### `_resolve_plan_with_fallback(spec, config, adapter)`

*No documentation available.*

### `fetch_one(spec)`

Fetch, validate, and persist a single ontology described by *spec*.

Args:
spec: Ontology fetch specification describing sources and formats.
config: Optional resolved configuration overriding global defaults.
correlation_id: Correlation identifier for structured logging.
logger: Optional logger to reuse instead of configuring a new one.
force: When ``True``, bypass local cache checks and redownload artifacts.

Returns:
FetchResult: Structured result containing manifest metadata and resolver attempts.

Raises:
ResolverError: If all resolver candidates fail to retrieve the ontology.

### `plan_one(spec)`

Return a resolver plan for a single ontology without performing downloads.

Args:
spec: Fetch specification describing the ontology to plan.
config: Optional resolved configuration providing defaults and limits.
correlation_id: Correlation identifier reused for logging context.
logger: Logger instance used to emit resolver telemetry.

Returns:
PlannedFetch containing the normalized spec, resolver name, and plan.

Raises:
ResolverError: If all resolvers fail to produce a plan for ``spec``.
ConfigurationError: If licence checks reject the planned ontology.

### `plan_all(specs)`

Return resolver plans for a collection of ontologies.

Args:
specs: Iterable of fetch specifications to resolve.
config: Optional resolved configuration reused across plans.
logger: Logger instance used for annotation-aware logging.
since: Optional cutoff date; plans older than this timestamp are filtered out.

Returns:
List of PlannedFetch entries describing each ontology plan.

Raises:
ResolverError: Propagated when fallback planning fails for any spec.
ConfigurationError: When licence enforcement rejects a planned ontology.

### `fetch_all(specs)`

Fetch a sequence of ontologies sequentially.

Args:
specs: Iterable of fetch specifications to process.
config: Optional resolved configuration shared across downloads.
logger: Logger used to emit progress and error events.
force: When True, skip manifest reuse and download everything again.

Returns:
List of FetchResult entries corresponding to completed downloads.

Raises:
OntologyDownloadError: Propagated when downloads fail and the pipeline
is configured to stop on error.

### `_safe_lock_component(value)`

Return a filesystem-safe token for lock filenames.

### `_version_lock(ontology_id, version)`

Acquire an inter-process lock for a specific ontology version.

### `validate_level(cls, value)`

Validate logging level values.

Args:
value: Logging level provided in configuration.

Returns:
Uppercase logging level string accepted by :mod:`logging`.

Raises:
ValueError: If the supplied level is not among the supported options.

### `validate_rate_limits(cls, value)`

Ensure rate limit strings follow the expected pattern.

Args:
value: Mapping of service name to rate limit expression.

Returns:
Validated mapping preserving the original values.

Raises:
ValueError: If any rate limit fails to match the accepted pattern.

### `rate_limit_per_second(self)`

Convert ``per_host_rate_limit`` to a requests-per-second value.

Args:
None.

Returns:
Requests-per-second value derived from the configuration.

Raises:
ValueError: If the configured rate limit string is invalid.

### `parse_service_rate_limit(self, service)`

Parse a per-service rate limit to requests per second.

Args:
service: Logical service name to look up.

Returns:
Requests-per-second value when configured, otherwise ``None``.

Raises:
None.

### `normalized_allowed_hosts(self)`

Return allowlist entries normalized to lowercase punycode labels.

Args:
None.

Returns:
Tuple of (exact hostnames, wildcard suffixes) when entries exist,
otherwise ``None`` if no valid allowlist entries are configured.

Raises:
ValueError: If any configured hostname cannot be converted to punycode.

### `polite_http_headers(self)`

Return polite HTTP headers suitable for resolver API calls.

The headers include a deterministic ``User-Agent`` string, propagate a
``From`` contact address when configured, and synthesize an ``X-Request-ID``
correlated with the current fetch so API providers can trace requests.

Args:
correlation_id: Identifier attached to the current batch for log correlation.
request_id: Optional override for ``X-Request-ID`` header; auto-generated when ``None``.
timestamp: Optional timestamp used when constructing the request identifier.

Returns:
Mapping of header names to polite values complying with provider guidelines.

### `validate_prefer_source(cls, value)`

Allow custom resolver identifiers while warning about unknown entries.

Args:
value: Ordered list of resolver identifiers supplied by configuration.

Returns:
Sanitised resolver list preserving the original order.

Raises:
ValueError: If any resolver name is not recognized.

### `from_defaults(cls)`

Create an empty resolved configuration with library defaults.

Args:
None

Returns:
ResolvedConfig populated with default settings and no fetch specs.

### `format(self, record)`

Serialize a logging record into a JSON line.

Args:
record: Logging record produced by the underlying logger.

Returns:
JSON string containing logging metadata and contextual extras.

### `__getitem__(self, key)`

Return the expanded URI for the provided key.

Args:
key: Local name appended to the base namespace.

Returns:
Fully qualified IRI for the requested key.

### `bind(self, prefix, namespace)`

Associate a namespace prefix with a full URI.

Args:
prefix: Namespace shorthand used in Turtle output.
namespace: Fully qualified namespace IRI.

Returns:
None.

### `namespaces(self)`

Yield bound namespaces as ``(prefix, namespace)`` tuples.

Args:
None.

Returns:
Iterable of namespace bindings recorded by the manager.

### `join()`

Join path segments beneath the stub root directory.

Args:
*segments: Arbitrary path components appended to the root.

Returns:
Path anchored at ``root`` with the provided segments appended.

### `parse(self, source, format)`

Parse a Turtle file into the stub graph.

Args:
source: Local filesystem path to a Turtle document.
format: Optional rdflib format hint that is ignored by the stub.
_kwargs: Additional keyword arguments unused by the stub.

Returns:
The current graph instance populated with parsed triple strings.

Raises:
FileNotFoundError: If the Turtle file cannot be read.

### `serialize(self, destination, format)`

Serialize the stub graph back to Turtle text.

Args:
destination: Optional output target path or file-like object.
format: Optional rdflib format that is ignored by the stub.
_kwargs: Additional keyword arguments unused by the stub.

Returns:
Destination handle or Turtle text when serializing to a string.

Raises:
OSError: If writing to the provided destination fails.

### `bind(self, prefix, namespace)`

Attach a namespace binding to the internal namespace manager.

Args:
prefix: Namespace shorthand used in serialization.
namespace: Fully qualified namespace IRI.

Returns:
None.

### `namespaces(self)`

Return the stored namespace bindings for serialization.

Args:
None.

Returns:
Iterable containing prefix-to-namespace mappings.

### `__len__(self)`

Return the number of parsed triples retained by the stub.

Args:
None.

Returns:
Integer count of Turtle triples captured during parsing.

### `__iter__(self)`

Iterate over the stored Turtle triple representations.

Args:
None.

Returns:
Iterator yielding Turtle triple strings from the stub graph.

### `get_ontology(iri)`

Return a stub ontology bound to the provided IRI.

Args:
iri: Ontology IRI used to instantiate the stub.

Returns:
`_StubOntology` instance wrapping the provided IRI.

### `prepare_version(self, ontology_id, version)`

Return a working directory prepared for the given ontology version.

Args:
ontology_id: Identifier of the ontology being downloaded.
version: Version string representing the in-flight download.

Returns:
Path to a freshly prepared directory tree ready for population.

### `ensure_local_version(self, ontology_id, version)`

Ensure the requested version is present locally and return its path.

Args:
ontology_id: Identifier whose version must be present.
version: Version string that should exist on local storage.

Returns:
Path to the local directory containing the requested version.

### `available_versions(self, ontology_id)`

Return sorted version identifiers currently stored for an ontology.

Args:
ontology_id: Identifier whose known versions are requested.

Returns:
Sorted list of version strings recognised by the backend.

### `available_ontologies(self)`

Return sorted ontology identifiers known to the backend.

Args:
None.

Returns:
Alphabetically sorted list of ontology identifiers the backend can
service.

### `finalize_version(self, ontology_id, version, local_dir)`

Persist the working directory after validation succeeds.

Args:
ontology_id: Identifier of the ontology that completed processing.
version: Version string associated with the finalized artifacts.
local_dir: Directory containing the validated ontology payload.

Returns:
None.

### `version_path(self, ontology_id, version)`

Return the canonical storage path for ``ontology_id``/``version``.

Args:
ontology_id: Identifier of the ontology being queried.
version: Version string for which a canonical path is needed.

Returns:
Path pointing to the storage location for the requested version.

### `delete_version(self, ontology_id, version)`

Delete a stored version returning the number of bytes reclaimed.

Args:
ontology_id: Identifier whose version should be removed.
version: Version string targeted for deletion.

Returns:
Number of bytes reclaimed by removing the stored version.

Raises:
OSError: If the underlying storage provider fails to delete data.

### `set_latest_version(self, ontology_id, version)`

Update the latest version marker for operators and CLI tooling.

Args:
ontology_id: Identifier whose latest marker requires updating.
version: Version string to record as the active release.

Returns:
None.

### `_version_dir(self, ontology_id, version)`

Return the directory where a given ontology version is stored.

Args:
ontology_id: Identifier whose storage directory is required.
version: Version string combined with the identifier.

Returns:
Path pointing to ``root/<ontology_id>/<version>``.

### `prepare_version(self, ontology_id, version)`

Create the working directory structure for a download run.

Args:
ontology_id: Identifier of the ontology being processed.
version: Canonical version string for the ontology.

Returns:
Path to the prepared base directory containing ``original``,
``normalized``, and ``validation`` subdirectories.

### `ensure_local_version(self, ontology_id, version)`

Ensure a local workspace exists for ``ontology_id``/``version``.

Args:
ontology_id: Identifier whose workspace must exist.
version: Version string that should map to a directory.

Returns:
Path to the local directory for the ontology version.

### `available_versions(self, ontology_id)`

Return sorted versions already present for an ontology.

Args:
ontology_id: Identifier whose stored versions should be listed.

Returns:
Sorted list of version strings found under the storage root.

### `available_ontologies(self)`

Return ontology identifiers discovered under ``root``.

Args:
None.

Returns:
Sorted list of ontology identifiers available locally.

### `finalize_version(self, ontology_id, version, local_dir)`

Finalize a local version directory (no-op for purely local storage).

Args:
ontology_id: Identifier that finished processing.
version: Version string associated with the processed ontology.
local_dir: Directory containing the ready-to-serve ontology.

Returns:
None.

### `version_path(self, ontology_id, version)`

Return the local storage directory for the requested version.

Args:
ontology_id: Identifier being queried.
version: Version string whose storage path is needed.

Returns:
Path pointing to the stored ontology version.

### `delete_version(self, ontology_id, version)`

Delete a stored ontology version returning reclaimed bytes.

Args:
ontology_id: Identifier whose stored version should be removed.
version: Version string targeted for deletion.

Returns:
Number of bytes reclaimed by removing the version directory.

Raises:
OSError: Propagated if filesystem deletion fails.

### `set_latest_version(self, ontology_id, version)`

Update symlink and marker file indicating the latest version.

Args:
ontology_id: Identifier whose latest marker should be updated.
version: Version string to record as the latest processed build.

Returns:
None.

### `_remote_version_path(self, ontology_id, version)`

Return the remote filesystem path for the specified ontology version.

Args:
ontology_id: Identifier whose remote storage path is required.
version: Version string associated with the ontology release.

Returns:
Posix-style path referencing the remote storage location.

### `available_versions(self, ontology_id)`

Return versions aggregated from local cache and remote storage.

Args:
ontology_id: Identifier whose version catalogue is required.

Returns:
Sorted list combining local and remote version identifiers.

### `available_ontologies(self)`

Return ontology identifiers available locally or remotely.

Args:
None.

Returns:
Sorted set union of local and remote ontology identifiers.

### `ensure_local_version(self, ontology_id, version)`

Mirror a remote ontology version into the local cache when absent.

Args:
ontology_id: Identifier whose version should exist locally.
version: Version string to ensure within the local cache.

Returns:
Path to the local directory containing the requested version.

### `finalize_version(self, ontology_id, version, local_dir)`

Upload the finalized local directory to the remote store.

Args:
ontology_id: Identifier of the ontology that has completed processing.
version: Version string associated with the finalised ontology.
local_dir: Directory containing the validated ontology payload.

Returns:
None.

### `delete_version(self, ontology_id, version)`

Delete both local and remote copies of a stored version.

Args:
ontology_id: Identifier whose stored version should be deleted.
version: Version string targeted for deletion.

Returns:
Total bytes reclaimed across local and remote storage.

Raises:
OSError: Propagated if remote deletion fails irrecoverably.

### `consume(self, tokens)`

Consume tokens from the bucket, sleeping until capacity is available.

Args:
tokens: Number of tokens required for the current download request.

Returns:
None

### `_preliminary_head_check(self, url, session)`

Probe the origin with HEAD to audit media type and size before downloading.

The HEAD probe allows the pipeline to abort before streaming large
payloads that exceed configured limits and to log early warnings for
mismatched Content-Type headers reported by the origin.

Args:
url: Fully qualified download URL resolved by the planner.
session: Prepared requests session used for outbound calls.

Returns:
Tuple ``(content_type, content_length)`` extracted from response
headers. Each element is ``None`` when the origin omits it.

Raises:
ConfigError: If the origin reports a payload larger than the
configured ``max_download_size_gb`` limit.

### `_validate_media_type(self, actual_content_type, expected_media_type, url)`

Validate that the received ``Content-Type`` header is acceptable, tolerating aliases.

RDF endpoints often return generic XML or Turtle aliases, so the
validator accepts a small set of known MIME variants while still
surfacing actionable warnings for unexpected types.

Args:
actual_content_type: Raw header value reported by the origin server.
expected_media_type: MIME type declared by resolver metadata.
url: Download URL logged when mismatches occur.

Returns:
None

### `__call__(self, url, output_file, pooch_logger)`

Stream ontology content to disk while enforcing download policies.

Args:
url: Secure download URL resolved by the planner.
output_file: Temporary filename managed by pooch during download.
pooch_logger: Logger instance supplied by pooch (unused).

Raises:
ConfigError: If download limits are exceeded or filesystem errors occur.
requests.HTTPError: Propagated when HTTP status codes indicate failure.

Returns:
None

### `to_dict(self)`

Represent the validation result as a JSON-serializable dict.

Args:
None.

Returns:
Dictionary with boolean status, detail payload, and output paths.

### `_replace(match)`

*No documentation available.*

### `_flush(writer)`

*No documentation available.*

### `_parse()`

Parse the ontology with rdflib to populate the graph object.

### `_determine_max_workers()`

*No documentation available.*

### `to_dict(self)`

Return a JSON-serializable dictionary for the manifest.

Args:
None

Returns:
Dictionary representing the manifest payload.

### `to_json(self)`

Serialize the manifest to a stable, human-readable JSON string.

Args:
None

Returns:
JSON document encoding the manifest metadata.

### `plan(self, spec, config, logger)`

Return a FetchPlan describing how to obtain the ontology.

Args:
spec: Ontology fetch specification under consideration.
config: Fully resolved configuration containing defaults.
logger: Logger adapter scoped to the current fetch request.

Returns:
Concrete plan containing download URL, headers, and metadata.

### `_add_candidate(candidate)`

*No documentation available.*

### `_add(name)`

*No documentation available.*

### `terms(self)`

Return a static list of ontology term identifiers.

Args:
None.

Returns:
Iterable containing deterministic ontology term IDs.

### `dump(self, destination, format)`

Write an empty ontology JSON payload to ``destination``.

Args:
destination: Filesystem path where the JSON stub is stored.
format: Serialization format, kept for API equivalence.

Returns:
None.

### `load(self)`

Provide a fluent API returning the ontology itself.

Args:
None.

Returns:
The same stub ontology instance, mimicking owlready2.

Raises:
None.

### `classes(self)`

Return a deterministic set of class identifiers.

Args:
None.

Returns:
List of representative ontology class names.

### `_handler(signum, frame)`

Signal handler converting SIGALRM into :class:`ValidationTimeout`.

Args:
signum: Received signal number.
frame: Current stack frame (unused).

### `_execute_candidate()`

*No documentation available.*

### `_emit(text)`

*No documentation available.*

## 3. Classes

### `ConfigError`

Raised when ontology configuration files are invalid or inconsistent.

Attributes:
message: Human-readable explanation of the configuration flaw.

Examples:
>>> try:
...     raise ConfigError("missing id")
... except ConfigError as exc:
...     assert "missing id" in str(exc)

### `LoggingConfiguration`

Structured logging options for ontology download operations.

Attributes:
level: Logging level for downloader telemetry (DEBUG, INFO, etc.).
max_log_size_mb: Maximum size of log files before rotation occurs.
retention_days: Number of days log files are retained on disk.

Examples:
>>> config = LoggingConfiguration(level="debug")
>>> config.level
'DEBUG'

### `ValidationConfig`

Validation limits governing parser execution.

Attributes:
parser_timeout_sec: Maximum runtime allowed for ontology parsing.
max_memory_mb: Memory ceiling allocated to validation routines.
skip_reasoning_if_size_mb: Threshold above which reasoning is skipped.
streaming_normalization_threshold_mb: File size threshold for streaming normalization.

Examples:
>>> ValidationConfig(parser_timeout_sec=120).parser_timeout_sec
120

### `DownloadConfiguration`

HTTP download, throttling, and polite header settings for resolvers.

Attributes:
max_retries: Maximum retry attempts for failed download requests.
timeout_sec: Base timeout applied to metadata requests.
download_timeout_sec: Timeout applied to streaming download operations.
backoff_factor: Exponential backoff multiplier between retry attempts.
per_host_rate_limit: Token bucket rate limit string for host throttling.
max_download_size_gb: Maximum permitted download size in gigabytes.
concurrent_downloads: Allowed number of simultaneous downloads.
validate_media_type: Whether to enforce Content-Type validation.
rate_limits: Optional per-service rate limit overrides.
allowed_hosts: Optional allowlist restricting download hostnames.
polite_headers: Default polite HTTP headers applied to resolver API calls.

Examples:
>>> cfg = DownloadConfiguration(per_host_rate_limit="10/second")
>>> round(cfg.rate_limit_per_second(), 2)
10.0

### `DefaultsConfig`

Collection of default settings for ontology fetch specifications.

Attributes:
accept_licenses: Licenses accepted by default during downloads.
normalize_to: Preferred formats for normalized ontology output.
prefer_source: Ordered list of resolver priorities.
http: Download configuration defaults.
validation: Validation configuration defaults.
logging: Logging configuration defaults.
continue_on_error: Whether processing continues after failures.
resolver_fallback_enabled: Whether automatic resolver fallback is enabled.

Examples:
>>> defaults = DefaultsConfig()
>>> "obo" in defaults.prefer_source
True

### `ResolvedConfig`

Container for merged configuration defaults and fetch specifications.

Attributes:
defaults: Default configuration applied to all ontology fetch specs.
specs: List of individual fetch specifications after merging defaults.

Examples:
>>> config = ResolvedConfig.from_defaults()
>>> isinstance(config.defaults, DefaultsConfig)
True

### `EnvironmentOverrides`

Environment variable overrides for ontology downloader defaults.

Attributes:
max_retries: Override for retry count via ``ONTOFETCH_MAX_RETRIES``.
timeout_sec: Override for metadata timeout via ``ONTOFETCH_TIMEOUT_SEC``.
download_timeout_sec: Override for streaming timeout.
per_host_rate_limit: Override for per-host rate limit string.
backoff_factor: Override for exponential backoff multiplier.
log_level: Override for logging level.

Examples:
>>> overrides = EnvironmentOverrides()
>>> overrides.model_config['env_prefix']
'ONTOFETCH_'

### `JSONFormatter`

Formatter emitting JSON structured logs.

Attributes:
default_msec_format: Fractional second format applied to timestamps.

Examples:
>>> formatter = JSONFormatter()
>>> isinstance(formatter.format(logging.LogRecord("test", 20, __file__, 1, "msg", (), None)), str)  # doctest: +SKIP
True

### `_StubNamespace`

Minimal replacement mimicking rdflib Namespace behaviour.

Attributes:
_base: Base IRI string used to expand namespace members.

Examples:
>>> ns = _StubNamespace("http://example.org/")
>>> ns["Term"]
'http://example.org/Term'

### `_StubNamespaceManager`

Provide a namespaces() method compatible with rdflib.

Attributes:
_bindings: Mapping of prefixes to namespace IRIs.

Examples:
>>> manager = _StubNamespaceManager()
>>> manager.bind("ex", "http://example.org/")
>>> list(manager.namespaces())
[('ex', 'http://example.org/')]

### `_StubGraph`

Lightweight graph implementation mirroring rdflib essentials.

Attributes:
_triples: In-memory list of Turtle triple strings.
_last_text: Raw Turtle content captured during parsing.
namespace_manager: Stub namespace manager for rdflib compatibility.

Examples:
>>> graph = _StubGraph()
>>> graph.parse("tests/data/example.ttl")  # doctest: +SKIP
>>> len(graph)
0

### `StorageBackend`

Protocol describing the operations required by the downloader pipeline.

Attributes:
root_path: Canonical base path that implementations expose for disk
storage.  Remote-only backends can synthesize this attribute for
instrumentation purposes.

Examples:
>>> class MemoryBackend(StorageBackend):
...     root_path = Path("/tmp")  # pragma: no cover - illustrative stub
...     def prepare_version(self, ontology_id: str, version: str) -> Path:
...         ...

### `LocalStorageBackend`

Storage backend that keeps ontology artifacts on the local filesystem.

Attributes:
root: Base directory that stores ontology versions grouped by identifier.

Examples:
>>> backend = LocalStorageBackend(LOCAL_ONTOLOGY_DIR)
>>> backend.available_ontologies()
[]

### `FsspecStorageBackend`

Hybrid storage backend that mirrors artifacts to an fsspec location.

Attributes:
fs: ``fsspec`` filesystem instance used for remote operations.
base_path: Root path within the remote filesystem where artefacts live.

Examples:
>>> backend = FsspecStorageBackend("memory://ontologies")  # doctest: +SKIP
>>> backend.available_ontologies()  # doctest: +SKIP
[]

### `DownloadResult`

Result metadata for a completed download operation.

Attributes:
path: Final file path where the ontology document was stored.
status: Download status (`fresh`, `updated`, or `cached`).
sha256: SHA-256 checksum of the downloaded artifact.
etag: HTTP ETag returned by the upstream server, when available.
last_modified: Upstream last-modified header value if provided.

Examples:
>>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None)
>>> result.status
'fresh'

### `DownloadFailure`

Raised when an HTTP download attempt fails.

Attributes:
status_code: Optional HTTP status code returned by the upstream service.
retryable: Whether the failure is safe to retry with an alternate resolver.

Examples:
>>> raise DownloadFailure("Unavailable", status_code=503, retryable=True)
Traceback (most recent call last):
DownloadFailure: Unavailable

### `TokenBucket`

Token bucket used to enforce per-host and per-service rate limits.

Each unique combination of host and logical service identifier receives
its own bucket so resolvers can honour provider-specific throttling
guidance without starving other endpoints.

Attributes:
rate: Token replenishment rate per second.
capacity: Maximum number of tokens the bucket may hold.
tokens: Current token balance available for consumption.
timestamp: Monotonic timestamp of the last refill.
lock: Threading lock protecting bucket state.

Examples:
>>> bucket = TokenBucket(rate_per_sec=2.0, capacity=4.0)
>>> bucket.consume(1.0)  # consumes immediately
>>> isinstance(bucket.tokens, float)
True

### `StreamingDownloader`

Custom downloader supporting HEAD validation, conditional requests, resume, and caching.

The downloader shares a :mod:`requests` session so it can issue a HEAD probe
prior to streaming content, verifies Content-Type and Content-Length against
expectations, and persists ETag/Last-Modified headers for cache-friendly
revalidation.

Attributes:
destination: Final location where the ontology will be stored.
custom_headers: HTTP headers supplied by the resolver.
http_config: Download configuration governing retries and limits.
previous_manifest: Manifest from prior runs used for caching.
logger: Logger used for structured telemetry.
status: Final download status (`fresh`, `updated`, or `cached`).
response_etag: ETag returned by the upstream server, if present.
response_last_modified: Last-modified timestamp provided by the server.
expected_media_type: MIME type provided by the resolver for validation.

Examples:
>>> from pathlib import Path
>>> from DocsToKG.OntologyDownload import DownloadConfiguration
>>> downloader = StreamingDownloader(
...     destination=Path("/tmp/ontology.owl"),
...     headers={},
...     http_config=DownloadConfiguration(),
...     previous_manifest={},
...     logger=logging.getLogger("test"),
... )
>>> downloader.status
'fresh'

### `ValidationRequest`

Parameters describing a single validation task.

Attributes:
name: Identifier of the validator to execute.
file_path: Path to the ontology document to inspect.
normalized_dir: Directory used to write normalized artifacts.
validation_dir: Directory for validator reports and logs.
config: Resolved configuration that supplies timeout thresholds.

Examples:
>>> from pathlib import Path
>>> from DocsToKG.OntologyDownload import ResolvedConfig
>>> req = ValidationRequest(
...     name="rdflib",
...     file_path=Path("ontology.owl"),
...     normalized_dir=Path("normalized"),
...     validation_dir=Path("validation"),
...     config=ResolvedConfig.from_defaults(),
... )
>>> req.name
'rdflib'

### `ValidationResult`

Outcome produced by a validator.

Attributes:
ok: Indicates whether the validator succeeded.
details: Arbitrary metadata describing validator output.
output_files: Generated files for downstream processing.

Examples:
>>> result = ValidationResult(ok=True, details={"triples": 10}, output_files=["ontology.ttl"])
>>> result.ok
True

### `ValidationTimeout`

Raised when a validation task exceeds the configured timeout.

Args:
message: Optional description of the timeout condition.

Examples:
>>> raise ValidationTimeout("rdflib exceeded 60s")
Traceback (most recent call last):
...
ValidationTimeout: rdflib exceeded 60s

### `ValidatorSubprocessError`

Raised when a validator subprocess exits unsuccessfully.

Attributes:
message: Human-readable description of the underlying subprocess failure.

Examples:
>>> raise ValidatorSubprocessError("rdflib validator crashed")
Traceback (most recent call last):
...
ValidatorSubprocessError: rdflib validator crashed

### `OntologyDownloadError`

Base exception for ontology download failures.

Args:
message: Description of the failure encountered.

Examples:
>>> raise OntologyDownloadError("unexpected error")
Traceback (most recent call last):
...
OntologyDownloadError: unexpected error

### `ResolverError`

Raised when resolver planning fails.

Args:
message: Description of the resolver failure.

Examples:
>>> raise ResolverError("resolver unavailable")
Traceback (most recent call last):
...
ResolverError: resolver unavailable

### `ValidationError`

Raised when validation encounters unrecoverable issues.

Args:
message: Human-readable description of the validation failure.

Examples:
>>> raise ValidationError("robot validator crashed")
Traceback (most recent call last):
...
ValidationError: robot validator crashed

### `ConfigurationError`

Raised when configuration or manifest validation fails.

Args:
message: Details about the configuration inconsistency.

Examples:
>>> raise ConfigurationError("manifest missing sha256")
Traceback (most recent call last):
...
ConfigurationError: manifest missing sha256

### `FetchSpec`

Specification describing a single ontology download.

Attributes:
id: Stable identifier for the ontology to fetch.
resolver: Name of the resolver strategy used to locate resources.
extras: Resolver-specific configuration overrides.
target_formats: Normalized ontology formats that should be produced.

Examples:
>>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
>>> spec.resolver
'obo'

### `FetchResult`

Outcome of a single ontology fetch operation.

Attributes:
spec: Fetch specification that initiated the download.
local_path: Path to the downloaded ontology document.
status: Final download status (e.g., `success`, `skipped`).
sha256: SHA-256 digest of the downloaded file.
manifest_path: Path to the generated manifest JSON file.
artifacts: Ancillary files produced during extraction or validation.

Examples:
>>> from pathlib import Path
>>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
>>> result = FetchResult(
...     spec=spec,
...     local_path=Path("CHEBI.owl"),
...     status="success",
...     sha256="deadbeef",
...     manifest_path=Path("manifest.json"),
...     artifacts=(),
... )
>>> result.status
'success'

### `Manifest`

Provenance information for a downloaded ontology artifact.

Attributes:
schema_version: Manifest schema version identifier.
id: Ontology identifier recorded in the manifest.
resolver: Resolver used to retrieve the ontology.
url: Final URL from which the ontology was fetched.
filename: Local filename of the downloaded artifact.
version: Resolver-reported ontology version, if available.
license: License identifier associated with the ontology.
status: Result status reported by the downloader.
sha256: Hash of the downloaded artifact for integrity checking.
normalized_sha256: Hash of the canonical normalized TTL output.
fingerprint: Composite fingerprint combining key provenance values.
etag: HTTP ETag returned by the upstream server, when provided.
last_modified: Upstream last-modified timestamp, if supplied.
downloaded_at: UTC timestamp of the completed download.
target_formats: Desired conversion targets for normalization.
validation: Mapping of validator names to their results.
artifacts: Additional file paths generated during processing.
resolver_attempts: Ordered record of resolver attempts during download.

Examples:
>>> manifest = Manifest(
...     schema_version="1.0",
...     id="CHEBI",
...     resolver="obo",
...     url="https://example.org/chebi.owl",
...     filename="chebi.owl",
...     version=None,
...     license="CC-BY",
...     status="success",
...     sha256="deadbeef",
...     normalized_sha256=None,
...     fingerprint=None,
...     etag=None,
...     last_modified=None,
...     downloaded_at="2024-01-01T00:00:00Z",
...     target_formats=("owl",),
...     validation={},
...     artifacts=(),
...     resolver_attempts=(),
... )
>>> manifest.resolver
'obo'

### `Resolver`

Protocol describing resolver planning behaviour.

Attributes:
None

Examples:
>>> import logging
>>> spec = FetchSpec(id="CHEBI", resolver="dummy", extras={}, target_formats=("owl",))
>>> class DummyResolver:
...     def plan(self, spec, config, logger):
...         return FetchPlan(
...             url="https://example.org/chebi.owl",
...             headers={},
...             filename_hint="chebi.owl",
...             version="v1",
...             license="CC-BY",
...             media_type="application/rdf+xml",
...         )
...
>>> plan = DummyResolver().plan(spec, ResolvedConfig.from_defaults(), logging.getLogger("test"))
>>> plan.url
'https://example.org/chebi.owl'

### `ResolverCandidate`

Resolver plan captured for download-time fallback.

Attributes:
resolver: Name of the resolver that produced the plan.
plan: Concrete :class:`FetchPlan` describing how to fetch the ontology.

Examples:
>>> candidate = ResolverCandidate(
...     resolver="obo",
...     plan=FetchPlan(
...         url="https://example.org/hp.owl",
...         headers={},
...         filename_hint=None,
...         version="2024-01-01",
...         license="CC-BY",
...         media_type="application/rdf+xml",
...         service="obo",
...     ),
... )
>>> candidate.resolver
'obo'

### `PlannedFetch`

Plan describing how an ontology would be fetched without side effects.

Attributes:
spec: Original fetch specification provided by the caller.
resolver: Name of the resolver selected to satisfy the plan.
plan: Concrete :class:`FetchPlan` generated by the resolver.
candidates: Ordered list of resolver candidates available for fallback.

Examples:
>>> fetch_plan = PlannedFetch(
...     spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=("owl",)),
...     resolver="obo",
...     plan=FetchPlan(
...         url="https://example.org/hp.owl",
...         headers={},
...         filename_hint="hp.owl",
...         version="2024-01-01",
...         license="CC-BY-4.0",
...         media_type="application/rdf+xml",
...     ),
...     candidates=(
...         ResolverCandidate(
...             resolver="obo",
...             plan=FetchPlan(
...                 url="https://example.org/hp.owl",
...                 headers={},
...                 filename_hint="hp.owl",
...                 version="2024-01-01",
...                 license="CC-BY-4.0",
...                 media_type="application/rdf+xml",
...             ),
...         ),
...     ),
... )
>>> fetch_plan.resolver
'obo'

### `_StubOntology`

*No documentation available.*

### `_StubOntology`

*No documentation available.*

### `_Alarm`

Sentinel exception raised when the alarm signal fires.

Args:
message: Optional description associated with the exception.

Attributes:
message: Optional description associated with the exception.

Examples:
>>> try:
...     raise _Alarm()
... except _Alarm:
...     pass
