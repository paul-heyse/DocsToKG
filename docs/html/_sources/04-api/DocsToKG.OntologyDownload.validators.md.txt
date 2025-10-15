# 1. Module: validators

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.validators``.

Ontology Validation Pipeline

This module implements the post-download validation workflow that verifies
ontology integrity, generates normalized artifacts, and captures structured
telemetry for DocsToKG. Validators can leverage optional dependencies such as
rdflib, Pronto, Owlready2, ROBOT, and Arelle while falling back gracefully
when utilities are absent.

Key Features:
- Uniform :class:`ValidationRequest` / :class:`ValidationResult` data model
- Timeout and memory instrumentation for resource-intensive validators
- JSON reporting helpers compatible with automated documentation generation
- Pluggable registry enabling selective validator execution

Usage:
    from DocsToKG.OntologyDownload.validators import run_validators

    results = run_validators(requests, logger)
    print(results["rdflib"].details)

## 1. Functions

### `_log_memory(logger, validator, event)`

Emit memory usage diagnostics for a validator when debug logging is enabled.

Args:
logger: Logger responsible for validator telemetry.
validator: Name of the validator emitting the event.
event: Lifecycle label describing when the measurement is captured.

Returns:
None

### `_write_validation_json(path, payload)`

Persist structured validation metadata to disk as JSON.

Args:
path: Destination path for the JSON payload.
payload: Mapping containing validation results.

Returns:
None

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

Parse ontologies with rdflib and optionally produce Turtle output.

Args:
request: Validation request describing the target ontology and output directories.
logger: Logger adapter used for structured validation events.

Returns:
ValidationResult capturing success state, metadata, and generated files.

Raises:
ValidationTimeout: Propagated when parsing exceeds configured timeout.

### `validate_pronto(request, logger)`

Execute Pronto-based validation and emit OBO Graphs when requested.

Args:
request: Validation request describing ontology inputs and output directories.
logger: Structured logger for recording warnings and failures.

Returns:
ValidationResult with parsed ontology statistics and generated artifacts.

Raises:
ValidationTimeout: Propagated when Pronto takes longer than allowed.

### `validate_owlready2(request, logger)`

Inspect ontologies with Owlready2 to count entities and catch parsing errors.

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

### `run_validators(requests, logger)`

Execute registered validators and aggregate their results.

Args:
requests: Iterable of validation requests that specify validators to run.
logger: Logger adapter shared across validation executions.

Returns:
Mapping from validator name to the corresponding ValidationResult.

### `to_dict(self)`

Represent the validation result as a JSON-serializable dict.

Args:
None

Returns:
Dictionary with boolean status, detail payload, and output paths.

### `_parse()`

Parse the ontology with rdflib to populate the graph object.

### `_load()`

Load the ontology into memory using Pronto.

### `_handler(signum, frame)`

Signal handler converting SIGALRM into :class:`ValidationTimeout`.

Args:
signum: Received signal number.
frame: Current stack frame (unused).

### `_execute()`

Load the ontology and capture term statistics.

## 2. Classes

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
>>> from DocsToKG.OntologyDownload.config import ResolvedConfig
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

### `_Alarm`

Sentinel exception raised when the alarm signal fires.

Attributes:
message: Optional description associated with the exception.

Examples:
>>> try:
...     raise _Alarm()
... except _Alarm:
...     pass
