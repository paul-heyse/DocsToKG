# 1. Module: validators

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.validators``.

## 1. Overview

The standalone module ``DocsToKG.OntologyDownload.validators`` has been retired. The
validation pipeline now lives in ``DocsToKG.OntologyDownload.ontology_download`` and
is exposed through the consolidated ``DocsToKG.OntologyDownload`` facade.

This module implements the post-download workflow that verifies ontologies,
normalizes output, and collects structured telemetry for DocsToKG. Validators
support streaming normalization for large ontologies, deterministic hashing for
manifest fingerprints, and optional dependency fallbacks for tools such as
rdflib, Pronto, Owlready2, ROBOT, and Arelle.

### Plugin registration example

To add bespoke validation steps, publish an entry point under
``docstokg.ontofetch.validator``. A minimal plugin might look like:

```python
import json

from DocsToKG.OntologyDownload import ValidationResult


def validate_custom(request, logger):
    report = request.validation_dir / "custom.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ok": True, "validator": request.name}
    report.write_text(json.dumps(payload))
    return ValidationResult(ok=True, details=payload, output_files=[str(report)])
```

Declare the plugin in ``pyproject.toml``:

```toml
[project.entry-points."docstokg.ontofetch.validator"]
custom = "my_package.validators:validate_custom"
```

After installation the validator participates in ``run_validators`` alongside
the built-in adapters.

## 2. Functions

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

### `_python_merge_sort(source, destination)`

Sort an N-Triples file using a disk-backed merge strategy.

### `_term_to_string(term, namespace_manager)`

*No documentation available.*

### `_canonicalize_turtle(graph)`

Return canonical Turtle output with sorted prefixes and triples.

The canonical form mirrors the ontology downloader specification by sorting
prefixes lexicographically and emitting triples ordered by subject,
predicate, and object so downstream hashing yields deterministic values.

### `_canonicalize_blank_nodes_line(line, mapping)`

Replace blank node identifiers with deterministic sequential labels.

### `_sort_triple_file(source, destination)`

Sort serialized triple lines using platform sort when available.

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

### `_replace(match)`

*No documentation available.*

### `_iter_canonical_lines()`

*No documentation available.*

### `_flush(writer)`

*No documentation available.*

### `_parse()`

Parse the ontology with rdflib to populate the graph object.

### `_handler(signum, frame)`

Signal handler converting SIGALRM into :class:`ValidationTimeout`.

Args:
signum: Received signal number.
frame: Current stack frame (unused).

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

### `ValidatorSubprocessError`

Raised when a validator subprocess exits unsuccessfully.

Attributes:
message: Human-readable description of the underlying subprocess failure.

Examples:
>>> raise ValidatorSubprocessError("rdflib validator crashed")
Traceback (most recent call last):
...
ValidatorSubprocessError: rdflib validator crashed

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
