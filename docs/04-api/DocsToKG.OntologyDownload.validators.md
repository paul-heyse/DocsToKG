# Module: validators

Ontology Validation Pipeline

This module defines validators that run after ontology documents are fetched.
Each validator parses the downloaded artifact, records structured results, and
optionally emits normalized representations for downstream document processing.

## Functions

### `_log_memory(logger, validator, event)`

*No documentation available.*

### `_write_validation_json(path, payload)`

*No documentation available.*

### `_run_with_timeout(func, timeout_sec)`

*No documentation available.*

### `_prepare_xbrl_package(request, logger)`

*No documentation available.*

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

### `validate_robot(request, logger)`

Run ROBOT CLI validation and conversion workflows when available.

Args:
request: Validation request detailing ontology paths and output locations.
logger: Logger adapter for reporting warnings and CLI errors.

Returns:
ValidationResult describing generated outputs or encountered issues.

### `validate_arelle(request, logger)`

Validate XBRL ontologies with Arelle CLI if installed.

Args:
request: Validation request referencing the ontology under test.
logger: Logger used to communicate validation progress and failures.

Returns:
ValidationResult indicating whether the validation completed and
referencing any produced log files.

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

*No documentation available.*

### `_load()`

*No documentation available.*

### `_handler(signum, frame)`

*No documentation available.*

### `_execute()`

*No documentation available.*

### `parse(self, path)`

*No documentation available.*

### `__len__(self)`

*No documentation available.*

### `serialize(self, destination, format)`

*No documentation available.*

### `terms(self)`

*No documentation available.*

### `dump(self, destination, format)`

*No documentation available.*

### `classes(self)`

*No documentation available.*

### `load(self)`

*No documentation available.*

### `get_ontology(uri)`

*No documentation available.*

## Classes

### `ValidationRequest`

Parameters describing a single validation task.

Attributes:
name: Identifier of the validator to execute.
file_path: Path to the ontology document to inspect.
normalized_dir: Directory used to write normalized artifacts.
validation_dir: Directory for validator reports and logs.
config: Resolved configuration that supplies timeout thresholds.

### `ValidationResult`

Outcome produced by a validator.

Attributes:
ok: Indicates whether the validator succeeded.
details: Arbitrary metadata describing validator output.
output_files: Generated files for downstream processing.

### `ValidationTimeout`

Raised when a validation task exceeds the configured timeout.

### `_StubGraph`

*No documentation available.*

### `_StubRDFLib`

*No documentation available.*

### `_StubOntology`

*No documentation available.*

### `_StubPronto`

*No documentation available.*

### `_StubLoadedOntology`

*No documentation available.*

### `_StubOntologyWrapper`

*No documentation available.*

### `_StubOwlready2`

*No documentation available.*

### `_Alarm`

*No documentation available.*
