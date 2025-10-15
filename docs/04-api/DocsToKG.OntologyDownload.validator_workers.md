# 1. Module: validator_workers

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.validator_workers``.

Subprocess workers for memory-intensive ontology validators.

These helpers execute within isolated Python interpreters to release
memory promptly after heavy validation steps such as Pronto and
Owlready2 parsing. Each worker reads a JSON payload from ``stdin`` and
emits a JSON document to ``stdout`` describing the validation result.

The module is intentionally lightweight so spawning subprocesses remains
fast. It relies on the optional dependency accessors in
``DocsToKG.OntologyDownload.optdeps`` which transparently provide stub
implementations during testing when the real dependencies are not
installed.

## 1. Functions

### `_run_pronto(payload)`

Execute Pronto validation logic in a subprocess context and emit JSON.

Args:
payload: Mapping containing ``file_path`` of the ontology and optional
``normalized_path`` where serialized output should be written.

Returns:
Dictionary describing the validation outcome, including metrics such as
``terms`` and an ``ok`` flag.

### `_run_owlready2(payload)`

Execute Owlready2 validation logic in a subprocess context and emit JSON.

Args:
payload: Mapping containing ``file_path`` that should be parsed by Owlready2.

Returns:
Dictionary containing validation status metadata such as entity counts.

### `main()`

Parse command line arguments and execute the requested worker.

Args:
None

Returns:
None
