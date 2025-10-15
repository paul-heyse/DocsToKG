# 1. Module: cli

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.cli``.

Ontology Downloader CLI

This module exposes the `ontofetch` command-line experience for DocsToKG.
It provides entry points for downloading ontologies, inspecting manifests,
re-running validators, and bootstrapping configuration files. The CLI is
designed to support both automated pipelines and human operators by offering
structured JSON output, progress tables, and detailed error reporting.

Key Features:
- Multi-command interface covering pull, show, validate, init, and config tasks
- Seamless integration with resolver planning and validation subsystems
- Support for JSON output to aid automation and downstream tooling
- Logging configuration that aligns with DocsToKG observability standards

Dependencies:
- argparse: command-line parsing
- pathlib: filesystem path handling
- DocsToKG.OntologyDownload.core: download orchestration helpers
- DocsToKG.OntologyDownload.validators: validation pipeline execution

Usage:
    from DocsToKG.OntologyDownload import cli

    if __name__ == "__main__":
        raise SystemExit(cli.main())

## 1. Functions

### `_build_parser()`

Configure the top-level CLI parser and subcommands.

Returns:
Parser instance with sub-commands for pull, show, validate, init, and config.

Raises:
None

### `_parse_target_formats(value)`

Normalize comma-separated target format strings.

Args:
value: Raw CLI argument possibly containing comma-delimited formats.

Returns:
List of stripped format identifiers, or an empty list when no formats are supplied.

### `_results_to_dict(result)`

Convert a ``FetchResult`` instance into a JSON-friendly mapping.

Args:
result: Completed fetch result returned by :func:`fetch_one`.

Returns:
Dictionary containing resolver metadata, manifest location, and artifact list.

### `_ensure_manifest_path(ontology_id, version)`

Return the manifest path for a given ontology and version.

Args:
ontology_id: Identifier for the ontology whose manifest is requested.
version: Optional version string; when omitted the latest available is used.

Returns:
Path to the manifest JSON file on disk.

Raises:
ConfigError: If the ontology or manifest cannot be located locally.

### `_load_manifest(manifest_path)`

Read and parse a manifest JSON document from disk.

Args:
manifest_path: Filesystem location of the manifest file.

Returns:
Dictionary representation of the manifest contents.

### `_handle_pull(args, base_config)`

Execute the ``pull`` subcommand workflow.

Args:
args: Parsed CLI arguments for the pull command.
base_config: Optional pre-loaded configuration used when no spec file is supplied.

Returns:
List of fetch results describing downloaded or cached ontologies.

Raises:
ConfigError: If neither ID arguments nor a configuration file are provided.

### `_handle_show(args)`

Display ontology manifest information for the ``show`` command.

Args:
args: Parsed CLI arguments including ontology identifier and output format.

Returns:
None

Raises:
ConfigError: When the manifest cannot be located.

### `_selected_validators(args)`

Determine which validators should execute based on CLI flags.

Args:
args: Parsed CLI arguments for the ``validate`` command.

Returns:
Sequence containing validator names in execution order.

### `_handle_validate(args, config)`

Run validators for a previously downloaded ontology.

Args:
args: Parsed CLI arguments specifying ontology ID, version, and output format.
config: Resolved configuration supplying validator defaults.

Returns:
Mapping of validator names to their structured result payloads.

Raises:
ConfigError: If the manifest or downloaded artifacts cannot be located.

### `_handle_init(path)`

Create a starter ``sources.yaml`` file for new installations.

Args:
path: Destination path for the generated configuration template.

Returns:
None

Raises:
ConfigError: If the target file already exists.

### `_handle_config_validate(path)`

Validate a configuration file and return a summary report.

Args:
path: Filesystem path to the configuration file under validation.

Returns:
Dictionary describing validation status, ontology count, and file path.

### `main(argv)`

Entry point for the ontology downloader CLI.

Args:
argv: Optional argument vector supplied for testing or scripting.

Returns:
Process exit code indicating success (`0`) or failure.

Raises:
ConfigError: If configuration files are invalid or unsafe to overwrite.
OntologyDownloadError: If download or validation operations fail.
