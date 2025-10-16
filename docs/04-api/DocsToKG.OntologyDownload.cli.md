# 1. Module: cli

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.cli``.

## 1. Overview

Ontology downloader CLI entry points.

The `ontofetch` command exposes planning, pull, validation, diagnostics, and
storage management workflows described in the ontology download refactor.
Operators can override concurrency and host allowlists from the CLI, diff plan
outputs, prune historical versions, and run comprehensive `doctor` diagnostics
without editing configuration files. JSON output modes support automation while
rich ASCII tables summarise resolver fallback chains and validator results.

## 2. Functions

### `format_table(headers, rows)`

Render a padded ASCII table.

Args:
headers: Column headers rendered on the first row.
rows: Iterable of table rows; each row must match the header length.

Returns:
Formatted table as a multi-line string.

### `format_plan_rows(plans)`

Convert planner output into rows for table rendering.

Args:
plans: Planned fetch objects emitted by the planner.

Returns:
List of tuples suitable for :func:`format_table`.

### `format_results_table(results)`

Render download results as a table summarizing status and file paths.

Args:
results: Download results produced by :func:`fetch_all`.

Returns:
Formatted table summarizing ontology downloads.

### `format_validation_summary(results)`

Summarise validator outcomes in a compact status table.

Args:
results: Mapping of validator names to structured result payloads.

Returns:
Table highlighting validator status and notable detail messages.

### `_build_parser()`

Configure the top-level CLI parser and subcommands.

Returns:
Argument parser providing the consolidated ``ontofetch`` CLI.

### `_parse_target_formats(value)`

Normalize comma-separated target format strings.

Args:
value: Raw CLI argument possibly containing comma-delimited formats.

Returns:
List of stripped format identifiers, or an empty list when no formats are supplied.

### `_parse_positive_int(value)`

Parse CLI argument ensuring it is a positive integer.

Args:
value: Textual representation of a positive integer.

Returns:
Parsed positive integer value.

Raises:
argparse.ArgumentTypeError: If ``value`` is not a positive integer.

### `_parse_allowed_hosts(value)`

Split comma-delimited host allowlist argument into unique entries.

Args:
value: Raw CLI argument containing comma-delimited hostnames.

Returns:
List of unique hostnames preserving input order.

### `_normalize_plan_args(args)`

Ensure ``plan`` command defaults to the ``run`` subcommand when omitted.

Args:
args: Original CLI argument vector.

Returns:
Updated argument list with explicit subcommands injected as needed.

### `_parse_since_arg(value)`

Parse ``YYYY-MM-DD`` strings into timezone-aware datetimes.

Args:
value: Date string provided on the command line.

Returns:
Timezone-aware datetime instance aligned to UTC.

Raises:
argparse.ArgumentTypeError: If ``value`` is not a valid date.

### `_parse_since(value)`

Parse optional date input into timezone-aware datetimes.

Args:
value: Either a string date or a pre-parsed datetime.

Returns:
Normalized datetime in UTC, or ``None`` when not supplied.

### `_format_bytes(num)`

Return human-readable byte count representation.

Args:
num: Byte count to format.

Returns:
String describing the size using binary units.

### `_apply_cli_overrides(config, args)`

Mutate resolved configuration based on CLI override arguments.

Args:
config: Resolved configuration subject to mutation.
args: Parsed CLI namespace containing override values.

### `_results_to_dict(result)`

Convert a ``FetchResult`` instance into a JSON-friendly mapping.

Args:
result: Completed fetch result returned by :func:`fetch_one`.

Returns:
Dictionary containing resolver metadata, manifest location, and artifact list.

### `_compute_plan_diff(baseline, current)`

Compute additions, removals, and modifications between plan snapshots.

### `_format_plan_diff(diff)`

Render human-readable diff lines from plan comparison.

### `_plan_to_dict(plan)`

Convert a planned fetch into a JSON-friendly dictionary.

Args:
plan: Planned fetch data produced by :func:`plan_one` or :func:`plan_all`.

Returns:
Mapping containing resolver metadata and planned download details suitable
for serialization.

### `_infer_version_timestamp(version)`

Attempt to derive a datetime from a version string.

Args:
version: Version identifier emitted by upstream resolvers.

Returns:
Datetime derived from the version string, or ``None`` when no format matches.

### `_resolve_version_metadata(ontology_id, version)`

Return path, timestamp, and size metadata for a stored version.

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

### `_collect_version_metadata(ontology_id)`

Return sorted metadata entries for stored ontology versions.

Args:
ontology_id: Identifier of the ontology to inspect.

Returns:
List of metadata dictionaries ordered by most recent timestamp first.

### `_resolve_specs_from_args(args, base_config)`

Return configuration and fetch specifications derived from CLI arguments.

Args:
args: Parsed command-line arguments for `pull`/`plan` commands.
base_config: Optional pre-loaded configuration used when no spec file is supplied.

Returns:
Tuple containing the active resolved configuration and the list of fetch specs
that should be processed.

Raises:
ConfigError: If neither explicit IDs nor a configuration file are provided.

### `_handle_pull(args, base_config)`

Execute the ``pull`` subcommand workflow.

### `_handle_plan(args, base_config)`

Resolve plans without executing downloads.

### `_handle_plan_diff(args, base_config)`

Compare current plan output against a baseline plan file.

### `_handle_prune(args, logger)`

Delete surplus ontology versions based on ``--keep`` parameter.

### `_doctor_report()`

Collect diagnostic information for the ``doctor`` command.

Returns:
Mapping capturing disk, dependency, network, and configuration status.

### `_print_doctor_report(report)`

Render human-readable diagnostics from :func:`_doctor_report`.

Args:
report: Diagnostics mapping generated by :func:`_doctor_report`.

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

Raises:
ConfigError: If the target file already exists.

### `_handle_config_validate(path)`

Validate a configuration file and return a summary report.

Args:
path: Filesystem path to the configuration file under validation.

Returns:
Dictionary describing validation status, ontology count, and file path.

### `_normalize_argv(argv)`

Rewrite legacy aliases to the canonical subcommand syntax.

Args:
argv: Raw argument vector supplied by the user.

Returns:
Adjusted argument list with legacy ``plan diff`` rewired to ``plan-diff``.

### `main(argv)`

Entry point for the ontology downloader CLI.

Args:
argv: Optional argument vector supplied for testing or scripting.

Returns:
Process exit code indicating success (`0`) or failure.

Raises:
ConfigError: If configuration files are invalid or unsafe to overwrite.
OntologyDownloadError: If download or validation operations fail.

### `_format_row(values)`

*No documentation available.*
