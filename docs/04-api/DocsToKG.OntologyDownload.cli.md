# Module: cli

Command-line interface for the ontology downloader.

## Functions

### `_build_parser()`

*No documentation available.*

### `_parse_target_formats(value)`

*No documentation available.*

### `_results_to_dict(result)`

*No documentation available.*

### `_format_table(headers, rows)`

*No documentation available.*

### `_format_validation_summary(results)`

*No documentation available.*

### `_ensure_manifest_path(ontology_id, version)`

*No documentation available.*

### `_load_manifest(manifest_path)`

*No documentation available.*

### `_handle_pull(args, base_config)`

*No documentation available.*

### `_handle_show(args)`

*No documentation available.*

### `_selected_validators(args)`

*No documentation available.*

### `_handle_validate(args, config)`

*No documentation available.*

### `_handle_init(path)`

*No documentation available.*

### `_handle_config_validate(path)`

*No documentation available.*

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
