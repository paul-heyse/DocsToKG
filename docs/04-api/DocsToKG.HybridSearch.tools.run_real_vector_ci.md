# Module: run_real_vector_ci

CI helper to execute real-vector regression tests and collect validation artifacts.

## Functions

### `clean_directory(path)`

Remove and recreate a directory to ensure a clean workspace.

Args:
path: Directory path to reset.

Returns:
None

### `main(argv)`

Entry point for executing real-vector CI regression suites.

Args:
argv: Optional list of command-line arguments.

Returns:
Process exit code indicating success (`0`) or failure.
