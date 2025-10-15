# 1. Module: run_docling_parallel_with_vllm_debug

This reference documents the DocsToKG module ``DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug``.

Legacy PDF → DocTags Converter with vLLM (DEPRECATED).

⚠️  This script is deprecated. Use the unified CLI instead:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf

This shim forwards invocations to the unified CLI for backward compatibility
and will be removed in a future release.

## 1. Functions

### `build_parser()`

Build the deprecated argument parser and forward to the unified CLI.

Returns:
Parser instance sourced from the unified CLI implementation.

Raises:
ImportError: If the unified CLI module cannot be imported.

### `parse_args(argv)`

Parse command-line arguments using the forwarded parser.

Args:
argv: Optional list of CLI arguments to parse instead of :data:`sys.argv`.

Returns:
Namespace containing CLI arguments supported by the unified command.

Raises:
SystemExit: Propagated when parsing fails.

### `main(argv)`

Forward to the unified CLI with PDF mode forced.

Args:
argv: Optional CLI argument list excluding the executable.

Returns:
Exit status returned by the unified CLI entry point.

Raises:
SystemExit: Propagated if the invoked CLI terminates.
