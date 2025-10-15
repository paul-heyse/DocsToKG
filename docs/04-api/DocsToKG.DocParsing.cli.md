# 1. Module: cli

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli``.

Unified DocParsing command-line interface.

This module consolidates the individual DocParsing CLIs into a single entry
point with subcommands. Invoke it with:

    python -m DocsToKG.DocParsing.cli <command> [options...]

Available commands:
    - chunk:     Run the Docling hybrid chunker.
    - embed:     Generate BM25, SPLADE, and dense vectors for chunks.
    - doctags:   Convert HTML/PDF corpora into DocTags.

## 1. Functions

### `_run_chunk(argv)`

Execute the Docling chunker subcommand.

Args:
argv: Argument vector forwarded to the chunker parser.

Returns:
Process exit code produced by the Docling chunker pipeline.

### `_run_embed(argv)`

Execute the embedding pipeline subcommand.

Args:
argv: Argument vector forwarded to the embedding parser.

Returns:
Process exit code produced by the embedding pipeline.

### `_build_doctags_parser(prog)`

Create an :mod:`argparse` parser configured for DocTags conversion.

Args:
prog: Program name displayed in help output.

Returns:
Argument parser instance for the ``doctags`` subcommand.

### `_detect_mode(input_dir)`

Infer conversion mode based on the contents of ``input_dir``.

Args:
input_dir: Directory searched for PDF and HTML inputs.

Returns:
``"pdf"`` when only PDFs are present, ``"html"`` when only HTML files exist.

Raises:
ValueError: If both formats are present or neither can be detected.

### `_merge_args(parser, overrides)`

Merge override values into the default parser namespace.

Args:
parser: Parser providing default argument values.
overrides: Mapping of argument names to replacement values.

Returns:
Namespace populated with defaults and supplied overrides.

### `_run_doctags(argv)`

Execute the DocTags conversion subcommand.

Args:
argv: Argument vector provided by the CLI dispatcher.

Returns:
Process exit code from the selected DocTags backend.

### `main(argv)`

Dispatch to one of the DocParsing subcommands.

Args:
argv: Optional argument vector supplied programmatically.

Returns:
Process exit code returned by the selected subcommand.

### `chunk(argv)`

Programmatic helper mirroring ``docparse chunk``.

Args:
argv: Optional argument vector supplied for testing.

Returns:
Process exit code returned by the chunker pipeline.

### `embed(argv)`

Programmatic helper mirroring ``docparse embed``.

Args:
argv: Optional argument vector supplied for testing.

Returns:
Process exit code returned by the embedding pipeline.

### `doctags(argv)`

Programmatic helper mirroring ``docparse doctags``.

Args:
argv: Optional argument vector supplied for testing.

Returns:
Process exit code returned by the DocTags conversion pipeline.

## 2. Classes

### `_Command`

Callable wrapper storing handler metadata for subcommands.

Attributes:
handler: Callable invoked with the subcommand argument vector.
help: Short help text displayed in CLI usage.

Examples:
>>> cmd = _Command(_run_chunk, "Run the chunker")
>>> cmd.handler([])  # doctest: +SKIP
0
