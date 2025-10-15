# 1. Module: chunk_and_coalesce

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.chunk_and_coalesce``.

Convenience CLI wrapper for the DocTags chunking pipeline.

## 1. Functions

### `build_parser()`

Expose the chunker parser with an enhanced description.

Args:
None

Returns:
:class:`argparse.ArgumentParser` configured for chunking CLI usage.

Raises:
None

### `main(argv)`

Parse arguments and invoke the chunking pipeline.

Args:
argv: Optional sequence of command-line arguments.

Returns:
Exit code from the chunking pipeline.

Raises:
SystemExit: Propagated when argument parsing fails.
