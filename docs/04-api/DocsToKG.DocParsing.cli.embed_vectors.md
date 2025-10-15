# 1. Module: embed_vectors

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.embed_vectors``.

Convenience CLI wrapper for the hybrid embedding pipeline.

## 1. Functions

### `build_parser()`

Return the embedding parser with a concise description.

Args:
None

Returns:
:class:`argparse.ArgumentParser` configured for embedding CLI usage.

Raises:
None

### `main(argv)`

Parse arguments and invoke the embedding pipeline.

Args:
argv: Optional sequence of command-line arguments.

Returns:
Exit code from the embedding pipeline.

Raises:
SystemExit: Propagated when argument parsing fails.
