# 1. Module: embed_vectors

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.embed_vectors``.

Embedding CLI Wrapper

This CLI exposes the hybrid embedding pipeline that generates BM25, SPLADE, and
Qwen vectors for DocsToKG chunk files. It reuses the core parser and runtime
logic from ``DocsToKG.DocParsing.EmbeddingV2`` while enhancing descriptions for
operator-facing documentation.

Key Features:
- Share argument definitions with the primary embedding module
- Provide concise messaging suited for orchestration scripts
- Support ``python -m`` invocation without altering defaults

Usage:
    python -m DocsToKG.DocParsing.cli.embed_vectors --resume

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
