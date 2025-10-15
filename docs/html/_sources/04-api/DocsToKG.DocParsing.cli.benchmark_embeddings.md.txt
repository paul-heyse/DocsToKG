# 1. Module: benchmark_embeddings

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.benchmark_embeddings``.

DocParsing Embedding Benchmark CLI

This command-line interface estimates the performance benefits of the streaming
embedding pipeline relative to the legacy whole-corpus workflow. It relies on
synthetic, deterministic inputs to provide fast feedback without requiring the
full DocsToKG environment.

Key Features:
- Parameterise chunk counts, token lengths, and dense vector dimensionality
- Leverage testing utilities to model runtime and memory characteristics
- Output a ready-to-share textual summary for performance reports

Usage:
    python -m DocsToKG.DocParsing.cli.benchmark_embeddings --chunks 1024 --tokens 512

Dependencies:
- argparse: Parse command-line options exposed by the CLI.
- DocsToKG.DocParsing.testing: Provides simulation primitives used under the hood.

## 1. Functions

### `build_parser()`

Construct the CLI parser for the synthetic benchmark harness.

Args:
None: Parser creation does not require inputs.

Returns:
:class:`argparse.ArgumentParser` configured with benchmark options.

Raises:
None

### `main(argv)`

Execute the synthetic benchmark and emit a human-friendly summary.

Args:
argv: Optional sequence of command-line arguments. When ``None`` the
values from :data:`sys.argv` are used.

Returns:
Exit code where ``0`` indicates the benchmark completed successfully.

Raises:
SystemExit: Propagated if argument parsing fails.
