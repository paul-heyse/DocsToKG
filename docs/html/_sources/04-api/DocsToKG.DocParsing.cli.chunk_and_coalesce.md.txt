# 1. Module: chunk_and_coalesce

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.chunk_and_coalesce``.

Chunking CLI Wrapper

This lightweight CLI exposes the Docling hybrid chunker with DocsToKG-specific
defaults. It delegates substantive work to
``DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin`` while presenting a
user-friendly interface aligned with the wider DocParsing toolchain.

Key Features:
- Share the same argument surface as the standalone chunking script
- Provide descriptive help text for DocsToKG operators
- Enable orchestration scripts to call the chunker via ``python -m``

Usage:
    python -m DocsToKG.DocParsing.cli.chunk_and_coalesce --in-dir Data/DocTagsFiles

## 1. Functions

### `build_parser()`

Expose the chunker parser with an enhanced description.

Args:
None: Parser creation does not require inputs.

Returns:
:class:`argparse.ArgumentParser` configured for chunking CLI usage.

Raises:
None

### `main(argv)`

Parse arguments and invoke the chunking pipeline.

Args:
argv: Optional sequence of command-line arguments. When ``None`` the
values from :data:`sys.argv` are used.

Returns:
Exit code returned by the underlying chunking pipeline.

Raises:
SystemExit: Propagated when argument parsing fails.
