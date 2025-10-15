# 1. Module: run_docling_html_to_doctags_parallel

This reference documents the DocsToKG module ``DocsToKG.DocParsing.run_docling_html_to_doctags_parallel``.

Legacy HTML → DocTags Converter (DEPRECATED).

⚠️  This script is deprecated. Use the unified CLI instead:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode html

This shim forwards invocations to the unified CLI for backward compatibility
and will be removed in a future release.

## 1. Functions

### `build_parser()`

Build argument parser (deprecated, forwards to unified CLI).

### `parse_args(argv)`

Parse command-line arguments (deprecated).

### `main(argv)`

Forward to unified CLI with HTML mode forced.
