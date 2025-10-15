# 1. Module: html_pipeline

This reference documents the DocsToKG module ``DocsToKG.DocParsing.html_pipeline``.

Parallel HTML → DocTags conversion pipeline.

Implements Docling HTML conversions across multiple processes while tracking
manifests, resume/force semantics, and advisory file locks. The pipeline is
used by the DocsToKG CLI to transform raw HTML corpora into DocTags ready for
chunking and embedding.

## 1. Functions

### `build_parser()`

Construct an argument parser for the HTML → DocTags converter.

### `parse_args(argv=None)`

Parse command-line arguments for standalone execution.

### `main(args=None)`

Entrypoint for parallel HTML-to-DocTags conversion across a dataset.
