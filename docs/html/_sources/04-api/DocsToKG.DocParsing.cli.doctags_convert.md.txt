# 1. Module: doctags_convert

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.doctags_convert``.

Unified CLI for converting HTML or PDF corpora into DocTags.

## 1. Functions

### `build_parser()`

Construct the unified DocTags conversion argument parser.

Args:
None: Parser creation does not require inputs.

Returns:
:class:`argparse.ArgumentParser` populated with DocTags CLI options.

Raises:
None

### `detect_mode(input_dir)`

Inspect ``input_dir`` and infer conversion mode based on file types.

Args:
input_dir: Directory whose contents determine the appropriate backend.

Returns:
``"pdf"`` or ``"html"`` depending on the detected file extensions.

Raises:
ValueError: If both PDF and HTML files are present (or neither).

### `_merge_args(parser, overrides)`

*No documentation available.*

### `main(argv)`

Dispatch conversion to the HTML or PDF backend based on requested mode.

Args:
argv: Optional CLI arguments; defaults to :data:`sys.argv` when omitted.

Returns:
Exit code from the selected conversion backend.

Raises:
ValueError: If conversion mode cannot be determined automatically.
