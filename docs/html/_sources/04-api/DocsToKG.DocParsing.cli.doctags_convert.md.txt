# 1. Module: doctags_convert

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli.doctags_convert``.

Unified DocTags Conversion CLI

This command-line interface orchestrates HTML and PDF conversions to DocTags
using Docling backends. It consolidates disparate scripts into a single entry
point that auto-detects the appropriate backend, manages manifests, and shares
DocsToKG-wide defaults.

Key Features:
- Auto-detect the conversion backend from input directory contents
- Forward CLI options to specialised HTML and PDF pipelines
- Integrate with DocsToKG resume/force semantics for idempotent runs

Usage:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode auto --input Data/HTML

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

Examples:
>>> tmp = Path("/tmp/docstokg-cli-examples")
>>> _ = tmp.mkdir(exist_ok=True)
>>> _ = (tmp / "example.html").write_text("<html></html>", encoding="utf-8")
>>> detect_mode(tmp)
'html'

### `_merge_args(parser, overrides)`

Return a parser namespace seeded with override values.

Args:
parser: Parser whose default values should seed the namespace.
overrides: Mapping of argument names to explicit override values.

Returns:
:class:`argparse.Namespace` with defaults populated and overrides applied.

### `main(args)`

Dispatch conversion to the HTML or PDF backend based on requested mode.

Args:
args: Either an :class:`argparse.Namespace`, a list of CLI arguments, or ``None``.

Returns:
Exit code from the selected conversion backend.

Raises:
ValueError: If conversion mode cannot be determined automatically.
