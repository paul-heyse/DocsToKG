# Module: run_docling_html_to_doctags_parallel

HTML → DocTags (parallel, CPU-only; no captioning/classification, no HF auth)

- Input : Data/HTML/ (recurses; excludes *.normalized.html)
- Output: Data/DocTagsFiles/<mirrored_subdirs>/*.doctags

Example:
  python run_docling_html_to_doctags_parallel.py       --input  Data/HTML       --output Data/DocTagsFiles       --workers 12

## Functions

### `_get_converter()`

Instantiate one converter per worker process.

### `detect_data_root(start)`

*No documentation available.*

### `list_htmls(root)`

*No documentation available.*

### `convert_one(html_path, input_root, output_root, overwrite)`

Returns (relative_path, status) where status in {'ok','skip','fail: ...'}.

### `main()`

*No documentation available.*
