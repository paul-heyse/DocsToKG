# 1. Module: pdf_pipeline

This reference documents the DocsToKG module ``DocsToKG.DocParsing.pdf_pipeline``.

PDF conversion pipeline with CUDA safety guarantees.

This lightweight module exists primarily so test coverage can assert that the
DocsToKG PDF conversion entrypoint enforces the ``spawn`` multiprocessing start
method. The implementation keeps a close surface to the legacy script without
depending on heavyweight optional libraries, allowing tests to monkeypatch
behaviour as required.

## 1. Functions

### `_tqdm(iterable)`

Fallback progress iterator used when :mod:`tqdm` is unavailable.

Args:
iterable: Optional iterable passed to ``tqdm``.
**_kwargs: Additional keyword arguments ignored by the stub.

Returns:
The supplied iterable when provided; otherwise an empty list.

### `parse_args()`

Return CLI arguments for the PDF conversion pipeline.

Args:
None

Returns:
Namespace containing parsed CLI arguments.

Raises:
SystemExit: If the provided arguments fail standard argparse checks.

### `ensure_vllm(_args)`

Start the legacy VLLM service if required for the pipeline.

Args:
_args: Parsed CLI arguments (unused but retained for compatibility).

Returns:
Tuple containing the preferred port, a server handle (when started),
and a boolean indicating whether a new server boot occurred.

### `stop_vllm(server)`

Stop a VLLM server when :func:`ensure_vllm` started one.

Args:
server: Optional server handle returned by :func:`ensure_vllm`.

Returns:
None

### `list_pdfs(directory)`

Return sorted PDF paths under ``directory``.

Args:
directory: Directory expected to contain PDF artefacts.

Returns:
Sorted list of PDF file paths present in ``directory``.

### `convert_one(task)`

Convert a single PDF artefact into DocTags output (stub).

Args:
task: Tuple containing the source PDF path and the target directory.

Returns:
Tuple pairing the original PDF path with a status string.

Raises:
OSError: If the destination directory cannot be created.

### `main(args)`

Run the PDF conversion pipeline.

Args:
args: Parsed CLI arguments. When ``None`` the arguments are read from
:data:`sys.argv`.

Returns:
Process exit code where ``0`` indicates success.

### `start_vllm(self)`

Return a stubbed VLLM server handle while avoiding startup cost.

Args:
*_args: Ignored positional arguments.
**_kwargs: Ignored keyword arguments.

Returns:
None

### `wait_for_vllm(self)`

Report served model identifiers for validation routines.

Args:
*_args: Ignored positional arguments.
**_kwargs: Ignored keyword arguments.

Returns:
Sequence containing mock model identifiers.

### `validate_served_models(self)`

No-op validation stub executed after VLLM bootstrapping.

Args:
*_args: Ignored positional arguments.
**_kwargs: Ignored keyword arguments.

Returns:
None

Raises:
None.

### `manifest_append(self)`

Forward manifest writes to the shared helper during tests.

Args:
*args: Positional arguments passed to :func:`manifest_append`.
**kwargs: Keyword arguments passed to :func:`manifest_append`.

Returns:
None

## 2. Classes

### `_LegacyModule`

Mimic the legacy pdf_pipeline helper functions for test isolation.

Tests monkeypatch this container to avoid starting heavyweight services
such as VLLM while still exercising orchestration logic.

Attributes:
None: Instances expose stubbed methods only.

Examples:
>>> legacy = _LegacyModule()
>>> legacy.start_vllm() is None
True
