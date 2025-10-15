# 1. Module: run_docling_parallel_with_vllm_debug

This reference documents the DocsToKG module ``DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug``.

Start (or reuse) a local vLLM server for Granite-Docling, then run parallel Docling conversions.

Improvements:
- Port-smart: reuse healthy vLLM on 8000; else find another free port.
- Rich diagnostics: stream vLLM logs; print HTTP status and bodies from /v1/models and /metrics.
- tqdm progress bars for vLLM warmup and per-PDF conversion progress.

## 1. Functions

### `build_parser()`

Construct the argument parser for the PDF → DocTags converter.

Args:
None: Parser construction does not require inputs.

Returns:
Argument parser configured with all supported CLI options.

Raises:
ValueError: If parser configuration fails due to invalid defaults.

### `parse_args(argv)`

Parse CLI arguments for standalone execution.

Args:
argv: Optional CLI argument list. When ``None`` the values from
:data:`sys.argv` are used.

Returns:
Namespace containing parsed CLI options.

Raises:
SystemExit: Propagated if ``argparse`` detects invalid arguments.

### `_normalize_status(raw)`

Coerce legacy status strings into the canonical vocabulary.

### `_safe_float(value)`

Convert the supplied value to ``float`` when possible.

### `normalize_conversion_result(result, task)`

Adapt heterogeneous worker return values into :class:`PdfConversionResult`.

Historically the converter returned a tuple ``(doc_id, status)``.  The
refactor switched to ``PdfConversionResult`` which broke a regression test
that still emits tuples.  This helper accepts both shapes—as well as dicts
produced by ad-hoc stubs—and populates missing metadata from the associated
:class:`PdfTask` when available.

Args:
result: Object returned by a worker invocation.
task: Related :class:`PdfTask` used to back-fill metadata when needed.

Returns:
Normalised :class:`PdfConversionResult` instance encapsulating the
conversion outcome.

### `port_is_free(port)`

Determine whether a TCP port on localhost is currently available.

Args:
port: Port number to probe on the loopback interface.

Returns:
True when the port is unused; otherwise False.

### `probe_models(port, timeout)`

Inspect the `/v1/models` endpoint exposed by a vLLM HTTP server.

Args:
port: HTTP port where the vLLM server is expected to listen.
timeout: Seconds to wait for the HTTP request before aborting.

Returns:
Tuple containing the list of model identifiers (if any), the raw response
body, and the HTTP status code. Missing models or connection failures are
represented by `(None, <error>, None)`.

### `probe_metrics(port, timeout)`

Check whether the vLLM `/metrics` endpoint is healthy.

Args:
port: HTTP port where the vLLM server should expose metrics.
timeout: Seconds to wait for the HTTP response before aborting.

Returns:
Tuple of `(is_healthy, status_code)` where `is_healthy` is True when the
endpoint responds with HTTP 200.

### `stream_logs(proc, prefix)`

Continuously stream stdout lines from a child process to the console.

Args:
proc: Running subprocess whose stdout should be tailed.
prefix: Text prefix applied to each emitted log line for readability.

Returns:
None: This routine streams output for side effects only.

### `start_vllm(port)`

Launch a vLLM server process on the requested port.

Args:
port: Port on which the vLLM HTTP server should listen.

Returns:
Started subprocess handle for the vLLM server.

Raises:
SystemExit: If the `vllm` executable is not present on `PATH`.

### `wait_for_vllm(port, proc, timeout_s)`

Poll the vLLM server until `/v1/models` responds with success.

Args:
port: HTTP port where the server is expected to listen.
proc: Subprocess handle representing the running vLLM instance.
timeout_s: Maximum time in seconds to wait for readiness.

Returns:
None

Raises:
RuntimeError: If the server exits prematurely or fails to become ready
within the allotted timeout.

### `stop_vllm(proc, own, grace)`

Terminate a managed vLLM process if this script launched it.

Args:
proc: Subprocess handle returned by `start_vllm`, or None.
own: Indicates whether the caller owns the process lifetime.
grace: Seconds to wait for graceful shutdown before forcing exit.

Returns:
None

### `ensure_vllm(preferred)`

Ensure a vLLM server is available, launching one when necessary.

Args:
preferred: Preferred TCP port for the server.

Returns:
Tuple containing `(port, process, owns_process)` where `process` is the
managed subprocess handle (or None if reusing an existing server) and
`owns_process` indicates whether the caller should terminate it.

### `list_pdfs(root)`

Collect PDF files under a directory recursively.

Args:
root: Directory whose subtree should be scanned for PDFs.

Returns:
Sorted list of paths to PDF files.

### `convert_one(task)`

Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

Args:
task: Description of the conversion request, including paths and port.

Returns:
Populated :class:`PdfConversionResult` reporting success, skip, or failure.

Raises:
ValueError: Propagated when the underlying conversion libraries raise
validation errors prior to being caught by this helper.

### `main(args)`

Coordinate vLLM startup and parallel DocTags conversion.

Args:
args: Optional argument namespace injected during programmatic use.

Returns:
Process exit code, where ``0`` indicates success.

### `__getitem__(self, index)`

Provide tuple-like access for compatibility with legacy tests.

Args:
index: Position requested by tuple-style accessors.

Returns:
PDF path when ``index`` is ``0``.

Raises:
IndexError: If ``index`` is not ``0``.

## 2. Classes

### `PdfTask`

Work item representing a single PDF conversion request.

Attributes:
pdf_path: Absolute path to the PDF document to convert.
output_dir: Destination directory where DocTags are stored.
port: vLLM HTTP port used for remote inference.
input_hash: Content hash representing the PDF for change detection.
doc_id: Identifier derived from the PDF path for manifest entries.
output_path: Final DocTags artifact location.

Examples:
>>> task = PdfTask(Path("/tmp/sample.pdf"), Path("/tmp/out"), 8000, "hash", "doc", Path("/tmp/out/doc.doctags"))
>>> task.doc_id
'doc'

### `PdfConversionResult`

Structured result returned by worker processes.

Attributes:
doc_id: Document identifier associated with the conversion.
status: Outcome string such as ``"success"`` or ``"failure"``.
duration_s: Worker runtime in seconds.
input_path: Original PDF path recorded for manifest entries.
input_hash: Content hash used to detect stale outputs.
output_path: Location of the produced DocTags file.
error: Optional error detail captured during conversion.

Examples:
>>> PdfConversionResult("doc", "success", 1.0, "in.pdf", "hash", "out.doctags")
PdfConversionResult(doc_id='doc', status='success', duration_s=1.0, input_path='in.pdf', input_hash='hash', output_path='out.doctags', error=None)
