# 1. Module: run_docling_parallel_with_vllm_debug

This reference documents the DocsToKG module ``DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug``.

Parallel PDF → DocTags Conversion with vLLM Orchestration

This module launches (or reuses) a local vLLM server and executes Docling PDF
conversions in parallel worker processes. It provides resilience features such
as automatic port selection, manifest-aware resume semantics, and detailed
logging for observability.

Key Features:
- Automatically reuse an existing healthy vLLM instance or launch a new one
- Stream structured log output and metrics for debugging and monitoring
- Populate DocsToKG manifests with success, skip, and failure records
- Coordinate multiprocessing workers while respecting GPU resource limits

Usage:
    python -m DocsToKG.DocParsing.run_docling_parallel_with_vllm_debug         --input Data/PDFs --workers 4

Dependencies:
- vllm (optional): Provides the Granite-Docling model served via HTTP.
- requests: Probe vLLM readiness and metrics endpoints.
- tqdm: Display warmup and conversion progress bars.

## 1. Functions

### `_dedupe_preserve_order(names)`

Return a list containing ``names`` without duplicates while preserving order.

Args:
names: Iterable of candidate names that may include duplicates or empty values.

Returns:
List of unique names in their original encounter order.

Examples:
>>> _dedupe_preserve_order(["a", "b", "a", "c"])
['a', 'b', 'c']

### `_normalize_served_model_names(raw)`

Flatten CLI-provided served model names into a deduplicated tuple.

Args:
raw: Sequence containing strings or nested iterables of strings sourced from
CLI options.

Returns:
Tuple of unique served model names with defaults applied when none were given.

Examples:
>>> _normalize_served_model_names([["a", "b"], "b"])
('a', 'b')

### `detect_vllm_version()`

Detect the installed vLLM package version for diagnostics.

Args:
None

Returns:
Version string reported by the local vLLM installation or ``"unknown"`` when
the package cannot be imported.

### `validate_served_models(available, expected)`

Ensure that at least one of the expected served model names is available.

Args:
available: Model aliases exposed by the running vLLM server.
expected: Tuple of acceptable model aliases configured for conversion.

Returns:
None

Raises:
RuntimeError: If none of the expected model names are present.

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

Args:
raw: Status string emitted by historical workers or manifests.

Returns:
Canonical status string (``"success"``, ``"skip"``, or ``"failure"``).

### `_safe_float(value)`

Convert the supplied value to ``float`` when possible.

Args:
value: Object that may represent a numeric scalar.

Returns:
Floating point representation of ``value`` or ``0.0`` if conversion fails.

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

### `stream_logs(proc, prefix, tail)`

Continuously stream stdout lines from a child process to the console.

Args:
proc: Running subprocess whose stdout should be tailed.
prefix: Text prefix applied to each emitted log line for readability.
tail: Optional deque that accumulates the most recent log lines.

Returns:
None: This routine streams output for side effects only.

### `start_vllm(port, model_path, served_model_names, gpu_memory_utilization)`

Launch a vLLM server process on the requested port.

Args:
port: Port on which the vLLM HTTP server should listen.
model_path: Local directory or HF repository containing model weights.
served_model_names: Aliases registered with the OpenAI-compatible API.
gpu_memory_utilization: Fraction of GPU memory the server may allocate.

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
Model names reported by the server upon readiness.

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
None.

### `ensure_vllm(preferred, model_path, served_model_names, gpu_memory_utilization)`

Ensure a vLLM server is available, launching one when necessary.

Args:
preferred: Preferred TCP port for the server.
model_path: Model repository or path passed to the vLLM CLI.
served_model_names: Aliases that should be exposed via the OpenAI API.
gpu_memory_utilization: Fractional GPU memory reservation for the server.

Returns:
Tuple containing `(port, process, owns_process)` where `process` is the
managed subprocess handle (or None if reusing an existing server) and
`owns_process` indicates whether the caller should terminate it.

Raises:
RuntimeError: If the launched or reused server does not expose the expected
model aliases, indicating a misconfiguration.

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

### `main(args)`

Coordinate vLLM startup and parallel DocTags conversion.

Args:
args: Optional argument namespace injected during programmatic use.

Returns:
Process exit code, where ``0`` indicates success.

Raises:
RuntimeError: If vLLM fails to start, becomes unhealthy, or conversion
retries exhaust without success.
ValueError: If required configuration (such as auto-detected mode) is invalid.

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
served_model_names: Collection of aliases configured for the vLLM server.
inference_model: Primary model name used when issuing chat completions.

Examples:
>>> task = PdfTask(
...     Path("/tmp/sample.pdf"),
...     Path("/tmp/out"),
...     8000,
...     "hash",
...     "doc",
...     Path("/tmp/out/doc.doctags"),
...     ("granite-docling-258M",),
...     "granite-docling-258M",
... )
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
