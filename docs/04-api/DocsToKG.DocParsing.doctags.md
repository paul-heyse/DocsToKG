# 1. Module: doctags

This reference documents the DocsToKG module ``DocsToKG.DocParsing.doctags``.

## 1. Overview

DocParsing Doctags Pipeline Utilities

This module hosts the PDF → DocTags conversion workflow _and_ shared helpers
used by other DocParsing doctags pipelines. It coordinates vLLM server lifecycle,
manifest bookkeeping, and CLI argument scaffolding so chunking and embedding
components can import consistent behaviours.

Key Features:
- Shared CLI helpers (`add_data_root_option`, `add_resume_force_options`,
  `prepare_data_root`, `resolve_pipeline_path`) to centralise directory and
  resume/force handling.
- PDF conversion pipeline that spins up a vLLM inference server, distributes
  work across processes, and writes DocTags with manifest telemetry.
- Utility routines for manifest updates, GPU resource configuration, and
  polite rate control against vLLM endpoints.

Usage:
    from DocsToKG.DocParsing import doctags as doctags_module

    parser = doctags_module.pdf_build_parser()
    args = parser.parse_args(["--data-root", "/datasets/Data"])
    exit_code = doctags_module.pdf_main(args)

## 2. Functions

### `ensure_docling_dependencies()`

Validate that required Docling packages are installed.

### `_http_session()`

Return a shared ``requests.Session`` configured with retries.

### `add_data_root_option(parser)`

Attach the shared ``--data-root`` option to a CLI parser.

Args:
parser (argparse.ArgumentParser): Parser being configured.

Returns:
None

Examples:
>>> parser = argparse.ArgumentParser()
>>> add_data_root_option(parser)
>>> any(action.dest == "data_root" for action in parser._actions)
True

### `add_resume_force_options(parser)`

Attach ``--resume`` and ``--force`` switches to a CLI parser.

Args:
parser (argparse.ArgumentParser): Parser being configured.
resume_help (str): Help text describing resume semantics.
force_help (str): Help text describing force semantics.

Returns:
None

Examples:
>>> parser = argparse.ArgumentParser()
>>> add_resume_force_options(
...     parser,
...     resume_help="Resume processing",
...     force_help="Force reprocessing",
... )
>>> sorted(action.dest for action in parser._actions if action.option_strings)
['force', 'help', 'resume']

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

### `pdf_build_parser()`

Construct the argument parser for the PDF → DocTags converter.

Args:
None: Parser construction does not require inputs.

Returns:
Argument parser configured with all supported CLI options.

Raises:
ValueError: If parser configuration fails due to invalid defaults.

### `pdf_parse_args(argv)`

Parse CLI arguments for standalone execution.

Args:
argv: Optional CLI argument list. When ``None`` the values from
:data:`sys.argv` are used.

Returns:
Namespace containing parsed CLI options.

Raises:
SystemExit: Propagated if ``argparse`` detects invalid arguments.

### `_safe_float(value)`

Convert the supplied value to ``float`` when possible.

Args:
value: Object that may represent a numeric scalar.

Returns:
Floating point representation of ``value`` or ``0.0`` if conversion fails.

### `normalize_conversion_result(result, task)`

Validate that worker results conform to :class:`PdfConversionResult`.

Args:
result: Object returned by a worker invocation.
task: Deprecated parameter retained for API stability; ignored.

Returns:
The original :class:`PdfConversionResult` when the type check passes.

Raises:
TypeError: If the worker returned an unexpected result type.

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

### `pdf_convert_one(task)`

Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

Args:
task: Description of the conversion request, including paths and port.

Returns:
Populated :class:`PdfConversionResult` reporting success, skip, or failure.

### `pdf_main(args)`

Coordinate vLLM startup and parallel DocTags conversion.

Args:
args: Optional argument namespace injected during programmatic use.

Returns:
Process exit code, where ``0`` indicates success.

Raises:
RuntimeError: If vLLM fails to start, becomes unhealthy, or conversion
retries exhaust without success.
ValueError: If required configuration (such as auto-detected mode) is invalid.

### `html_build_parser()`

Construct an argument parser for the HTML → DocTags converter.

Args:
None: Parser initialization does not require inputs.

Returns:
Configured :class:`argparse.ArgumentParser` instance.

Raises:
None

### `html_parse_args(argv)`

Parse command-line arguments for standalone execution.

Args:
argv: Optional CLI argument vector. When ``None`` the values from
:data:`sys.argv` are used.

Returns:
Namespace containing parsed CLI options.

Raises:
SystemExit: Propagated if ``argparse`` detects invalid options.

### `_get_converter()`

Instantiate and cache a Docling HTML converter per worker process.

### `list_htmls(root)`

Enumerate HTML-like files beneath a directory tree.

Args:
root: Directory whose subtree should be searched for HTML files.

Returns:
Sorted list of discovered HTML file paths excluding normalized outputs.

### `html_convert_one(task)`

Convert a single HTML file to DocTags, honoring overwrite semantics.

Args:
task: Conversion details including paths, hash, and overwrite policy.

Returns:
:class:`ConversionResult` capturing the conversion status.

Raises:
ValueError: Propagated when Docling validation fails prior to internal handling.

### `html_main(args)`

Entrypoint for parallel HTML-to-DocTags conversion across a dataset.

Args:
args: Optional pre-parsed CLI namespace to override command-line inputs.

Returns:
Process exit code, where ``0`` denotes success.

### `_should_install_docling_test_stubs()`

Return ``True`` when docling stubs should be installed for tests.

### `_ensure_stub_module(name)`

Register a lightweight stub module in :mod:`sys.modules` if missing.

### `_install_docling_test_stubs()`

Install minimal docling/docling-core shims for unit tests.

### `from_env(cls)`

Build a configuration exclusively from environment variables.

### `from_args(cls, args)`

Create a configuration by merging env vars, optional config files, and CLI arguments.

### `finalize(self)`

Normalise derived fields after configuration sources are applied.

### `convert(self)`

Raise to indicate conversions are not executed in stub mode.

### `__repr__(self)`

*No documentation available.*

### `__hash__(self)`

*No documentation available.*

### `__eq__(self, other)`

*No documentation available.*

## 3. Classes

### `DoctagsCfg`

Structured configuration for DocTags conversion stages.

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
vlm_prompt: Prompt text passed to the VLM for PDF page conversion.
vlm_stop: Stop tokens used to terminate VLM generation.

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
...     "Convert this page to docling.",
...     ("</doctag>", "<|end_of_text|>"),
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

### `HtmlTask`

Work item describing a single HTML conversion job.

Attributes:
html_path: Absolute path to the HTML file to be converted.
relative_id: Relative identifier for manifest entries.
output_path: Destination DocTags path.
input_hash: Content hash used for resume detection.
overwrite: Flag indicating whether existing outputs should be replaced.

Examples:
>>> HtmlTask(Path("/tmp/a.html"), "doc", Path("/tmp/doc.doctags"), "hash", False)
HtmlTask(html_path=PosixPath('/tmp/a.html'), relative_id='doc', output_path=PosixPath('/tmp/doc.doctags'), input_hash='hash', overwrite=False)

### `HtmlConversionResult`

Structured result emitted by worker processes.

Attributes:
doc_id: Document identifier matching manifest entries.
status: Conversion outcome (``"success"``, ``"skip"``, or ``"failure"``).
duration_s: Time in seconds spent converting.
input_path: Source HTML path recorded for auditing.
input_hash: Content hash captured prior to conversion.
output_path: Destination DocTags path.
error: Optional error detail for failures.

Examples:
>>> HtmlConversionResult("doc", "success", 1.0, "in.html", "hash", "out.doctags")
HtmlConversionResult(doc_id='doc', status='success', duration_s=1.0, input_path='in.html', input_hash='hash', output_path='out.doctags', error=None)

### `DocumentConverter`

Stubbed ``docling`` converter preserving the real API surface.

### `PdfFormatOption`

Stub container matching ``docling`` PDF format options.

### `DoclingParseV4DocumentBackend`

Placeholder backend used when ``docling`` is unavailable.

### `AcceleratorDevice`

Minimal enum-like device sentinel for accelerator selection.

### `AcceleratorOptions`

Stub options matching the constructor signature of the real class.

### `_EnumValue`

*No documentation available.*

### `ConversionStatus`

Enum-like result state mirrors for Docling conversion outcomes.

### `InputFormat`

Enum-like input format sentinel used in Docling pipelines.

### `VlmPipelineOptions`

Catch-all structure mimicking the real VLM pipeline options.

### `ApiVlmOptions`

Stub capturing API VLM options forwarded by planners.

### `ResponseFormat`

Response format sentinel mirroring Docling's enumeration.

### `VlmPipeline`

Placeholder VLM pipeline used during tests.
