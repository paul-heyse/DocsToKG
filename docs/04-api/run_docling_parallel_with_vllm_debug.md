# Module: run_docling_parallel_with_vllm_debug

Start (or reuse) a local vLLM server for Granite-Docling, then run parallel Docling conversions.

Improvements:
- Port-smart: reuse healthy vLLM on 8000; else find another free port.
- Rich diagnostics: stream vLLM logs; print HTTP status and bodies from /v1/models and /metrics.
- tqdm progress bars for vLLM warmup and per-PDF conversion progress.

## Functions

### `find_data_root(start)`

Walk up from `start` to find a directory that has Data/PDFs.
Returns `start/"Data"` if nothing matches above.

### `port_is_free(port)`

*No documentation available.*

### `probe_models(port, timeout)`

Return (names, raw_text, status) from GET /v1/models, or (None, raw, status) on failure.

### `probe_metrics(port, timeout)`

Return (ok, status) from /metrics; OK if 200.

### `find_free_port(start, span)`

*No documentation available.*

### `stream_logs(proc, prefix)`

Continuously read vLLM stdout and print lines.

### `start_vllm(port)`

*No documentation available.*

### `wait_for_vllm(port, proc, timeout_s)`

*No documentation available.*

### `stop_vllm(proc, own, grace)`

*No documentation available.*

### `ensure_vllm(preferred)`

Return (port, process, owns_process).

### `list_pdfs(root)`

*No documentation available.*

### `convert_one(args)`

*No documentation available.*

### `main()`

*No documentation available.*
