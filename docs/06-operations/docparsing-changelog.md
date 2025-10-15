# DocParsing Pipeline Changelog

## 2025-03-09 — refactor-docparsing-pipeline implementation

### Summary
* Added configurable `--model`, `--served-model-name`, and `--gpu-memory-utilization` flags to the PDF DocTags converter while surfacing the same switches through the unified CLI wrapper.
* Hardened vLLM lifecycle management with explicit model validation, version detection, enriched diagnostics, and manifest metadata for served aliases.
* Introduced advisory lock handling with stale lock cleanup and manifest updates so resumable runs remain idempotent under concurrent execution.

### Breaking Changes
* None. Existing entry points continue to work with their previous defaults.

### Performance Benchmarks
| Scenario | Baseline | Refactored | Notes |
|----------|----------|------------|-------|
| PDF → DocTags (5 docs, Granite-Docling) | _Pending_ | _Pending_ | GPU-backed run required; execute `python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf --workers 2 --input <pdf_dir> --output <out_dir> --model <model_path>` on a CUDA host and record peak GPU memory via `nvidia-smi`. |

> **Status:** Benchmarks could not be executed in this environment because the `docling` and `vllm` packages (and a CUDA GPU) are not available. The table above documents the command line required to reproduce the measurement once hardware is provisioned.

### Migration Guide
1. Prefer the unified CLI wrapper:
   ```bash
   python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf --model /path/to/model \
       --served-model-name granite-docling-258M --served-model-name ibm-granite/granite-docling-258M
   ```
2. Update automation to consume the manifest metadata fields `model_name`, `served_models`, and `vllm_version` for auditing served model changes.
3. When running multiple converters concurrently, rely on the built-in lock handling; manual `.lock` file management is no longer necessary.

### Validation Checklist
* `pytest tests/test_cuda_safety.py tests/test_docparsing_common.py`
* Structured logging verified manually through updated log context for vLLM startup failures.
