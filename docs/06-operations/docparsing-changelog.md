# 1. DocParsing Pipeline Changelog

## 2. 2025-03-15 — make-pydantic-required-collapse-schemas

### 2.1 Summary
* Made `pydantic>=2,<3` a hard dependency for DocParsing schema helpers by removing optional-import stubs.
* Consolidated chunk/vector schema metadata into `DocsToKG.DocParsing.formats` and introduced a temporary shim that warns on `DocsToKG.DocParsing.schemas` imports (removal planned for DocsToKG 0.3.0).
* Deleted fake Pydantic test packages, added regression coverage to prevent legacy imports, and documented the new canonical module across runbooks.

### 2.2 Migration Guide
1. Update any remaining imports to pull `ChunkRow`, `VectorRow`, and schema helpers from `DocsToKG.DocParsing.formats`.
2. Ensure deployment environments install `pydantic>=2,<3`; the runtime now raises immediately if the dependency is missing.
3. Treat `DocsToKG.DocParsing.schemas` as deprecated and schedule its removal in alignment with the 0.3.0 release.

### 2.3 Validation Checklist
* `pytest tests/docparsing/test_import_shim.py`
* `pytest tests/docparsing -k "not synthetic"`
* `python scripts/enforce_no_legacy_schema_imports.py`


## 1. 2025-03-09 — refactor-docparsing-pipeline implementation

### 1.1 Summary
* Added configurable `--model`, `--served-model-name`, and
  `--gpu-memory-utilization` flags to the PDF DocTags converter while surfacing
  the same switches through the unified CLI wrapper.
* Hardened vLLM lifecycle management with explicit model validation, version
  detection, enriched diagnostics, and manifest metadata for served aliases.
* Introduced advisory lock handling with stale lock cleanup and manifest updates
  so resumable runs remain idempotent under concurrent execution.
* Guarded SPLADE and Qwen embedding paths with actionable dependency checks and
  synthetic stubs so `--help` remains available without optional packages.
* Added a synthetic benchmarking harness covered by
  `pytest tests/docparsing/test_synthetic_benchmark.py` plus end-to-end CLI
  tests that install lightweight dependency stubs, validate schema compliance,
  and assert manifest entries.

### 1.2 Breaking Changes
* None. Existing entry points continue to work with their previous defaults.

### 1.3 Performance Benchmarks
| Scenario | Baseline | Refactored | Notes |
|----------|----------|------------|-------|
| PDF → DocTags (5 docs, Granite-Docling) | _Pending_ | _Pending_ | GPU-backed run required; see command below. |

```bash
python -m DocsToKG.DocParsing.core.cli doctags \
  --mode pdf \
  --workers 2 \
  --input <pdf_dir> \
  --output <out_dir> \
  --model <model_path>
```

> Record peak GPU memory via `nvidia-smi` while the command executes.

> **Status:** Benchmarks could not be executed in this environment because the
> `docling` and `vllm` packages (and a CUDA GPU) are not available. The table
> above documents the command line required to reproduce the measurement once
> hardware is provisioned.

### 1.4 Synthetic Streaming Benchmark

Running the synthetic harness via
`pytest tests/docparsing/test_synthetic_benchmark.py -k simulate` estimates the
following improvements:

* **Throughput:** 3.781 s → 2.193 s (≈1.72× faster)
* **Peak memory:** 5.00 MiB → 2.10 MiB (≈58 % reduction)

The harness relies solely on the new testing stubs, making it safe to execute on
development laptops without GPUs or optional DocParsing dependencies.

### 1.5 Migration Guide
1. Prefer the unified CLI wrapper:
   ```bash
   python -m DocsToKG.DocParsing.core.cli doctags --mode pdf --model /path/to/model \
       --served-model-name granite-docling-258M --served-model-name ibm-granite/granite-docling-258M
   ```
2. Update automation to consume the manifest metadata fields `model_name`, `served_models`, and `vllm_version` for auditing served model changes.
3. When running multiple converters concurrently, rely on the built-in lock handling; manual `.lock` file management is no longer necessary.

### 1.6 Validation Checklist
* `pytest tests/test_cuda_safety.py tests/test_docparsing_common.py`
* Structured logging verified manually through updated log context for vLLM startup failures.

## 3. 2025-03-20 — standardize-jsonl-locking

### 3.1 Summary
* Adopted `filelock.FileLock` for all DocParsing manifest/telemetry writes by re-implementing `core.concurrency.acquire_lock`.
* Swapped bespoke `_iter_jsonl_records` for a `jsonlines`-backed iterator that preserves `skip_invalid`, `max_errors`, `start`, and `end` semantics.
* Refactored telemetry sinks to accept an injectable writer, enabling tests to provide lock-free fakes while the default continues to call `jsonl_append_iter(..., atomic=True)`.

### 3.2 Migration Guide
1. Ensure environments install `filelock>=3.20.0` and `jsonlines>=4.0.0` (bundled with the `DocsToKG[docparse]` extra).
2. Custom telemetry integrations should pass a writer callable if they need specialised append behaviour; the default remains lock + atomic append.
3. Remove any bespoke `.lock` sentinel handling from downstream extensions—use the shared helper instead.

### 3.3 Validation Checklist
* `pytest tests/docparsing/test_concurrency_lock.py`
* `pytest tests/docparsing/test_docparsing_core.py -k telemetry`
* `pytest tests/docparsing/test_chunk_validation_telemetry.py`
* `./.venv/bin/python scripts/enforce_no_manual_docparse_locks.py`

