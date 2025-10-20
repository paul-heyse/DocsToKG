# 1. Module: env

This reference documents the DocsToKG module ``DocsToKG.DocParsing.env``.

## 1. Overview

Environment and path helpers for DocParsing.

This module centralises filesystem discovery, environment initialisation, and
dependency checks so that orchestrators can rely on a single location for these
concerns. Importing the module is side-effect free; directories and environment
variables are only modified when the exported functions are invoked.

## 2. Functions

### `expand_path(path)`

Return ``path`` expanded to an absolute :class:`Path`.

### `resolve_hf_home()`

Resolve the HuggingFace cache directory respecting ``HF_HOME``.

### `resolve_model_root(hf_home)`

Resolve the DocsToKG model root honouring ``DOCSTOKG_MODEL_ROOT``.

### `looks_like_filesystem_path(candidate)`

Return ``True`` when ``candidate`` appears to reference a local path.

### `resolve_pdf_model_path(cli_value)`

Determine PDF model path using CLI and environment precedence.

### `init_hf_env(hf_home, model_root)`

Initialise Hugging Face and transformer cache environment variables.

### `_detect_cuda_device()`

Best-effort detection of CUDA availability to choose a default device.

### `ensure_model_environment(hf_home, model_root)`

Initialise and cache the HuggingFace/model-root environment settings.

### `_ensure_optional_dependency(module_name, message)`

Import ``module_name`` or raise with ``message``.

### `ensure_splade_dependencies(import_error)`

Validate that SPLADE optional dependencies are importable.

### `ensure_qwen_dependencies(import_error)`

Validate that Qwen/vLLM optional dependencies are importable.

### `ensure_splade_environment()`

Bootstrap SPLADE-related environment defaults and return resolved settings. When a cache directory is provided the helper seeds both ``DOCSTOKG_SPLADE_DIR`` and the legacy ``DOCSTOKG_SPLADE_MODEL_DIR`` variables with the resolved path.

### `ensure_qwen_environment()`

Bootstrap Qwen/vLLM environment defaults and return resolved settings.

### `detect_data_root(start)`

Locate the DocsToKG Data directory via env var or ancestor scan.

### `_ensure_dir(path)`

Create ``path`` if needed and return its absolute form.

### `_resolve_data_path(root, name)`

Resolve ``name`` relative to the DocsToKG data root without creating it.

### `data_doctags(root)`

Return the DocTags directory path relative to the data root.

Args:
root: Optional override for the data root.
ensure: When ``True`` (default) the directory is created if missing.

### `data_chunks(root)`

Return the chunk directory path relative to the data root.

### `data_vectors(root)`

Return the vector directory path relative to the data root.

### `data_manifests(root)`

Return the manifest directory path relative to the data root.

### `prepare_data_root(data_root_arg, default_root)`

Resolve and prepare the DocsToKG data root for a pipeline invocation.

### `resolve_pipeline_path()`

Derive a pipeline directory path respecting data-root overrides.

### `data_pdfs(root)`

Return the PDFs directory path relative to the data root.

### `data_html(root)`

Return the HTML directory path relative to the data root.
