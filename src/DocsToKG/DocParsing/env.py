"""
Environment and path helpers for DocParsing.

This module centralises filesystem discovery, environment initialisation, and
dependency checks so that orchestrators can rely on a single location for these
concerns. Importing the module is side-effect free; directories and environment
variables are only modified when the exported functions are invoked.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

PDF_MODEL_SUBDIR = Path("granite-docling-258M")

SPLADE_DEPENDENCY_MESSAGE = (
    "Optional dependency 'sentence-transformers' is required for SPLADE embeddings. "
    "Install it with `pip install sentence-transformers` or disable SPLADE generation."
)
QWEN_DEPENDENCY_MESSAGE = (
    "Optional dependency 'vllm' is required for Qwen dense embeddings. "
    "Install it with `pip install vllm` before running the embedding pipeline."
)


def expand_path(path: str | Path) -> Path:
    """Return ``path`` expanded to an absolute :class:`Path`."""

    return Path(path).expanduser().resolve()


def resolve_hf_home() -> Path:
    """Resolve the HuggingFace cache directory respecting ``HF_HOME``."""

    env = os.getenv("HF_HOME")
    if env:
        return expand_path(env)
    return expand_path(Path.home() / ".cache" / "huggingface")


def resolve_model_root(hf_home: Optional[Path] = None) -> Path:
    """Resolve the DocsToKG model root honouring ``DOCSTOKG_MODEL_ROOT``."""

    env = os.getenv("DOCSTOKG_MODEL_ROOT")
    if env:
        return expand_path(env)
    base = hf_home if hf_home is not None else resolve_hf_home()
    return expand_path(base)


def looks_like_filesystem_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` appears to reference a local path."""

    expanded = Path(candidate).expanduser()
    drive, _ = os.path.splitdrive(candidate)
    if drive:
        return True
    if expanded.is_absolute() or expanded.exists():
        return True
    prefixes = ["~", "."]
    if os.sep not in prefixes:
        prefixes.append(os.sep)
    alt = os.altsep
    if alt and alt not in prefixes:
        prefixes.append(alt)
    return any(candidate.startswith(prefix) for prefix in prefixes)


def resolve_pdf_model_path(cli_value: str | None = None) -> str:
    """Determine PDF model path using CLI and environment precedence.

    Values that resemble filesystem paths are expanded to absolute paths while
    other identifiers (for example Hugging Face repository IDs) are returned
    verbatim so remote downloads continue to function.
    """

    if cli_value:
        if looks_like_filesystem_path(cli_value):
            return str(expand_path(cli_value))
        return cli_value
    env_model = os.getenv("DOCLING_PDF_MODEL")
    if env_model:
        if looks_like_filesystem_path(env_model):
            return str(expand_path(env_model))
        return env_model
    model_root = resolve_model_root()
    return str(expand_path(model_root / PDF_MODEL_SUBDIR))


def init_hf_env(
    hf_home: Optional[Path] = None,
    model_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Initialise Hugging Face and transformer cache environment variables."""

    resolved_hf = expand_path(hf_home) if isinstance(hf_home, Path) else resolve_hf_home()
    resolved_model_root = (
        expand_path(model_root) if isinstance(model_root, Path) else resolve_model_root(resolved_hf)
    )

    os.environ["HF_HOME"] = str(resolved_hf)
    os.environ["HF_HUB_CACHE"] = str(resolved_hf / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(resolved_hf / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(resolved_model_root)
    os.environ["DOCSTOKG_MODEL_ROOT"] = str(resolved_model_root)

    return resolved_hf, resolved_model_root


_MODEL_ENV: Tuple[Path, Path] | None = None


def _detect_cuda_device() -> str:
    """Best-effort detection of CUDA availability to choose a default device."""

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover - depends on runtime
            return "cuda"
    except Exception:  # pragma: no cover - torch missing or misconfigured
        pass
    return "cpu"


def ensure_model_environment(
    hf_home: Optional[Path] = None, model_root: Optional[Path] = None
) -> Tuple[Path, Path]:
    """Initialise and cache the HuggingFace/model-root environment settings."""

    global _MODEL_ENV
    if _MODEL_ENV is None or hf_home is not None or model_root is not None:
        _MODEL_ENV = init_hf_env(hf_home=hf_home, model_root=model_root)
    return _MODEL_ENV


def _ensure_optional_dependency(
    module_name: str, message: str, *, import_error: Exception | None = None
) -> None:
    """Import ``module_name`` or raise with ``message``."""

    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - dependency missing
        cause = import_error or exc
        raise ImportError(message) from cause


def ensure_splade_dependencies(import_error: Exception | None = None) -> None:
    """Validate that SPLADE optional dependencies are importable."""

    _ensure_optional_dependency(
        "sentence_transformers", SPLADE_DEPENDENCY_MESSAGE, import_error=import_error
    )


def ensure_qwen_dependencies(import_error: Exception | None = None) -> None:
    """Validate that Qwen/vLLM optional dependencies are importable."""

    _ensure_optional_dependency("vllm", QWEN_DEPENDENCY_MESSAGE, import_error=import_error)


def ensure_splade_environment(
    *, device: Optional[str] = None, cache_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Bootstrap SPLADE-related environment defaults and return resolved settings."""

    resolved_device = (
        device
        or os.getenv("DOCSTOKG_SPLADE_DEVICE")
        or os.getenv("SPLADE_DEVICE")
        or _detect_cuda_device()
    )
    os.environ.setdefault("DOCSTOKG_SPLADE_DEVICE", resolved_device)
    os.environ.setdefault("SPLADE_DEVICE", resolved_device)

    env_info: Dict[str, str] = {"device": resolved_device}

    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
        os.environ.setdefault("DOCSTOKG_SPLADE_MODEL_DIR", str(cache_path))
        env_info["model_dir"] = str(cache_path)

    return env_info


def ensure_qwen_environment(
    *, device: Optional[str] = None, dtype: Optional[str] = None, model_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Bootstrap Qwen/vLLM environment defaults and return resolved settings."""

    resolved_device = (
        device
        or os.getenv("DOCSTOKG_QWEN_DEVICE")
        or os.getenv("VLLM_DEVICE")
        or _detect_cuda_device()
    )
    os.environ.setdefault("DOCSTOKG_QWEN_DEVICE", resolved_device)
    os.environ.setdefault("VLLM_DEVICE", resolved_device)

    resolved_dtype = dtype or os.getenv("DOCSTOKG_QWEN_DTYPE") or "bfloat16"
    os.environ.setdefault("DOCSTOKG_QWEN_DTYPE", resolved_dtype)

    env_info: Dict[str, str] = {"device": resolved_device, "dtype": str(resolved_dtype)}

    if model_dir is not None:
        model_path = Path(model_dir).expanduser().resolve()
        os.environ.setdefault("DOCSTOKG_QWEN_MODEL_DIR", str(model_path))
        env_info["model_dir"] = str(model_path)

    return env_info


def detect_data_root(start: Optional[Path] = None) -> Path:
    """Locate the DocsToKG Data directory via env var or ancestor scan."""

    env_root = os.getenv("DOCSTOKG_DATA_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        return env_path

    start_path = Path.cwd() if start is None else Path(start).resolve()
    expected_dirs = ["PDFs", "HTML", "DocTagsFiles", "ChunkedDocTagFiles"]
    for ancestor in [start_path, *start_path.parents]:
        candidate = ancestor / "Data"
        if any((candidate / directory).is_dir() for directory in expected_dirs):
            return candidate.resolve()

    return (start_path / "Data").resolve()


def _ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return its absolute form."""

    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _resolve_data_path(root: Optional[Path], name: str) -> Path:
    """Resolve ``name`` relative to the DocsToKG data root without creating it."""

    if root is not None:
        base = Path(root).expanduser().resolve()
    else:
        base = detect_data_root()
    return (base / name).resolve()


def data_doctags(root: Optional[Path] = None, *, ensure: bool = True) -> Path:
    """Return the DocTags directory path relative to the data root.

    Args:
        root: Optional override for the data root.
        ensure: When ``True`` (default) the directory is created if missing.
    """

    path = _resolve_data_path(root, "DocTagsFiles")
    return _ensure_dir(path) if ensure else path


def data_chunks(root: Optional[Path] = None, *, ensure: bool = True) -> Path:
    """Return the chunk directory path relative to the data root."""

    path = _resolve_data_path(root, "ChunkedDocTagFiles")
    return _ensure_dir(path) if ensure else path


def data_vectors(root: Optional[Path] = None, *, ensure: bool = True) -> Path:
    """Return the vector directory path relative to the data root."""

    path = _resolve_data_path(root, "Embeddings")
    return _ensure_dir(path) if ensure else path


def data_manifests(root: Optional[Path] = None, *, ensure: bool = True) -> Path:
    """Return the manifest directory path relative to the data root."""

    path = _resolve_data_path(root, "Manifests")
    return _ensure_dir(path) if ensure else path


def prepare_data_root(data_root_arg: Optional[Path], default_root: Path) -> Path:
    """Resolve and prepare the DocsToKG data root for a pipeline invocation."""

    if data_root_arg is not None:
        resolved = Path(data_root_arg).expanduser().resolve()
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved)
    else:
        resolved = default_root
    _ensure_dir((resolved / "Manifests").resolve())
    return resolved


def resolve_pipeline_path(
    *,
    cli_value: Optional[Path],
    default_path: Path,
    resolved_data_root: Path,
    data_root_overridden: bool,
    resolver: Callable[[Path], Path],
) -> Path:
    """Derive a pipeline directory path respecting data-root overrides."""

    if data_root_overridden and (cli_value is None or cli_value == default_path):
        return resolver(resolved_data_root)
    if cli_value is None:
        return default_path
    return cli_value


def data_pdfs(root: Optional[Path] = None, *, ensure: bool = True) -> Path:
    """Return the PDFs directory path relative to the data root."""

    path = _resolve_data_path(root, "PDFs")
    return _ensure_dir(path) if ensure else path


def data_html(root: Optional[Path] = None, *, ensure: bool = True) -> Path:
    """Return the HTML directory path relative to the data root."""

    path = _resolve_data_path(root, "HTML")
    return _ensure_dir(path) if ensure else path


__all__ = [
    "PDF_MODEL_SUBDIR",
    "detect_data_root",
    "data_chunks",
    "data_doctags",
    "data_html",
    "data_manifests",
    "data_pdfs",
    "data_vectors",
    "ensure_model_environment",
    "ensure_qwen_dependencies",
    "ensure_qwen_environment",
    "ensure_splade_dependencies",
    "ensure_splade_environment",
    "expand_path",
    "init_hf_env",
    "looks_like_filesystem_path",
    "prepare_data_root",
    "resolve_hf_home",
    "resolve_model_root",
    "resolve_pdf_model_path",
    "resolve_pipeline_path",
]
