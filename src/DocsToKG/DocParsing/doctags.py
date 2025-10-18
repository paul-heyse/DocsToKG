# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.doctags",
#   "purpose": "Implements DocsToKG.DocParsing.doctags behaviors and helpers",
#   "sections": [
#     {
#       "id": "doctagscfg",
#       "name": "DoctagsCfg",
#       "anchor": "class-doctagscfg",
#       "kind": "class"
#     },
#     {
#       "id": "ensure-docling-dependencies",
#       "name": "ensure_docling_dependencies",
#       "anchor": "function-ensure-docling-dependencies",
#       "kind": "function"
#     },
#     {
#       "id": "add-data-root-option",
#       "name": "add_data_root_option",
#       "anchor": "function-add-data-root-option",
#       "kind": "function"
#     },
#     {
#       "id": "add-resume-force-options",
#       "name": "add_resume_force_options",
#       "anchor": "function-add-resume-force-options",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-served-model-names",
#       "name": "_normalize_served_model_names",
#       "anchor": "function-normalize-served-model-names",
#       "kind": "function"
#     },
#     {
#       "id": "detect-vllm-version",
#       "name": "detect_vllm_version",
#       "anchor": "function-detect-vllm-version",
#       "kind": "function"
#     },
#     {
#       "id": "validate-served-models",
#       "name": "validate_served_models",
#       "anchor": "function-validate-served-models",
#       "kind": "function"
#     },
#     {
#       "id": "pdf-build-parser",
#       "name": "pdf_build_parser",
#       "anchor": "function-pdf-build-parser",
#       "kind": "function"
#     },
#     {
#       "id": "pdf-parse-args",
#       "name": "pdf_parse_args",
#       "anchor": "function-pdf-parse-args",
#       "kind": "function"
#     },
#     {
#       "id": "pdftask",
#       "name": "PdfTask",
#       "anchor": "class-pdftask",
#       "kind": "class"
#     },
#     {
#       "id": "pdfconversionresult",
#       "name": "PdfConversionResult",
#       "anchor": "class-pdfconversionresult",
#       "kind": "class"
#     },
#     {
#       "id": "safe-float",
#       "name": "_safe_float",
#       "anchor": "function-safe-float",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-conversion-result",
#       "name": "normalize_conversion_result",
#       "anchor": "function-normalize-conversion-result",
#       "kind": "function"
#     },
#     {
#       "id": "port-is-free",
#       "name": "port_is_free",
#       "anchor": "function-port-is-free",
#       "kind": "function"
#     },
#     {
#       "id": "probe-models",
#       "name": "probe_models",
#       "anchor": "function-probe-models",
#       "kind": "function"
#     },
#     {
#       "id": "probe-metrics",
#       "name": "probe_metrics",
#       "anchor": "function-probe-metrics",
#       "kind": "function"
#     },
#     {
#       "id": "stream-logs",
#       "name": "stream_logs",
#       "anchor": "function-stream-logs",
#       "kind": "function"
#     },
#     {
#       "id": "start-vllm",
#       "name": "start_vllm",
#       "anchor": "function-start-vllm",
#       "kind": "function"
#     },
#     {
#       "id": "wait-for-vllm",
#       "name": "wait_for_vllm",
#       "anchor": "function-wait-for-vllm",
#       "kind": "function"
#     },
#     {
#       "id": "stop-vllm",
#       "name": "stop_vllm",
#       "anchor": "function-stop-vllm",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-vllm",
#       "name": "ensure_vllm",
#       "anchor": "function-ensure-vllm",
#       "kind": "function"
#     },
#     {
#       "id": "list-pdfs",
#       "name": "list_pdfs",
#       "anchor": "function-list-pdfs",
#       "kind": "function"
#     },
#     {
#       "id": "pdf-convert-one",
#       "name": "pdf_convert_one",
#       "anchor": "function-pdf-convert-one",
#       "kind": "function"
#     },
#     {
#       "id": "pdf-main",
#       "name": "pdf_main",
#       "anchor": "function-pdf-main",
#       "kind": "function"
#     },
#     {
#       "id": "html-build-parser",
#       "name": "html_build_parser",
#       "anchor": "function-html-build-parser",
#       "kind": "function"
#     },
#     {
#       "id": "html-parse-args",
#       "name": "html_parse_args",
#       "anchor": "function-html-parse-args",
#       "kind": "function"
#     },
#     {
#       "id": "htmltask",
#       "name": "HtmlTask",
#       "anchor": "class-htmltask",
#       "kind": "class"
#     },
#     {
#       "id": "htmlconversionresult",
#       "name": "HtmlConversionResult",
#       "anchor": "class-htmlconversionresult",
#       "kind": "class"
#     },
#     {
#       "id": "get-converter",
#       "name": "_get_converter",
#       "anchor": "function-get-converter",
#       "kind": "function"
#     },
#     {
#       "id": "list-htmls",
#       "name": "list_htmls",
#       "anchor": "function-list-htmls",
#       "kind": "function"
#     },
#     {
#       "id": "sanitize-html-file",
#       "name": "_sanitize_html_file",
#       "anchor": "function-sanitize-html-file",
#       "kind": "function"
#     },
#     {
#       "id": "html-convert-one",
#       "name": "html_convert_one",
#       "anchor": "function-html-convert-one",
#       "kind": "function"
#     },
#     {
#       "id": "html-main",
#       "name": "html_main",
#       "anchor": "function-html-main",
#       "kind": "function"
#     },
#     {
#       "id": "should-install-docling-test-stubs",
#       "name": "_should_install_docling_test_stubs",
#       "anchor": "function-should-install-docling-test-stubs",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-stub-module",
#       "name": "_ensure_stub_module",
#       "anchor": "function-ensure-stub-module",
#       "kind": "function"
#     },
#     {
#       "id": "install-docling-test-stubs",
#       "name": "_install_docling_test_stubs",
#       "anchor": "function-install-docling-test-stubs",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DocParsing Doctags Pipeline Utilities

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
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    if sys.path and sys.path[0] == str(script_dir):
        sys.path.pop(0)
    package_root = script_dir.parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import argparse
import os
import random
import re
import shutil
import socket
import subprocess as sp
import tempfile
import threading
import time
import types
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import requests
from tqdm import tqdm

from DocsToKG.DocParsing.config import (
    StageConfigBase,
    annotate_cli_overrides,
    parse_args_with_overrides,
)
from DocsToKG.DocParsing.core import (
    DEFAULT_HTTP_TIMEOUT,
    CLIOption,
    ResumeController,
    acquire_lock,
    build_subcommand,
    derive_doc_id_and_doctags_path,
    find_free_port,
    get_http_session,
    normalize_http_timeout,
    set_spawn_or_warn,
)
from DocsToKG.DocParsing.env import (
    PDF_MODEL_SUBDIR,
    data_doctags,
    data_html,
    data_manifests,
    data_pdfs,
    detect_data_root,
    ensure_model_environment,
    prepare_data_root,
    resolve_hf_home,
    resolve_model_root,
    resolve_pdf_model_path,
    resolve_pipeline_path,
)
from DocsToKG.DocParsing.io import (
    compute_content_hash,
    dedupe_preserve_order,
    load_manifest_index,
    manifest_append,
    resolve_attempts_path,
    resolve_manifest_path,
)
from DocsToKG.DocParsing.logging import (
    get_logger,
    log_event,
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
    telemetry_scope,
)
from DocsToKG.DocParsing.telemetry import StageTelemetry, TelemetrySink

try:  # pragma: no cover - optional dependency
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - guard for stripped-down runtime
    InvalidVersion = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]

_LOGGER = get_logger(__name__)

# --- Globals ---

__all__ = (
    "add_data_root_option",
    "add_resume_force_options",
    "detect_vllm_version",
    "ensure_vllm",
    "html_build_parser",
    "html_main",
    "html_parse_args",
    "normalize_conversion_result",
    "pdf_build_parser",
    "pdf_main",
    "pdf_parse_args",
    "prepare_data_root",
    "resolve_hf_home",
    "resolve_model_root",
    "resolve_pdf_model_path",
    "resolve_pipeline_path",
    "validate_served_models",
    "manifest_append",
    "DoctagsCfg",
)

MANIFEST_STAGE = "doctags-pdf"

# Execution defaults for the vLLM-backed conversion pipeline.
PREFERRED_PORT = 8000
PORT_SCAN_SPAN = 32
DEFAULT_WORKERS = min(12, (os.cpu_count() or 16) - 4)
WAIT_TIMEOUT_S = 300
DEFAULT_GPU_MEMORY_UTILIZATION = 0.30
DEFAULT_SERVED_MODEL_NAMES: Tuple[str, ...] = (
    "granite-docling-258M",
    "ibm-granite/granite-docling-258M",
)


@dataclass
class DoctagsCfg(StageConfigBase):
    """Structured configuration for DocTags conversion stages."""

    log_level: str = "INFO"
    data_root: Optional[Path] = None
    input: Optional[Path] = None
    output: Optional[Path] = None
    workers: int = DEFAULT_WORKERS
    port: int = PREFERRED_PORT
    model: Optional[str] = None
    served_model_names: Tuple[str, ...] = DEFAULT_SERVED_MODEL_NAMES
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    vlm_prompt: str = "Convert this page to docling."
    vlm_stop: Tuple[str, ...] = ("</doctag>", "<|end_of_text|>")
    vllm_wait_timeout: int = WAIT_TIMEOUT_S
    http_timeout: Tuple[float, float] = DEFAULT_HTTP_TIMEOUT
    resume: bool = False
    force: bool = False
    overwrite: bool = False
    mode: str = "pdf"
    html_sanitizer: str = "balanced"

    ENV_VARS: ClassVar[Dict[str, str]] = {
        "log_level": "DOCSTOKG_DOCTAGS_LOG_LEVEL",
        "data_root": "DOCSTOKG_DOCTAGS_DATA_ROOT",
        "input": "DOCSTOKG_DOCTAGS_INPUT",
        "output": "DOCSTOKG_DOCTAGS_OUTPUT",
        "workers": "DOCSTOKG_DOCTAGS_WORKERS",
        "port": "DOCSTOKG_DOCTAGS_PORT",
        "model": "DOCSTOKG_DOCTAGS_MODEL",
        "served_model_names": "DOCSTOKG_DOCTAGS_SERVED_MODELS",
        "gpu_memory_utilization": "DOCSTOKG_DOCTAGS_GPU_MEMORY_UTILIZATION",
        "vlm_prompt": "DOCSTOKG_DOCTAGS_VLM_PROMPT",
        "vlm_stop": "DOCSTOKG_DOCTAGS_VLM_STOP",
        "vllm_wait_timeout": "DOCSTOKG_DOCTAGS_VLLM_WAIT_TIMEOUT",
        "http_timeout": "DOCSTOKG_DOCTAGS_HTTP_TIMEOUT",
        "resume": "DOCSTOKG_DOCTAGS_RESUME",
        "force": "DOCSTOKG_DOCTAGS_FORCE",
        "overwrite": "DOCSTOKG_DOCTAGS_OVERWRITE",
        "mode": "DOCSTOKG_DOCTAGS_MODE",
        "config": "DOCSTOKG_DOCTAGS_CONFIG",
        "html_sanitizer": "DOCSTOKG_DOCTAGS_HTML_SANITIZER",
    }

    FIELD_PARSERS: ClassVar[Dict[str, Callable[[Any, Optional[Path]], Any]]] = {
        "config": StageConfigBase._coerce_optional_path,
        "log_level": StageConfigBase._coerce_str,
        "data_root": StageConfigBase._coerce_optional_path,
        "input": StageConfigBase._coerce_path,
        "output": StageConfigBase._coerce_path,
        "workers": StageConfigBase._coerce_int,
        "port": StageConfigBase._coerce_int,
        "model": StageConfigBase._coerce_str,
        "served_model_names": StageConfigBase._coerce_str_tuple,
        "gpu_memory_utilization": StageConfigBase._coerce_float,
        "vlm_prompt": StageConfigBase._coerce_str,
        "vlm_stop": StageConfigBase._coerce_str_tuple,
        "vllm_wait_timeout": StageConfigBase._coerce_int,
        "http_timeout": lambda value, _base_dir: normalize_http_timeout(value),
        "resume": StageConfigBase._coerce_bool,
        "force": StageConfigBase._coerce_bool,
        "overwrite": StageConfigBase._coerce_bool,
        "mode": StageConfigBase._coerce_str,
        "html_sanitizer": StageConfigBase._coerce_str,
    }

    @classmethod
    def from_env(
        cls,
        *,
        mode: str = "pdf",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> "DoctagsCfg":
        """Build a configuration exclusively from environment variables."""

        base_kwargs = dict(defaults or {})
        base_kwargs.setdefault("mode", mode)
        cfg = cls(**base_kwargs)
        cfg.apply_env()
        if cfg.data_root is None:
            fallback_root = os.getenv("DOCSTOKG_DATA_ROOT")
            if fallback_root:
                cfg.data_root = StageConfigBase._coerce_optional_path(fallback_root, None)
        cfg.finalize()
        return cfg

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        mode: str = "pdf",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> "DoctagsCfg":
        """Create a configuration by merging env vars, optional config files, and CLI arguments."""

        cfg = cls.from_env(mode=mode, defaults=defaults)
        config_path = getattr(args, "config", None)
        if config_path:
            cfg.update_from_file(Path(config_path))
        cfg.apply_args(args)
        cfg.finalize()
        return cfg

    def finalize(self) -> None:
        """Normalise derived fields after configuration sources are applied."""
        if self.data_root is not None:
            resolved_root = StageConfigBase._coerce_optional_path(self.data_root, None)
        else:
            env_root = os.getenv("DOCSTOKG_DATA_ROOT")
            if env_root:
                resolved_root = StageConfigBase._coerce_optional_path(env_root, None)
            else:
                resolved_root = detect_data_root()
        self.data_root = resolved_root

        mode = (self.mode or "pdf").lower()
        self.mode = mode

        if self.input is None:
            if mode == "html":
                self.input = data_html(resolved_root)
            else:
                self.input = data_pdfs(resolved_root)
        else:
            self.input = StageConfigBase._coerce_path(self.input, None)

        if self.output is None:
            self.output = data_doctags(resolved_root)
        else:
            self.output = StageConfigBase._coerce_path(self.output, None)

        if self.config is not None:
            self.config = StageConfigBase._coerce_optional_path(self.config, None)
        self.log_level = str(self.log_level or "INFO").upper()
        sanitizer = str(self.html_sanitizer or "balanced").lower()
        if sanitizer not in HTML_SANITIZER_CHOICES:
            raise ValueError(f"html_sanitizer must be one of {', '.join(HTML_SANITIZER_CHOICES)}")
        self.html_sanitizer = sanitizer
        served = StageConfigBase._coerce_str_tuple(self.served_model_names, None)
        if served:
            self.served_model_names = _normalize_served_model_names(served)
        else:
            self.served_model_names = DEFAULT_SERVED_MODEL_NAMES
        stop_values = StageConfigBase._coerce_str_tuple(self.vlm_stop, None)
        if stop_values:
            self.vlm_stop = tuple(stop_values)
        else:
            self.vlm_stop = ("</doctag>", "<|end_of_text|>")
        self.http_timeout = normalize_http_timeout(self.http_timeout)

        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.port <= 0:
            raise ValueError("port must be a positive integer")
        if self.gpu_memory_utilization < 0:
            raise ValueError("gpu_memory_utilization must be non-negative")

    from_sources = from_args


PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "cpu-small": {
        "workers": 1,
        "gpu_memory_utilization": 0.0,
        "vllm_wait_timeout": 180,
        "port": PREFERRED_PORT + 2,
    },
    "gpu-default": {
        "workers": DEFAULT_WORKERS,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "vllm_wait_timeout": WAIT_TIMEOUT_S,
        "port": PREFERRED_PORT,
    },
    "gpu-max": {
        "workers": max(1, (os.cpu_count() or 16) - 2),
        "gpu_memory_utilization": min(0.9, DEFAULT_GPU_MEMORY_UTILIZATION + 0.15),
        "vllm_wait_timeout": WAIT_TIMEOUT_S * 2,
        "port": PREFERRED_PORT + 4,
    },
}


PDF_CLI_OPTIONS: Tuple[CLIOption, ...] = (
    CLIOption(
        ("--config",),
        {"type": Path, "default": None, "help": "Path to stage config file (JSON/YAML/TOML)."},
    ),
    CLIOption(
        ("--profile",),
        {
            "type": str,
            "default": None,
            "choices": sorted(PROFILE_PRESETS),
            "help": "Apply a preset for batch sizes and workers (cpu-small, gpu-default, gpu-max).",
        },
    ),
    CLIOption(
        ("--log-level",),
        {
            "type": lambda value: str(value).upper(),
            "default": "INFO",
            "choices": ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            "help": "Logging verbosity for console output (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--input",),
        {
            "type": Path,
            "default": None,
            "help": "Folder with PDFs (defaults to data_root/PDFs).",
        },
    ),
    CLIOption(
        ("--output",),
        {
            "type": Path,
            "default": None,
            "help": "Folder for Doctags output (defaults to data_root/DocTagsFiles).",
        },
    ),
    CLIOption(
        ("--workers",),
        {"type": int, "default": DEFAULT_WORKERS, "help": "Parallel workers for PDF conversion"},
    ),
    CLIOption(
        ("--port",),
        {
            "type": int,
            "default": PREFERRED_PORT,
            "help": f"vLLM HTTP port to use (default: {PREFERRED_PORT}).",
        },
    ),
    CLIOption(
        ("--model",),
        {
            "type": str,
            "default": None,
            "help": (
                "Path or identifier for the vLLM model to serve. Defaults to DOCLING_PDF_MODEL, "
                f"DOCSTOKG_MODEL_ROOT/{PDF_MODEL_SUBDIR}, or HF_HOME/{PDF_MODEL_SUBDIR}."
            ),
        },
    ),
    CLIOption(
        ("--served-model-name",),
        {
            "dest": "served_model_names",
            "action": "append",
            "nargs": "+",
            "default": None,
            "help": "Model name to expose via OpenAI compatibility API (repeatable)",
        },
    ),
    CLIOption(
        ("--gpu-memory-utilization",),
        {
            "type": float,
            "default": DEFAULT_GPU_MEMORY_UTILIZATION,
            "help": "Fraction of GPU memory the vLLM server may allocate",
        },
    ),
    CLIOption(
        ("--http-timeout",),
        {
            "type": float,
            "nargs": 2,
            "metavar": ("CONNECT", "READ"),
            "default": None,
            "help": "Override HTTP connect/read timeout (seconds) for vLLM/docling probes",
        },
    ),
    CLIOption(
        ("--vlm-prompt",),
        {
            "type": str,
            "default": "Convert this page to docling.",
            "help": "Prompt passed to the VLM for PDF pages",
        },
    ),
    CLIOption(
        ("--vlm-stop",),
        {
            "action": "append",
            "default": ["</doctag>", "<|end_of_text|>"],
            "help": "Stop tokens for the VLM (repeatable)",
        },
    ),
    CLIOption(
        ("--vllm-wait-timeout",),
        {
            "type": int,
            "default": WAIT_TIMEOUT_S,
            "help": "Seconds to wait for vLLM server readiness (default: %(default)s).",
        },
    ),
)


def ensure_docling_dependencies() -> None:
    """Validate that required Docling packages are installed."""

    try:
        import docling_core  # noqa: F401  # pragma: no cover - import validation only
        from docling.document_converter import DocumentConverter  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when dependencies missing
        if _should_install_docling_test_stubs():
            _install_docling_test_stubs()
            try:  # pragma: no cover - exercised during tests
                import docling_core  # noqa: F401,E401
                from docling.document_converter import DocumentConverter  # noqa: F401,E401
            except ImportError as inner_exc:
                raise ImportError(
                    "DocTags conversion requires the 'docling' and 'docling-core' packages. "
                    "Install them with `pip install docling docling-core`."
                ) from inner_exc
            return
        raise ImportError(
            "DocTags conversion requires the 'docling' and 'docling-core' packages. "
            "Install them with `pip install docling docling-core`."
        ) from exc


# Thread hygiene for CPU libs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("DOCLING_CUDA_USE_FLASH_ATTENTION2", "1")

ARTIFACTS = os.environ.get("DOCLING_ARTIFACTS_PATH", "")


# --- CLI Helpers ---


def add_data_root_option(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--data-root`` option to a CLI parser.

    Args:
        parser (argparse.ArgumentParser): Parser being configured.

    Returns:
        None

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> add_data_root_option(parser)
        >>> any(action.dest == "data_root" for action in parser._actions)
        True
    """

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or $DOCSTOKG_DATA_ROOT."
        ),
    )


def add_resume_force_options(
    parser: argparse.ArgumentParser,
    *,
    resume_help: str,
    force_help: str,
) -> None:
    """Attach ``--resume`` and ``--force`` switches to a CLI parser.

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
    """

    parser.add_argument("--resume", action="store_true", help=resume_help)
    parser.add_argument("--force", action="store_true", help=force_help)


# --- vLLM Lifecycle ---


def _normalize_served_model_names(raw: Optional[Iterable[Iterable[str] | str]]) -> Tuple[str, ...]:
    """Flatten CLI-provided served model names into a deduplicated tuple.

    Args:
        raw: Sequence containing strings or nested iterables of strings sourced from
            CLI options.

    Returns:
        Tuple of unique served model names with defaults applied when none were given.

    Examples:
        >>> _normalize_served_model_names([["a", "b"], "b"])
        ('a', 'b')
    """

    if not raw:
        return DEFAULT_SERVED_MODEL_NAMES

    flattened: List[str] = []
    for entry in raw:
        if isinstance(entry, str):
            flattened.append(entry)
        else:
            flattened.extend(str(item) for item in entry)
    return dedupe_preserve_order(flattened) or DEFAULT_SERVED_MODEL_NAMES


def detect_vllm_version() -> str:
    """Detect the installed vLLM package version for diagnostics.

    Args:
        None

    Returns:
        Version string reported by the local vLLM installation or ``"unknown"`` when
        the package cannot be imported.
    """

    try:  # pragma: no cover - requires optional dependency
        import vllm  # type: ignore
    except Exception as exc:  # pragma: no cover - logged for operator visibility
        _LOGGER.warning(
            "Unable to import vLLM for version detection",
            extra={"extra_fields": {"error": str(exc)}},
        )
        return "unknown"

    version = getattr(vllm, "__version__", "unknown")
    _LOGGER.info(
        "Detected vLLM package",
        extra={"extra_fields": {"version": version}},
    )
    if Version is not None and version not in {"unknown", ""}:
        try:
            if Version(version) < Version("0.3.0"):
                _LOGGER.warning(
                    "vLLM version %s is below the supported minimum of 0.3.0",
                    version,
                )
        except InvalidVersion:  # pragma: no cover - defensive parsing guard
            _LOGGER.debug(
                "Could not parse vLLM version string", extra={"extra_fields": {"version": version}}
            )
    return version


def validate_served_models(available: Optional[List[str]], expected: Tuple[str, ...]) -> None:
    """Ensure that at least one of the expected served model names is available.

    Args:
        available: Model aliases exposed by the running vLLM server.
        expected: Tuple of acceptable model aliases configured for conversion.

    Returns:
        None

    Raises:
        RuntimeError: If none of the expected model names are present.
    """

    if not expected:
        return

    candidates = set(available or [])
    if any(name in candidates for name in expected):
        return

    raise RuntimeError(
        "Expected model not served",
        {
            "expected": list(expected),
            "available": list(candidates),
        },
    )


def pdf_build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the PDF → DocTags converter.

    Args:
        None: Parser construction does not require inputs.

    Returns:
        Argument parser configured with all supported CLI options.

    Raises:
        ValueError: If parser configuration fails due to invalid defaults.
    """

    parser = argparse.ArgumentParser(
        description="Convert PDF corpora to DocTags with an optional vLLM backend",
    )
    add_data_root_option(parser)
    build_subcommand(parser, PDF_CLI_OPTIONS)
    add_resume_force_options(
        parser,
        resume_help="Skip PDFs whose DocTags already exist with matching content hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def pdf_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone execution.

    Args:
        argv: Optional CLI argument list. When ``None`` the values from
            :data:`sys.argv` are used.

    Returns:
        Namespace containing parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` detects invalid arguments.
    """

    parser = pdf_build_parser()
    return parse_args_with_overrides(parser, argv)


# --- PDF Pipeline ---


@dataclass
class PdfTask:
    """Work item representing a single PDF conversion request.

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
    """

    pdf_path: Path
    output_dir: Path
    port: int
    input_hash: str
    doc_id: str
    output_path: Path
    served_model_names: Tuple[str, ...]
    inference_model: str
    vlm_prompt: str
    vlm_stop: Tuple[str, ...]


@dataclass
class PdfConversionResult:
    """Structured result returned by worker processes.

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
    """

    doc_id: str
    status: str
    duration_s: float
    input_path: str
    input_hash: str
    output_path: str
    error: Optional[str] = None


def _safe_float(value: Any) -> float:
    """Convert the supplied value to ``float`` when possible.

    Args:
        value: Object that may represent a numeric scalar.

    Returns:
        Floating point representation of ``value`` or ``0.0`` if conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def normalize_conversion_result(result: Any, task: Optional[PdfTask] = None) -> PdfConversionResult:
    """Validate that worker results conform to :class:`PdfConversionResult`.

    Args:
        result: Object returned by a worker invocation.
        task: Deprecated parameter retained for API stability; ignored.

    Returns:
        The original :class:`PdfConversionResult` when the type check passes.

    Raises:
        TypeError: If the worker returned an unexpected result type.
    """

    if isinstance(result, PdfConversionResult):
        return result
    raise TypeError(
        f"pdf_convert_one must return PdfConversionResult, received {type(result).__name__}"
    )


def port_is_free(port: int) -> bool:
    """Determine whether a TCP port on localhost is currently available.

    Args:
        port: Port number to probe on the loopback interface.

    Returns:
        True when the port is unused; otherwise False.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) != 0


def probe_models(
    port: int, timeout: Optional[object] = None
) -> Tuple[Optional[List[str]], Optional[str], Optional[int]]:
    """Inspect the `/v1/models` endpoint exposed by a vLLM HTTP server.

    Args:
        port: HTTP port where the vLLM server is expected to listen.
        timeout: Seconds to wait for the HTTP request before aborting.

    Returns:
        Tuple containing the list of model identifiers (if any), the raw response
        body, and the HTTP status code. Missing models or connection failures are
        represented by `(None, <error>, None)`.
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    session, request_timeout = get_http_session(timeout=timeout)
    try:
        r = session.get(url, timeout=request_timeout)
        raw = r.text
        if r.headers.get("content-type", "").startswith("application/json"):
            try:
                data = r.json()
            except Exception:
                data = None
        else:
            data = None
        names = []
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for m in data["data"]:
                mid = m.get("id") or m.get("name")
                if mid:
                    names.append(mid)
        return names if names else [], raw, r.status_code
    except requests.RequestException as e:
        return None, str(e), None


def probe_metrics(port: int, timeout: Optional[object] = None) -> Tuple[bool, Optional[int]]:
    """Check whether the vLLM `/metrics` endpoint is healthy.

    Args:
        port: HTTP port where the vLLM server should expose metrics.
        timeout: Seconds to wait for the HTTP response before aborting.

    Returns:
        Tuple of `(is_healthy, status_code)` where `is_healthy` is True when the
        endpoint responds with HTTP 200.
    """
    url = f"http://127.0.0.1:{port}/metrics"
    session, request_timeout = get_http_session(timeout=timeout)
    try:
        r = session.get(url, timeout=request_timeout)
        return (r.status_code == 200), r.status_code
    except requests.RequestException:
        return (False, None)


def stream_logs(proc: sp.Popen, prefix: str = "[vLLM] ", tail: Optional[Deque[str]] = None):
    """Continuously stream stdout lines from a child process to the console.

    Args:
        proc: Running subprocess whose stdout should be tailed.
        prefix: Text prefix applied to each emitted log line for readability.
        tail: Optional deque that accumulates the most recent log lines.

    Returns:
        None: This routine streams output for side effects only.
    """
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        s = line.rstrip()
        if not s:
            continue
        if tail is not None:
            tail.append(s)
        _LOGGER.info(
            "vLLM stdout",
            extra={
                "extra_fields": {
                    "source": "vllm",
                    "line": prefix + s,
                }
            },
        )


def start_vllm(
    port: int,
    model_path: str,
    served_model_names: Tuple[str, ...],
    gpu_memory_utilization: float,
) -> sp.Popen:
    """Launch a vLLM server process on the requested port.

    Args:
        port: Port on which the vLLM HTTP server should listen.
        model_path: Local directory or HF repository containing model weights.
        served_model_names: Aliases registered with the OpenAI-compatible API.
        gpu_memory_utilization: Fraction of GPU memory the server may allocate.

    Returns:
        Started subprocess handle for the vLLM server.

    Raises:
        SystemExit: If the `vllm` executable is not present on `PATH`.
    """
    if shutil.which("vllm") is None:
        _LOGGER.error("'vllm' not found on PATH", extra={"extra_fields": {"cmd": "vllm"}})
        sys.exit(1)

    cmd = [
        "vllm",
        "serve",
        str(model_path),
        "--port",
        str(port),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
    ]
    for name in served_model_names:
        cmd.extend(["--served-model-name", name])

    env = os.environ.copy()
    env.setdefault("VLLM_LOG_LEVEL", "INFO")  # INFO so we can see useful lines
    _LOGGER.info(
        "Starting vLLM",
        extra={"extra_fields": {"command": cmd, "env_log_level": env.get("VLLM_LOG_LEVEL")}},
    )
    proc = sp.Popen(cmd, env=env, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    tail: Deque[str] = deque(maxlen=50)
    setattr(proc, "_log_tail", tail)
    # Start log thread
    t = threading.Thread(target=stream_logs, args=(proc, "[vLLM] ", tail), daemon=True)
    t.start()
    return proc


def wait_for_vllm(
    port: int,
    proc: sp.Popen,
    *,
    timeout_s: int = WAIT_TIMEOUT_S,
    http_timeout: Optional[object] = None,
) -> List[str]:
    """Poll the vLLM server until `/v1/models` responds with success.

    Args:
        port: HTTP port where the server is expected to listen.
        proc: Subprocess handle representing the running vLLM instance.
        timeout_s: Maximum time in seconds to wait for readiness.
        http_timeout: Optional override for HTTP connect/read timeouts used during probes.

    Returns:
        Model names reported by the server upon readiness.

    Raises:
        RuntimeError: If the server exits prematurely or fails to become ready
            within the allotted timeout.
    """
    log_event(_LOGGER, "info", "Probing vLLM", port=port, timeout_s=timeout_s)
    start = time.time()
    attempt = 0
    with tqdm(total=timeout_s, unit="s", desc="vLLM warmup", leave=True) as bar:
        while True:
            attempt += 1
            if proc.poll() is not None:
                tail_text = ""
                try:
                    tail_lines = list(getattr(proc, "_log_tail", []))
                    leftover = proc.stdout.read() or ""
                    combined_tail = tail_lines[-50:]
                    tail_text = "\n".join(combined_tail) if combined_tail else leftover[-800:]
                    log_event(
                        _LOGGER, "error", "vLLM exited while waiting", port=port, tail=tail_text
                    )
                finally:
                    message = (
                        f"vLLM exited early with code {proc.returncode}. "
                        "Last log lines:\n"
                        f"{tail_text}\n"
                        "Common causes include missing model weights, incompatible CUDA drivers, "
                        "or insufficient GPU memory."
                    )
                    raise RuntimeError(message)

            names, raw, status = probe_models(port, timeout=http_timeout)
            if status == 200 and names:
                log_event(_LOGGER, "info", "vLLM models available", port=port, models=names)
                if bar.n < timeout_s:
                    bar.update(timeout_s - bar.n)
                return names
            if status == 200 and not names and (attempt == 1 or attempt % 5 == 0):
                log_event(_LOGGER, "info", "vLLM HTTP up; waiting for models", port=port)
            if status and status >= 400 and (attempt == 1 or attempt % 5 == 0):
                preview = (raw or "")[:200]
                log_event(
                    _LOGGER,
                    "warning",
                    "vLLM model probe returned error",
                    port=port,
                    status=status,
                    response_preview=preview,
                )

            elapsed = time.time() - start
            if elapsed >= timeout_s:
                raise RuntimeError(f"Timed out waiting for vLLM on port {port}")

            remaining = timeout_s - elapsed
            base_sleep = min(1.0 + 0.2 * attempt, 2.5)
            interval = min(remaining, base_sleep + random.uniform(0.2, 0.6))
            interval = max(0.2, interval)
            time.sleep(interval)
            bar.update(min(interval, remaining))


def stop_vllm(proc: Optional[sp.Popen], own: bool, grace=10):
    """Terminate a managed vLLM process if this script launched it.

    Args:
        proc: Subprocess handle returned by `start_vllm`, or None.
        own: Indicates whether the caller owns the process lifetime.
        grace: Seconds to wait for graceful shutdown before forcing exit.

    Returns:
        None.
    """
    if not own or proc is None or proc.poll() is not None:
        return
    _LOGGER.info(
        "Stopping vLLM",
        extra={"extra_fields": {"grace_seconds": grace}},
    )
    try:
        proc.terminate()
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < grace:
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass


def ensure_vllm(
    preferred: int,
    model_path: str,
    served_model_names: Tuple[str, ...],
    gpu_memory_utilization: float,
    *,
    wait_timeout_s: int = WAIT_TIMEOUT_S,
    http_timeout: Optional[object] = None,
) -> Tuple[int, Optional[sp.Popen], bool]:
    """Ensure a vLLM server is available, launching one when necessary.

    Args:
        preferred: Preferred TCP port for the server.
        model_path: Model repository or path passed to the vLLM CLI.
        served_model_names: Aliases that should be exposed via the OpenAI API.
        gpu_memory_utilization: Fractional GPU memory reservation for the server.
        wait_timeout_s: Seconds to wait for vLLM readiness.
        http_timeout: Optional override for HTTP connect/read timeout when probing the server.

    Returns:
        Tuple containing `(port, process, owns_process)` where `process` is the
        managed subprocess handle (or None if reusing an existing server) and
        `owns_process` indicates whether the caller should terminate it.

    Raises:
        RuntimeError: If the launched or reused server does not expose the expected
            model aliases, indicating a misconfiguration.
    """
    # 1) If preferred is free, start there
    if port_is_free(preferred):
        proc = start_vllm(preferred, model_path, served_model_names, gpu_memory_utilization)
        names = wait_for_vllm(preferred, proc, timeout_s=wait_timeout_s, http_timeout=http_timeout)
        validate_served_models(names, served_model_names)
        return preferred, proc, True

    # 2) If something is already on preferred, reuse if it's vLLM (any models list)
    names, raw, status = probe_models(preferred, timeout=http_timeout)
    if status == 200:
        validate_served_models(names, served_model_names)
        _LOGGER.info(
            "Reusing vLLM",
            extra={
                "extra_fields": {
                    "port": preferred,
                    "models": names,
                }
            },
        )
        return preferred, None, False

    # 3) Otherwise, pick a new free port
    alt = find_free_port(preferred + 1, PORT_SCAN_SPAN)
    _LOGGER.info(
        "Launching vLLM on alternate port",
        extra={
            "extra_fields": {
                "preferred_port": preferred,
                "alternate_port": alt,
            }
        },
    )
    proc = start_vllm(alt, model_path, served_model_names, gpu_memory_utilization)
    names = wait_for_vllm(alt, proc, timeout_s=wait_timeout_s, http_timeout=http_timeout)
    validate_served_models(names, served_model_names)
    return alt, proc, True


def list_pdfs(root: Path) -> List[Path]:
    """Collect PDF files under a directory recursively.

    Args:
        root: Directory whose subtree should be scanned for PDFs.

    Returns:
        Sorted list of paths to PDF files.
    """
    return sorted([p for p in root.rglob("*.pdf") if p.is_file()])


# PDF worker helpers
def pdf_convert_one(task: PdfTask) -> PdfConversionResult:
    """Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

    Args:
        task: Description of the conversion request, including paths and port.

    Returns:
        Populated :class:`PdfConversionResult` reporting success, skip, or failure.
    """

    start = time.perf_counter()
    pdf_path = task.pdf_path
    out_dir = task.output_dir
    port = task.port
    out_path = task.output_path
    inference_model = task.inference_model

    try:
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

        # Imports (exact modules)
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions  # <-- correct module
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="skip",
                duration_s=0.0,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
            )

        # Accelerator (use CUDA; keep CPU thread count small per worker)
        accel = AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CUDA)

        # Remote VLM (OpenAI-compatible) -> vLLM
        api = ApiVlmOptions(
            url=f"http://127.0.0.1:{port}/v1/chat/completions",
            params=dict(
                # use the name you served via --served-model-name
                model=inference_model,
                max_tokens=4096,  # <-- IMPORTANT for vLLM
                skip_special_tokens=False,
                temperature=0.1,
                stop=list(task.vlm_stop),
            ),
            prompt=task.vlm_prompt,
            timeout=120,
            scale=2.0,  # image scale for the request
            response_format=ResponseFormat.DOCTAGS,
        )

        # VLM pipeline options (this is the only options object used for VlmPipeline)
        vlm_opts = VlmPipelineOptions(
            enable_remote_services=True,
            accelerator_options=accel,
            do_picture_description=True,
            do_picture_classification=True,
            # ensure Docling renders page bitmaps for the VLM:
            generate_page_images=True,
            images_scale=2.0,  # render at 2x for fidelity
        )
        vlm_opts.vlm_options = api

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_opts,  # <-- only this options object is used
                )
            }
        )

        result = converter.convert(pdf_path, raises_on_error=False)
        if result.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
            err_msgs = []
            for err in getattr(result, "errors", []) or []:
                msg = getattr(err, "error_message", None)
                if msg:
                    err_msgs.append(msg)
            detail = f"status={getattr(result.status, 'value', result.status)}"
            if err_msgs:
                detail += " " + "; ".join(err_msgs)
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error=detail,
            )

        if result.document is None:
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error="empty-document",
            )

        try:
            with acquire_lock(out_path):
                if out_path.exists():
                    return PdfConversionResult(
                        doc_id=task.doc_id,
                        status="skip",
                        duration_s=time.perf_counter() - start,
                        input_path=str(pdf_path),
                        input_hash=task.input_hash,
                        output_path=str(out_path),
                    )
                result.document.save_as_doctags(out_path)
        except TimeoutError as exc:
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error=str(exc),
            )
        return PdfConversionResult(
            doc_id=task.doc_id,
            status="success",
            duration_s=time.perf_counter() - start,
            input_path=str(pdf_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
        )
    except Exception as exc:  # pragma: no cover - exercised during integration runs
        return PdfConversionResult(
            doc_id=task.doc_id,
            status="failure",
            duration_s=time.perf_counter() - start,
            input_path=str(pdf_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
            error=str(exc),
        )


def pdf_main(args: argparse.Namespace | None = None) -> int:
    """Coordinate vLLM startup and parallel DocTags conversion.

    Args:
        args: Optional argument namespace injected during programmatic use.

    Returns:
        Process exit code, where ``0`` indicates success.

    Raises:
        RuntimeError: If vLLM fails to start, becomes unhealthy, or conversion
            retries exhaust without success.
        ValueError: If required configuration (such as auto-detected mode) is invalid.
    """

    bootstrap_root = detect_data_root()
    try:
        data_pdfs(bootstrap_root)
        data_doctags(bootstrap_root)
    except Exception:
        pass

    if args is None:
        namespace = pdf_parse_args()
    elif isinstance(args, argparse.Namespace):
        namespace = args
        if getattr(namespace, "_cli_explicit_overrides", None) is None:
            keys = [name for name in vars(namespace) if not name.startswith("_")]
            annotate_cli_overrides(namespace, explicit=keys, defaults={})
    else:
        namespace = pdf_parse_args(args)

    profile = getattr(namespace, "profile", None)
    defaults = PROFILE_PRESETS.get(profile or "", {})
    cfg = DoctagsCfg.from_args(namespace, mode="pdf", defaults=defaults)
    config_snapshot = cfg.to_manifest()
    for field_def in fields(DoctagsCfg):
        setattr(namespace, field_def.name, getattr(cfg, field_def.name))
    if profile:
        config_snapshot.setdefault("profile", profile)

    log_level = cfg.log_level
    run_id = uuid.uuid4().hex
    logger = get_logger(
        __name__,
        level=str(log_level),
        base_fields={"run_id": run_id, "stage": MANIFEST_STAGE},
    )
    if profile and defaults:
        logger.info(
            "Applying profile",
            extra={
                "extra_fields": {
                    "profile": profile,
                    **{key: defaults[key] for key in sorted(defaults)},
                }
            },
        )
    set_spawn_or_warn(logger)
    import multiprocessing as mp

    logger.info(
        "Multiprocessing configuration",
        extra={
            "extra_fields": {
                "start_method": mp.get_start_method(allow_none=True),
                "cpu_count": os.cpu_count(),
            }
        },
    )

    args = namespace

    ensure_docling_dependencies()
    ensure_model_environment()

    served_model_names = cfg.served_model_names
    inference_model = served_model_names[0]
    model_path = resolve_pdf_model_path(cfg.model)
    args.model = model_path
    gpu_memory_utilization = float(cfg.gpu_memory_utilization)

    vllm_version = detect_vllm_version()

    data_root_override = namespace.data_root if hasattr(namespace, "data_root") else None
    resolved_root = cfg.data_root if cfg.data_root is not None else detect_data_root()

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    manifest_dir = data_manifests(resolved_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(cfg.input).resolve()
    output_dir = Path(cfg.output).resolve()

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot.update(
        {
            "mode": cfg.mode,
            "data_root": str(resolved_root),
            "input": str(input_dir),
            "output": str(output_dir),
            "model": str(model_path) if model_path else None,
            "gpu_memory_utilization": gpu_memory_utilization,
            "served_model_names": list(served_model_names),
            "workers": int(cfg.workers),
            "port": int(cfg.port),
            "resume": bool(cfg.resume),
            "force": bool(cfg.force),
            "vllm_wait_timeout": int(cfg.vllm_wait_timeout),
            "http_timeout": [float(cfg.http_timeout[0]), float(cfg.http_timeout[1])],
        }
    )

    telemetry_sink = TelemetrySink(
        resolve_attempts_path(MANIFEST_STAGE, resolved_root),
        resolve_manifest_path(MANIFEST_STAGE, resolved_root),
    )
    stage_telemetry = StageTelemetry(telemetry_sink, run_id=run_id, stage=MANIFEST_STAGE)
    with telemetry_scope(stage_telemetry):

        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id="__config__",
            duration_s=0.0,
            schema_version="docparse/1.1.0",
            input_path=input_dir,
            input_hash="",
            output_path=output_dir,
            config=config_snapshot,
        )

        logger.info(
            "I/O configuration",
            extra={
                "extra_fields": {
                    "data_root": str(resolved_root),
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "artifacts_cache": ARTIFACTS or "",
                    "model_path": model_path,
                    "served_models": list(served_model_names),
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "vllm_version": vllm_version,
                    "vllm_wait_timeout": int(cfg.vllm_wait_timeout),
                    "port": int(cfg.port),
                    "profile": profile,
                    "http_timeout": [float(cfg.http_timeout[0]), float(cfg.http_timeout[1])],
                }
            },
        )

        if cfg.force:
            logger.info("Force mode: reprocessing all documents")
        elif cfg.resume:
            logger.info("Resume mode enabled: unchanged outputs will be skipped")

        preflight_start = time.perf_counter()
        port, proc, owns = ensure_vllm(
            int(cfg.port),
            model_path,
            served_model_names,
            gpu_memory_utilization,
            wait_timeout_s=int(cfg.vllm_wait_timeout),
            http_timeout=cfg.http_timeout,
        )
        metrics_healthy, metrics_status = probe_metrics(port, timeout=cfg.http_timeout)
        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id="__service__",
            duration_s=round(time.perf_counter() - preflight_start, 3),
            schema_version="docparse/1.1.0",
            input_path=model_path,
            input_hash="",
            output_path=output_dir,
            served_models=list(served_model_names),
            vllm_version=vllm_version,
            port=port,
            owns_process=owns,
            metrics_healthy=metrics_healthy,
            metrics_status_code=metrics_status,
            http_timeout=[float(cfg.http_timeout[0]), float(cfg.http_timeout[1])],
        )
        logger.info(
            "vLLM server ready",
            extra={
                "extra_fields": {
                    "port": port,
                    "owns_process": owns,
                    "http_timeout": [float(cfg.http_timeout[0]), float(cfg.http_timeout[1])],
                }
            },
        )

        try:
            pdfs = list_pdfs(input_dir)
            if not pdfs:
                log_event(
                    logger,
                    "warning",
                    "No PDFs found",
                    stage=MANIFEST_STAGE,
                    doc_id="__aggregate__",
                    input_hash=None,
                    error_code="NO_INPUT_FILES",
                    input_dir=str(input_dir),
                )
                return 0

            manifest_index = (
                load_manifest_index(MANIFEST_STAGE, resolved_root) if cfg.resume else {}
            )
            resume_controller = ResumeController(cfg.resume, cfg.force, manifest_index)

            workers = max(1, int(cfg.workers))
            logger.info(
                "Launching workers",
                extra={
                    "extra_fields": {
                        "pdf_count": len(pdfs),
                        "workers": workers,
                    }
                },
            )

            tasks: List[PdfTask] = []
            ok = fail = skip = 0
            for pdf_path in pdfs:
                doc_id, out_path = derive_doc_id_and_doctags_path(pdf_path, input_dir, output_dir)
                input_hash = compute_content_hash(pdf_path)
                skip_doc, _ = resume_controller.should_skip(doc_id, out_path, input_hash)
                if skip_doc:
                    logger.info(
                        "Skipping document: output exists and input unchanged",
                        extra={
                            "extra_fields": {
                                "doc_id": doc_id,
                                "output_path": str(out_path),
                            }
                        },
                    )
                    manifest_log_skip(
                        stage=MANIFEST_STAGE,
                        doc_id=doc_id,
                        input_path=pdf_path,
                        input_hash=input_hash,
                        output_path=out_path,
                        schema_version="docparse/1.1.0",
                        parse_engine="docling-vlm",
                        model_name=inference_model,
                        served_models=list(served_model_names),
                        vllm_version=vllm_version,
                    )
                    skip += 1
                    continue

                tasks.append(
                    PdfTask(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        port=port,
                        input_hash=input_hash,
                        doc_id=doc_id,
                        output_path=out_path,
                        served_model_names=served_model_names,
                        inference_model=inference_model,
                        vlm_prompt=str(args.vlm_prompt),
                        vlm_stop=tuple(args.vlm_stop or []),
                    )
                )

            if not tasks:
                logger.info(
                    "Conversion summary",
                    extra={
                        "extra_fields": {
                            "ok": 0,
                            "skip": skip,
                            "fail": 0,
                        }
                    },
                )
                return 0

            with ProcessPoolExecutor(max_workers=workers) as ex:
                future_map = {ex.submit(pdf_convert_one, task): task for task in tasks}
                with tqdm(total=len(future_map), desc="Converting PDFs", unit="file") as pbar:
                    for fut in as_completed(future_map):
                        task = future_map[fut]
                        raw_result = fut.result()
                        result = normalize_conversion_result(raw_result, task)
                        if result.status == "success":
                            ok += 1
                        elif result.status == "skip":
                            skip += 1
                        else:
                            fail += 1
                            log_event(
                                logger,
                                "error",
                                "Conversion failed",
                                stage=MANIFEST_STAGE,
                                doc_id=result.doc_id,
                                input_hash=result.input_hash,
                                error_code="PDF_CONVERSION_FAILED",
                                error=result.error or "unknown",
                            )

                        duration = round(result.duration_s, 3)
                        common_extra = {
                            "parse_engine": "docling-vlm",
                            "model_name": task.inference_model,
                            "served_models": list(task.served_model_names),
                            "vllm_version": vllm_version,
                        }
                        if result.status == "success":
                            manifest_log_success(
                                stage=MANIFEST_STAGE,
                                doc_id=result.doc_id,
                                duration_s=duration,
                                schema_version="docparse/1.1.0",
                                input_path=result.input_path,
                                input_hash=result.input_hash,
                                output_path=result.output_path,
                                **common_extra,
                            )
                        elif result.status == "skip":
                            manifest_log_skip(
                                stage=MANIFEST_STAGE,
                                doc_id=result.doc_id,
                                input_path=result.input_path,
                                input_hash=result.input_hash,
                                output_path=result.output_path,
                                schema_version="docparse/1.1.0",
                                duration_s=duration,
                                **common_extra,
                            )
                        else:
                            manifest_log_failure(
                                stage=MANIFEST_STAGE,
                                doc_id=result.doc_id,
                                duration_s=duration,
                                schema_version="docparse/1.1.0",
                                input_path=result.input_path,
                                input_hash=result.input_hash,
                                output_path=result.output_path,
                                error=result.error or "unknown",
                                **common_extra,
                            )

                        pbar.update(1)

            logger.info(
                "Conversion summary",
                extra={
                    "extra_fields": {
                        "ok": ok,
                        "skip": skip,
                        "fail": fail,
                    }
                },
            )
        finally:
            stop_vllm(proc, owns, grace=10)
            logger.info("All done")

        return 0


# --- HTML Pipeline ---

HTML_MANIFEST_STAGE = "doctags-html"
HTML_DEFAULT_WORKERS = max(1, (os.cpu_count() or 8) - 1)
HTML_SANITIZER_CHOICES: Tuple[str, ...] = ("strict", "balanced", "permissive")

HTML_CLI_OPTIONS: Tuple[CLIOption, ...] = (
    CLIOption(
        ("--config",),
        {"type": Path, "default": None, "help": "Path to stage config file (JSON/YAML/TOML)."},
    ),
    CLIOption(
        ("--log-level",),
        {
            "type": lambda value: str(value).upper(),
            "default": "INFO",
            "choices": ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            "help": "Logging verbosity for console output (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--input",),
        {
            "type": Path,
            "default": None,
            "help": "Folder with HTML files (defaults to data_root/HTML)",
        },
    ),
    CLIOption(
        ("--output",),
        {
            "type": Path,
            "default": None,
            "help": "Destination for .doctags (defaults to data_root/DocTagsFiles)",
        },
    ),
    CLIOption(
        ("--workers",),
        {"type": int, "default": HTML_DEFAULT_WORKERS, "help": "Parallel workers"},
    ),
    CLIOption(
        ("--http-timeout",),
        {
            "type": float,
            "nargs": 2,
            "metavar": ("CONNECT", "READ"),
            "default": None,
            "help": "Override HTTP connect/read timeout (seconds) for docling service calls",
        },
    ),
    CLIOption(
        ("--overwrite",), {"action": "store_true", "help": "Overwrite existing .doctags files"}
    ),
    CLIOption(
        ("--html-sanitizer",),
        {
            "type": lambda value: str(value).lower(),
            "default": "balanced",
            "choices": list(HTML_SANITIZER_CHOICES),
            "help": (
                "HTML sanitizer profile controlling removal of scripts/styles/trackers "
                "(strict, balanced, permissive)."
            ),
        },
    ),
)

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # ensure CPU-only HTML conversions

_CONVERTER: "DocumentConverter | None" = None


def html_build_parser() -> argparse.ArgumentParser:
    """Construct an argument parser for the HTML → DocTags converter.

    Args:
        None: Parser initialization does not require inputs.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.

    Raises:
        None
    """

    parser = argparse.ArgumentParser(
        description="Convert HTML corpora to DocTags using Docling",
    )
    add_data_root_option(parser)
    build_subcommand(parser, HTML_CLI_OPTIONS)
    add_resume_force_options(
        parser,
        resume_help="Skip documents whose outputs already exist with matching content hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def html_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional CLI argument vector. When ``None`` the values from
            :data:`sys.argv` are used.

    Returns:
        Namespace containing parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` detects invalid options.
    """

    parser = html_build_parser()
    return parse_args_with_overrides(parser, argv)


@dataclass
class HtmlTask:
    """Work item describing a single HTML conversion job.

    Attributes:
        html_path: Absolute path to the HTML file to be converted.
        relative_id: Relative identifier for manifest entries.
        output_path: Destination DocTags path.
        input_hash: Content hash used for resume detection.
        overwrite: Flag indicating whether existing outputs should be replaced.

    Examples:
        >>> HtmlTask(Path("/tmp/a.html"), "doc", Path("/tmp/doc.doctags"), "hash", False)
        HtmlTask(html_path=PosixPath('/tmp/a.html'), relative_id='doc', output_path=PosixPath('/tmp/doc.doctags'), input_hash='hash', overwrite=False)
    """

    html_path: Path
    relative_id: str
    output_path: Path
    input_hash: str
    overwrite: bool
    sanitizer_profile: str = "balanced"
    sanitizer_profile: str


@dataclass
class HtmlConversionResult:
    """Structured result emitted by worker processes.

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
    """

    doc_id: str
    status: str
    duration_s: float
    input_path: str
    input_hash: str
    output_path: str
    error: str | None = None
    sanitizer_profile: Optional[str] = None
    sanitizer_profile: Optional[str] = None


def _get_converter() -> "DocumentConverter":
    """Instantiate and cache a Docling HTML converter per worker process."""

    from docling.document_converter import DocumentConverter

    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = DocumentConverter(use_vision=False)
    return _CONVERTER


def list_htmls(root: Path) -> List[Path]:
    """Enumerate HTML-like files beneath a directory tree.

    Args:
        root: Directory whose subtree should be searched for HTML files.

    Returns:
        Sorted list of discovered HTML file paths excluding normalized outputs.
    """
    exts = {".html", ".htm", ".xhtml"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.endswith(".normalized.html"):
            out.append(p)
    return sorted(out)


def _sanitize_html_file(path: Path, profile: str) -> Tuple[Path, Optional[Path]]:
    """Apply sanitizer profile to ``path``, returning the path to convert."""

    profile = (profile or "balanced").lower()
    if profile == "permissive":
        return path, None

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return path, None

    sanitized = raw
    # Remove script/style blocks for all profiles except permissive
    sanitized = re.sub(r"<script\b.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r"<style\b.*?</style>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)

    if profile == "strict":
        sanitized = re.sub(r"<iframe\b.*?</iframe>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(
            r"<link\b[^>]*rel=\s*['\"]?stylesheet[^>]*>",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"<meta\b[^>]*>", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"<!--.*?-->", "", sanitized, flags=re.DOTALL)
        sanitized = re.sub(
            r"\s+on[a-zA-Z]+\s*=\s*(\".*?\"|'[^']*'|[^\s>]+)",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )

    if sanitized == raw:
        return path, None

    tmp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".sanitized.html",
        dir=path.parent,
    )
    with tmp_handle as handle:
        handle.write(sanitized)
        temp_path = Path(handle.name)
    return temp_path, temp_path


def html_convert_one(task: HtmlTask) -> HtmlConversionResult:
    """Convert a single HTML file to DocTags, honoring overwrite semantics.

    Args:
        task: Conversion details including paths, hash, and overwrite policy.

    Returns:
        :class:`ConversionResult` capturing the conversion status.

    Raises:
        ValueError: Propagated when Docling validation fails prior to internal handling.
    """

    start = time.perf_counter()
    try:
        out_path = task.output_path
        if out_path.exists() and not task.overwrite:
            return HtmlConversionResult(
                doc_id=task.relative_id,
                status="skip",
                duration_s=0.0,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                sanitizer_profile=task.sanitizer_profile,
            )

        sanitized_path, temp_path = _sanitize_html_file(task.html_path, task.sanitizer_profile)
        converter = _get_converter()
        try:
            result = converter.convert(sanitized_path, raises_on_error=False)
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        if result.document is None:
            return HtmlConversionResult(
                doc_id=task.relative_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error="empty-document",
                sanitizer_profile=task.sanitizer_profile,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        # Serialize under a lock to avoid partial writes when workers race
        try:
            with acquire_lock(out_path):
                if out_path.exists() and not task.overwrite:
                    return HtmlConversionResult(
                        doc_id=task.relative_id,
                        status="skip",
                        duration_s=time.perf_counter() - start,
                        input_path=str(task.html_path),
                        input_hash=task.input_hash,
                        output_path=str(out_path),
                        sanitizer_profile=task.sanitizer_profile,
                    )
                result.document.save_as_doctags(tmp_path)
                try:
                    tmp_path.replace(out_path)
                finally:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
        except TimeoutError as exc:
            return HtmlConversionResult(
                doc_id=task.relative_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error=str(exc),
                sanitizer_profile=task.sanitizer_profile,
            )
        return HtmlConversionResult(
            doc_id=task.relative_id,
            status="success",
            duration_s=time.perf_counter() - start,
            input_path=str(task.html_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
            sanitizer_profile=task.sanitizer_profile,
        )

    except Exception as exc:  # pragma: no cover - integration failure path
        return HtmlConversionResult(
            doc_id=task.relative_id,
            status="failure",
            duration_s=time.perf_counter() - start,
            input_path=str(task.html_path),
            input_hash=task.input_hash,
            output_path=str(task.output_path),
            error=str(exc),
            sanitizer_profile=task.sanitizer_profile,
        )


def html_main(args: argparse.Namespace | None = None) -> int:
    """Entrypoint for parallel HTML-to-DocTags conversion across a dataset.

    Args:
        args: Optional pre-parsed CLI namespace to override command-line inputs.

    Returns:
        Process exit code, where ``0`` denotes success.
    """

    bootstrap_root = detect_data_root()
    try:
        data_html(bootstrap_root)
        data_doctags(bootstrap_root)
    except Exception:
        pass

    if args is None:
        namespace = html_parse_args()
    elif isinstance(args, argparse.Namespace):
        namespace = args
        if getattr(namespace, "_cli_explicit_overrides", None) is None:
            keys = [name for name in vars(namespace) if not name.startswith("_")]
            annotate_cli_overrides(namespace, explicit=keys, defaults={})
    else:
        namespace = html_parse_args(args)

    cfg = DoctagsCfg.from_args(namespace, mode="html")
    config_snapshot = cfg.to_manifest()
    for field_def in fields(DoctagsCfg):
        setattr(namespace, field_def.name, getattr(cfg, field_def.name))

    log_level = cfg.log_level
    run_id = uuid.uuid4().hex
    logger = get_logger(
        __name__,
        level=str(log_level),
        base_fields={"run_id": run_id, "stage": HTML_MANIFEST_STAGE},
    )
    set_spawn_or_warn(logger)

    import multiprocessing as mp

    logger.info(
        "Multiprocessing configuration",
        extra={
            "extra_fields": {
                "start_method": mp.get_start_method(allow_none=True),
                "cpu_count": os.cpu_count(),
            }
        },
    )

    args = namespace

    ensure_docling_dependencies()

    data_root_override = namespace.data_root if hasattr(namespace, "data_root") else None
    resolved_root = cfg.data_root if cfg.data_root is not None else detect_data_root()

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    manifest_dir = data_manifests(resolved_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(cfg.input).resolve()
    output_dir = Path(cfg.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "HTML conversion configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "workers": cfg.workers,
                "http_timeout": [float(cfg.http_timeout[0]), float(cfg.http_timeout[1])],
            }
        },
    )

    config_snapshot.update(
        {
            "mode": cfg.mode,
            "data_root": str(resolved_root),
            "input": str(input_dir),
            "output": str(output_dir),
            "workers": int(cfg.workers),
            "resume": bool(cfg.resume),
            "force": bool(cfg.force),
            "overwrite": bool(cfg.overwrite),
            "html_sanitizer": cfg.html_sanitizer,
            "http_timeout": [float(cfg.http_timeout[0]), float(cfg.http_timeout[1])],
        }
    )

    telemetry_sink = TelemetrySink(
        resolve_attempts_path(HTML_MANIFEST_STAGE, resolved_root),
        resolve_manifest_path(HTML_MANIFEST_STAGE, resolved_root),
    )
    stage_telemetry = StageTelemetry(telemetry_sink, run_id=run_id, stage=HTML_MANIFEST_STAGE)
    with telemetry_scope(stage_telemetry):

        manifest_log_success(
            stage=HTML_MANIFEST_STAGE,
            doc_id="__config__",
            duration_s=0.0,
            schema_version="docparse/1.1.0",
            input_path=input_dir,
            input_hash="",
            output_path=output_dir,
            config=config_snapshot,
        )

        if cfg.force:
            logger.info(
                "Force mode: reprocessing all documents",
                extra={"extra_fields": {"mode": "force"}},
            )
        elif cfg.resume:
            logger.info(
                "Resume mode enabled: unchanged outputs will be skipped",
                extra={"extra_fields": {"mode": "resume"}},
            )

        files = list_htmls(input_dir)
        if not files:
            log_event(
                logger,
                "warning",
                "No HTML files found",
                stage=HTML_MANIFEST_STAGE,
                doc_id="__aggregate__",
                input_hash=None,
                error_code="NO_INPUT_FILES",
                input_dir=str(input_dir),
            )
            return 0

        manifest_index = (
            load_manifest_index(HTML_MANIFEST_STAGE, resolved_root) if cfg.resume else {}
        )
        resume_controller = ResumeController(cfg.resume, cfg.force, manifest_index)

        tasks: List[HtmlTask] = []
        ok = fail = skip = 0
        for path in files:
            rel_path = path.relative_to(input_dir)
            doc_id = rel_path.as_posix()
            out_path = (output_dir / rel_path).with_suffix(".doctags")
            input_hash = compute_content_hash(path)
            skip_doc, _ = resume_controller.should_skip(doc_id, out_path, input_hash)
            if skip_doc and not cfg.overwrite:
                log_event(
                    logger,
                    "info",
                    "Skipping HTML document",
                    stage=HTML_MANIFEST_STAGE,
                    doc_id=doc_id,
                    input_hash=input_hash,
                    output_path=str(out_path),
                )
                manifest_log_skip(
                    stage=HTML_MANIFEST_STAGE,
                    doc_id=doc_id,
                    input_path=path,
                    input_hash=input_hash,
                    output_path=out_path,
                    schema_version="docparse/1.1.0",
                    parse_engine="docling-html",
                    html_sanitizer=cfg.html_sanitizer,
                )
                skip += 1
                continue
            tasks.append(
                HtmlTask(
                    html_path=path,
                    relative_id=doc_id,
                    output_path=out_path,
                    input_hash=input_hash,
                    overwrite=cfg.overwrite,
                    sanitizer_profile=cfg.html_sanitizer,
                )
            )

        if not tasks:
            logger.info(
                "HTML conversion summary",
                extra={
                    "extra_fields": {
                        "ok": 0,
                        "skip": skip,
                        "fail": 0,
                    }
                },
            )
            return 0

        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [ex.submit(html_convert_one, task) for task in tasks]
            for fut in tqdm(
                as_completed(futures), total=len(futures), unit="file", desc="HTML → DocTags"
            ):
                result = fut.result()
                duration = round(result.duration_s, 3)
                if result.status == "success":
                    ok += 1
                    manifest_log_success(
                        stage=HTML_MANIFEST_STAGE,
                        doc_id=result.doc_id,
                        duration_s=duration,
                        schema_version="docparse/1.1.0",
                        input_path=result.input_path,
                        input_hash=result.input_hash,
                        output_path=result.output_path,
                        parse_engine="docling-html",
                        html_sanitizer=result.sanitizer_profile,
                    )
                elif result.status == "skip":
                    skip += 1
                    manifest_log_skip(
                        stage=HTML_MANIFEST_STAGE,
                        doc_id=result.doc_id,
                        input_path=result.input_path,
                        input_hash=result.input_hash,
                        output_path=result.output_path,
                        schema_version="docparse/1.1.0",
                        duration_s=duration,
                        parse_engine="docling-html",
                        html_sanitizer=result.sanitizer_profile,
                    )
                else:
                    fail += 1
                    log_event(
                        logger,
                        "error",
                        "HTML conversion failure",
                        stage=HTML_MANIFEST_STAGE,
                        doc_id=result.doc_id,
                        input_hash=result.input_hash,
                        error_code="HTML_CONVERSION_FAILED",
                        error=result.error or "conversion failed",
                    )
                    manifest_log_failure(
                        stage=HTML_MANIFEST_STAGE,
                        doc_id=result.doc_id,
                        duration_s=duration,
                        schema_version="docparse/1.1.0",
                        input_path=result.input_path,
                        input_hash=result.input_hash,
                        output_path=result.output_path,
                        error=result.error or "conversion failed",
                        parse_engine="docling-html",
                        html_sanitizer=result.sanitizer_profile,
                    )

        logger.info(
            "HTML conversion summary",
            extra={
                "extra_fields": {
                    "ok": ok,
                    "skip": skip,
                    "fail": fail,
                }
            },
        )

        return 0


# --- Docling Test Stubs ---

_DOCLING_STUB_INSTALLED = False


def _should_install_docling_test_stubs() -> bool:
    """Return ``True`` when docling stubs should be installed for tests."""

    if os.getenv("DOCSTOKG_ENFORCE_DOCLING") == "1":
        return False
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _ensure_stub_module(name: str, *, package: bool = False) -> types.ModuleType:
    """Register a lightweight stub module in :mod:`sys.modules` if missing."""

    module = sys.modules.get(name)
    if module is not None:
        return module
    module = types.ModuleType(name)
    module.__file__ = "<docling-stub>"
    module.__spec__ = None
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
        module.__package__ = name
    else:
        module.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
    sys.modules[name] = module
    return module


def _install_docling_test_stubs() -> None:
    """Install minimal docling/docling-core shims for unit tests."""

    global _DOCLING_STUB_INSTALLED
    if _DOCLING_STUB_INSTALLED:
        return

    _ensure_stub_module("docling", package=True)
    _ensure_stub_module("docling.backend", package=True)
    _ensure_stub_module("docling.datamodel", package=True)
    _ensure_stub_module("docling.datamodel.pipeline", package=True)
    _ensure_stub_module("docling.pipeline", package=True)
    _ensure_stub_module("docling_core")

    converter_mod = _ensure_stub_module("docling.document_converter")

    class DocumentConverter:
        """Stubbed ``docling`` converter preserving the real API surface."""

        def __init__(self, *args, **kwargs) -> None:
            """Record constructor arguments for verification in tests."""
            self.args = args
            self.kwargs = kwargs

        def convert(self, *_args, **_kwargs):
            """Raise to indicate conversions are not executed in stub mode."""

            raise RuntimeError("Docling stubs do not perform real conversions.")

    class PdfFormatOption:
        """Stub container matching ``docling`` PDF format options."""

        def __init__(self, *args, **kwargs) -> None:
            """Capture PDF format options passed by callers under test."""
            self.args = args
            self.kwargs = kwargs

    converter_mod.DocumentConverter = DocumentConverter  # type: ignore[attr-defined]
    converter_mod.PdfFormatOption = PdfFormatOption  # type: ignore[attr-defined]

    backend_mod = _ensure_stub_module("docling.backend.docling_parse_v4_backend")

    class DoclingParseV4DocumentBackend:  # pragma: no cover - stub only
        """Placeholder backend used when ``docling`` is unavailable."""

        pass

    backend_mod.DoclingParseV4DocumentBackend = DoclingParseV4DocumentBackend  # type: ignore[attr-defined]

    accel_mod = _ensure_stub_module("docling.datamodel.accelerator_options")

    class AcceleratorDevice:  # pragma: no cover - stub only
        """Minimal enum-like device sentinel for accelerator selection."""

        CUDA = "cuda"

    class AcceleratorOptions:  # pragma: no cover - stub only
        """Stub options matching the constructor signature of the real class."""

        def __init__(self, num_threads: int = 1, device: str | None = None, **_kwargs) -> None:
            """Persist accelerator selection hints used by doctests."""
            self.num_threads = num_threads
            self.device = device or AcceleratorDevice.CUDA

    accel_mod.AcceleratorDevice = AcceleratorDevice  # type: ignore[attr-defined]
    accel_mod.AcceleratorOptions = AcceleratorOptions  # type: ignore[attr-defined]

    base_mod = _ensure_stub_module("docling.datamodel.base_models")

    class _EnumValue:
        """Simple enum stand-in that preserves a string ``value`` attribute."""

        def __init__(self, value: str) -> None:
            """Store ``value`` for comparisons and repr output."""
            self.value = value

        def __repr__(self) -> str:  # pragma: no cover - trivial
            """Return the stored value for readable debugging output."""
            return self.value

        def __hash__(self) -> int:  # pragma: no cover - trivial
            """Compute a hash based on the underlying string value."""
            return hash(self.value)

        def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
            """Support equality checks against other ``_EnumValue`` instances."""
            if isinstance(other, _EnumValue):
                return self.value == other.value
            return False

    class ConversionStatus:  # pragma: no cover - stub only
        """Enum-like result state mirrors for Docling conversion outcomes."""

        SUCCESS = _EnumValue("success")
        PARTIAL_SUCCESS = _EnumValue("partial_success")

    class InputFormat:  # pragma: no cover - stub only
        """Enum-like input format sentinel used in Docling pipelines."""

        PDF = _EnumValue("pdf")

    base_mod.ConversionStatus = ConversionStatus  # type: ignore[attr-defined]
    base_mod.InputFormat = InputFormat  # type: ignore[attr-defined]

    pipeline_opts_mod = _ensure_stub_module("docling.datamodel.pipeline_options")

    class VlmPipelineOptions:  # pragma: no cover - stub only
        """Catch-all structure mimicking the real VLM pipeline options."""

        def __init__(self, **kwargs) -> None:
            """Store arbitrary keyword options for inspection in tests."""
            for key, value in kwargs.items():
                setattr(self, key, value)

    pipeline_opts_mod.VlmPipelineOptions = VlmPipelineOptions  # type: ignore[attr-defined]

    pipeline_model_mod = _ensure_stub_module("docling.datamodel.pipeline_options_vlm_model")

    class ApiVlmOptions:  # pragma: no cover - stub only
        """Stub capturing API VLM options forwarded by planners."""

        def __init__(self, **kwargs) -> None:
            """Persist VLM option values on the stub instance."""
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ResponseFormat:  # pragma: no cover - stub only
        """Response format sentinel mirroring Docling's enumeration."""

        DOCTAGS = "doctags"

    pipeline_model_mod.ApiVlmOptions = ApiVlmOptions  # type: ignore[attr-defined]
    pipeline_model_mod.ResponseFormat = ResponseFormat  # type: ignore[attr-defined]

    pipeline_mod = _ensure_stub_module("docling.pipeline.vlm_pipeline")

    class VlmPipeline:  # pragma: no cover - stub only
        """Placeholder VLM pipeline used during tests."""

        pass

    pipeline_mod.VlmPipeline = VlmPipeline  # type: ignore[attr-defined]

    _DOCLING_STUB_INSTALLED = True


# --- Entry Points ---

if __name__ == "__main__":
    import sys as _sys

    argv = _sys.argv[1:]
    if argv and argv[0] in {"--html", "--pipeline=html"}:
        if argv[0] == "--html":
            _sys.argv = [_sys.argv[0]] + argv[1:]
        else:  # --pipeline=html
            _sys.argv = [_sys.argv[0]] + [arg for arg in argv if not arg.startswith("--pipeline=")]
        raise SystemExit(html_main())

    if argv and argv[0] in {"--pdf", "--pipeline=pdf"}:
        if argv[0] == "--pdf":
            _sys.argv = [_sys.argv[0]] + argv[1:]
        else:  # --pipeline=pdf
            _sys.argv = [_sys.argv[0]] + [arg for arg in argv if not arg.startswith("--pipeline=")]
        raise SystemExit(pdf_main())

    raise SystemExit(pdf_main())
