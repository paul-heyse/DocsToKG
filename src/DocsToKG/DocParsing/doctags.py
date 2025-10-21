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
#       "id": "iter-sorted-paths",
#       "name": "_iter_sorted_paths",
#       "anchor": "function-iter-sorted-paths",
#       "kind": "function"
#     },
#     {
#       "id": "iter-directory-files",
#       "name": "_iter_directory_files",
#       "anchor": "function-iter-directory-files",
#       "kind": "function"
#     },
#     {
#       "id": "iter-pdfs",
#       "name": "iter_pdfs",
#       "anchor": "function-iter-pdfs",
#       "kind": "function"
#     },
#     {
#       "id": "peek-iterable",
#       "name": "_peek_iterable",
#       "anchor": "function-peek-iterable",
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
import hashlib
import heapq
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
from dataclasses import dataclass, fields
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import httpx
from tqdm import tqdm

from DocsToKG.DocParsing.config import (
    StageConfigBase,
    annotate_cli_overrides,
    parse_args_with_overrides,
)
from DocsToKG.DocParsing.core import (
    DEFAULT_HTTP_TIMEOUT,
    CLIOption,
    ItemFingerprint,
    ItemOutcome,
    ResumeController,
    StageContext,
    StageError,
    StageHooks,
    StageOptions,
    StageOutcome,
    StagePlan,
    WorkItem,
    safe_write,
    build_subcommand,
    derive_doc_id_and_doctags_path,
    find_free_port,
    get_http_session,
    normalize_http_timeout,
    run_stage,
    set_spawn_or_warn,
)
from DocsToKG.DocParsing.core.concurrency import _acquire_lock
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
    relative_path,
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

_T = TypeVar("_T")

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
DEFAULT_WORKERS = max(1, min(12, (os.cpu_count() or 16) - 4))
WAIT_TIMEOUT_S = 300
DEFAULT_GPU_MEMORY_UTILIZATION = 0.30
DEFAULT_SERVED_MODEL_NAMES: Tuple[str, ...] = (
    "granite-docling-258M",
    "ibm-granite/granite-docling-258M",
)


def _compute_pdf_cfg_hash(cfg: "DoctagsCfg") -> str:
    """Return a stable hash for resume fingerprinting of PDF conversions."""

    payload = {
        "model": str(cfg.model or ""),
        "served_model_names": tuple(cfg.served_model_names),
        "gpu_memory_utilization": float(cfg.gpu_memory_utilization),
        "vlm_prompt": str(cfg.vlm_prompt),
        "vlm_stop": tuple(cfg.vlm_stop or ()),
        "http_timeout": tuple(float(t) for t in cfg.http_timeout),
        "overwrite": False,  # PDFs never overwrite downstream outputs
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_html_cfg_hash(cfg: "DoctagsCfg") -> str:
    """Return a stable hash for resume fingerprinting of HTML conversions."""

    payload = {
        "html_sanitizer": str(cfg.html_sanitizer),
        "overwrite": bool(cfg.overwrite),
        "http_timeout": tuple(float(t) for t in cfg.http_timeout),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_pdf_plan(
    *,
    pdf_paths: Sequence[Path],
    input_dir: Path,
    output_dir: Path,
    resolved_root: Path,
    port: int,
    cfg_hash: str,
    hash_alg: str,
    resume_controller: ResumeController,
    logger,
    inference_model: str,
    served_model_names: Sequence[str],
    vllm_version: str,
    vlm_prompt: str,
    vlm_stop: Tuple[str, ...],
) -> tuple[StagePlan, int]:
    """Create a StagePlan for DocTags PDF conversion."""

    plan_items: list[WorkItem] = []
    resume_skipped = 0
    served_models_list = list(served_model_names)

    for pdf_path in pdf_paths:
        doc_id, out_path = derive_doc_id_and_doctags_path(pdf_path, input_dir, output_dir)
        input_hash = compute_content_hash(pdf_path, algorithm=hash_alg)
        skip_doc, _entry = resume_controller.should_skip(doc_id, out_path, input_hash)
        if skip_doc:
            log_event(
                logger,
                "info",
                "Skipping document: output exists and input unchanged",
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                input_relpath=relative_path(pdf_path, resolved_root),
                output_relpath=relative_path(out_path, resolved_root),
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
                served_models=served_models_list,
                vllm_version=vllm_version,
                reason="resume-satisfied",
            )
            resume_skipped += 1
            continue

        fingerprint_path = out_path.with_suffix(out_path.suffix + ".fp.json")
        task = PdfTask(
            pdf_path=pdf_path,
            output_dir=output_dir,
            port=port,
            input_hash=input_hash,
            doc_id=doc_id,
            output_path=out_path,
            served_model_names=tuple(served_model_names),
            inference_model=inference_model,
            vlm_prompt=vlm_prompt,
            vlm_stop=vlm_stop,
        )
        metadata: Dict[str, Any] = {
            "task": task,
            "doc_id": doc_id,
            "input_path": str(pdf_path),
            "output_path": str(out_path),
            "input_hash": input_hash,
            "hash_alg": hash_alg,
            "fingerprint_path": str(fingerprint_path),
            "parse_engine": "docling-vlm",
            "model_name": inference_model,
            "served_models": served_models_list,
            "vllm_version": vllm_version,
        }
        try:
            size = max(1.0, float(pdf_path.stat().st_size))
        except OSError:
            size = 1.0
        plan_items.append(
            WorkItem(
                item_id=doc_id,
                inputs={"pdf": pdf_path},
                outputs={"doctags": out_path},
                cfg_hash=cfg_hash,
                cost_hint=size,
                metadata=metadata,
                fingerprint=ItemFingerprint(
                    path=fingerprint_path,
                    input_sha256=input_hash,
                    cfg_hash=cfg_hash,
                ),
            )
        )

    plan = StagePlan(
        stage_name=MANIFEST_STAGE, items=tuple(plan_items), total_items=len(plan_items)
    )
    return plan, resume_skipped


def _pdf_stage_worker(item: WorkItem) -> ItemOutcome:
    """Worker entrypoint that proxies to :func:`pdf_convert_one`."""

    metadata = item.metadata
    task: PdfTask = metadata["task"]
    result = normalize_conversion_result(pdf_convert_one(task), task)

    manifest_extra = {
        "parse_engine": metadata.get("parse_engine", "docling-vlm"),
        "model_name": metadata.get("model_name"),
        "served_models": metadata.get("served_models"),
        "vllm_version": metadata.get("vllm_version"),
    }

    if result.status == "success":
        _write_fingerprint(
            Path(metadata["fingerprint_path"]),
            input_sha256=result.input_hash,
            cfg_hash=item.cfg_hash,
        )
        return ItemOutcome(
            status="success",
            duration_s=result.duration_s,
            manifest=manifest_extra,
            result={"status": "success"},
            error=None,
        )

    if result.status == "skip":
        return ItemOutcome(
            status="skip",
            duration_s=result.duration_s,
            manifest=manifest_extra,
            result={"reason": "worker-skip"},
            error=None,
        )

    error_message = result.error or "unknown error"
    manifest_extra["error"] = error_message
    err = StageError(
        stage=MANIFEST_STAGE,
        item_id=result.doc_id,
        category="runtime",
        message=error_message,
        retryable=False,
    )
    return ItemOutcome(
        status="failure",
        duration_s=result.duration_s,
        manifest=manifest_extra,
        result={},
        error=err,
    )


def _make_pdf_stage_hooks(
    *,
    logger,
    resolved_root: Path,
    resume_skipped: int,
) -> StageHooks:
    """Create lifecycle hooks for the PDF stage runner."""

    def before_stage(context: StageContext) -> None:
        context.metadata["logger"] = logger
        context.metadata["resolved_root"] = resolved_root
        context.metadata["schema_version"] = "docparse/1.1.0"
        context.metadata["resume_skipped"] = resume_skipped

    def after_item(
        item: WorkItem,
        outcome_or_error: Union[ItemOutcome, StageError],
        context: StageContext,
    ) -> None:
        stage_logger = context.metadata.get("logger", logger)
        root = context.metadata.get("resolved_root", resolved_root)
        schema_version = context.metadata.get("schema_version", "docparse/1.1.0")
        metadata = item.metadata
        doc_id = metadata["doc_id"]
        input_path = Path(metadata["input_path"])
        output_path = Path(metadata["output_path"])
        input_hash = metadata["input_hash"]
        hash_alg = metadata["hash_alg"]
        rel_fields = {
            "stage": MANIFEST_STAGE,
            "doc_id": doc_id,
            "input_relpath": metadata.get("input_relpath", relative_path(input_path, root)),
            "output_relpath": metadata.get("output_relpath", relative_path(output_path, root)),
        }

        if isinstance(outcome_or_error, ItemOutcome):
            payload = dict(outcome_or_error.manifest)
            payload.setdefault("parse_engine", metadata.get("parse_engine", "docling-vlm"))
            payload.setdefault("model_name", metadata.get("model_name"))
            payload.setdefault("served_models", metadata.get("served_models"))
            payload.setdefault("vllm_version", metadata.get("vllm_version"))

            if outcome_or_error.status == "success":
                log_event(
                    stage_logger,
                    "info",
                    "DocTags PDF written",
                    status="success",
                    elapsed_ms=int(outcome_or_error.duration_s * 1000),
                    **rel_fields,
                )
                manifest_log_success(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    duration_s=outcome_or_error.duration_s,
                    schema_version=schema_version,
                    input_path=input_path,
                    input_hash=input_hash,
                    output_path=output_path,
                    hash_alg=hash_alg,
                    **payload,
                )
                return

            if outcome_or_error.status == "skip":
                reason = outcome_or_error.result.get("reason", "resume-satisfied")
                log_event(
                    stage_logger,
                    "info",
                    "DocTags PDF skipped",
                    status="skip",
                    reason=reason,
                    **rel_fields,
                )
                manifest_log_skip(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    input_path=input_path,
                    input_hash=input_hash,
                    output_path=output_path,
                    hash_alg=hash_alg,
                    schema_version=schema_version,
                    reason=reason,
                    **payload,
                )
                return

            error_message = (
                outcome_or_error.error.message if outcome_or_error.error else "unknown error"
            )
            log_event(
                stage_logger,
                "error",
                "DocTags PDF conversion failed",
                status="failure",
                error=error_message,
                **rel_fields,
            )
            failure_payload = dict(payload)
            failure_payload["error"] = error_message
            manifest_log_failure(
                stage=MANIFEST_STAGE,
                doc_id=doc_id,
                duration_s=outcome_or_error.duration_s,
                schema_version=schema_version,
                input_path=input_path,
                input_hash=input_hash,
                output_path=output_path,
                hash_alg=hash_alg,
                error=error_message,
                **failure_payload,
            )
            return

        # Runner-surfaced error
        error = outcome_or_error
        log_event(
            stage_logger,
            "error",
            "DocTags PDF conversion failed",
            status="failure",
            error=error.message,
            **rel_fields,
        )
        manifest_log_failure(
            stage=MANIFEST_STAGE,
            doc_id=doc_id,
            duration_s=0.0,
            schema_version=schema_version,
            input_path=input_path,
            input_hash=input_hash,
            output_path=output_path,
            hash_alg=hash_alg,
            error=error.message,
        )

    def after_stage(outcome: StageOutcome, context: StageContext) -> None:
        stage_logger = context.metadata.get("logger", logger)
        pre_skipped = int(context.metadata.get("resume_skipped", 0))
        total_skipped = outcome.skipped + pre_skipped
        log_event(
            stage_logger,
            "info",
            "DocTags PDF summary",
            scheduled=outcome.scheduled,
            succeeded=outcome.succeeded,
            failed=outcome.failed,
            skipped=total_skipped,
            cancelled=outcome.cancelled,
            wall_ms=round(outcome.wall_ms, 3),
            stage=MANIFEST_STAGE,
            doc_id="__summary__",
        )

    return StageHooks(
        before_stage=before_stage,
        after_item=after_item,
        after_stage=after_stage,
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
        utilization = float(self.gpu_memory_utilization)
        if not 0.0 <= utilization <= 1.0:
            raise ValueError(
                "gpu_memory_utilization must be between 0.0 and 1.0 (inclusive); "
                f"received {utilization}"
            )
        self.gpu_memory_utilization = utilization


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
            "help": "Fraction of GPU memory the vLLM server may allocate (0.0-1.0)",
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
