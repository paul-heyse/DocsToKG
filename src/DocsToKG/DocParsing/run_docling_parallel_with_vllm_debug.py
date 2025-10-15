#!/usr/bin/env python3
"""
Start (or reuse) a local vLLM server for Granite-Docling, then run parallel Docling conversions.

Improvements:
- Port-smart: reuse healthy vLLM on 8000; else find another free port.
- Rich diagnostics: stream vLLM logs; print HTTP status and bodies from /v1/models and /metrics.
- tqdm progress bars for vLLM warmup and per-PDF conversion progress.
"""

import argparse
import os
import shutil
import socket
import subprocess as sp
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from DocsToKG.DocParsing._common import (
    compute_content_hash,
    data_doctags,
    data_manifests,
    data_pdfs,
    detect_data_root,
    find_free_port,
    get_logger,
    load_manifest_index,
    manifest_append,
)

warnings.warn(
    "Direct invocation of run_docling_parallel_with_vllm_debug.py is deprecated. "
    "Use unified CLI: python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf",
    DeprecationWarning,
    stacklevel=2,
)

_LOGGER = get_logger(__name__)


# -------- Paths --------
DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_INPUT = data_pdfs(DEFAULT_DATA_ROOT)
DEFAULT_OUTPUT = data_doctags(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "doctags-pdf"


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or "
            "$DOCSTOKG_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder with PDFs (recurses).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Folder for Doctags output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers for PDF conversion",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip PDFs whose DocTags already exist with matching content hash",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone execution.

    Args:
        argv: Optional CLI argument list. When ``None`` the values from
            :data:`sys.argv` are used.

    Returns:
        Namespace containing parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` detects invalid arguments.
    """

    return build_parser().parse_args(argv)


MODEL_PATH = "/home/paul/hf-cache/granite-docling-258M"  # local untied snapshot

# -------- Settings --------
PREFERRED_PORT = 8000
PORT_SCAN_SPAN = 32
DEFAULT_WORKERS = min(12, (os.cpu_count() or 16) - 4)
WAIT_TIMEOUT_S = 300

# Thread hygiene for CPU libs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("DOCLING_CUDA_USE_FLASH_ATTENTION2", "1")

ARTIFACTS = os.environ.get("DOCLING_ARTIFACTS_PATH", "")


# -------- Utilities --------
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

    Examples:
        >>> task = PdfTask(Path("/tmp/sample.pdf"), Path("/tmp/out"), 8000, "hash", "doc", Path("/tmp/out/doc.doctags"))
        >>> task.doc_id
        'doc'
    """

    pdf_path: Path
    output_dir: Path
    port: int
    input_hash: str
    doc_id: str
    output_path: Path

    def __getitem__(self, index: int) -> Path:
        """Provide tuple-like access for compatibility with legacy tests.

        Args:
            index: Position requested by tuple-style accessors.

        Returns:
            PDF path when ``index`` is ``0``.

        Raises:
            IndexError: If ``index`` is not ``0``.
        """

        if index == 0:
            return self.pdf_path
        raise IndexError("PdfTask only supports index 0")


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


def _normalize_status(raw: Optional[str]) -> str:
    """Coerce legacy status strings into the canonical vocabulary."""

    if not raw:
        return "success"

    normalized = raw.strip().lower()
    if normalized in {"ok", "success", "succeeded", "done"}:
        return "success"
    if normalized in {"skip", "skipped", "skipping"}:
        return "skip"
    if normalized in {"fail", "failed", "error", "exception"}:
        return "failure"
    return normalized


def _safe_float(value: Any) -> float:
    """Convert the supplied value to ``float`` when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def normalize_conversion_result(result: Any, task: Optional[PdfTask] = None) -> PdfConversionResult:
    """Adapt heterogeneous worker return values into :class:`PdfConversionResult`.

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
    """

    if isinstance(result, PdfConversionResult):
        return result

    if isinstance(result, dict):
        payload: Dict[str, Any] = dict(result)
        doc_id = payload.get("doc_id") or (task.doc_id if task else "unknown")
        status = _normalize_status(payload.get("status"))
        return PdfConversionResult(
            doc_id=str(doc_id),
            status=status,
            duration_s=_safe_float(payload.get("duration_s", 0.0)),
            input_path=str(payload.get("input_path") or (task.pdf_path if task else "")),
            input_hash=str(payload.get("input_hash") or (task.input_hash if task else "")),
            output_path=str(payload.get("output_path") or (task.output_path if task else "")),
            error=payload.get("error"),
        )

    if isinstance(result, (tuple, list)):
        doc_id = (
            result[0]
            if len(result) > 0 and result[0] is not None
            else (task.doc_id if task else "unknown")
        )
        status = _normalize_status(result[1] if len(result) > 1 else None)
        duration = _safe_float(result[2] if len(result) > 2 else 0.0)
        input_path = (
            result[3]
            if len(result) > 3 and result[3] is not None
            else (task.pdf_path if task else "")
        )
        output_path = (
            result[4]
            if len(result) > 4 and result[4] is not None
            else (task.output_path if task else "")
        )
        input_hash = (
            result[5]
            if len(result) > 5 and result[5] is not None
            else (task.input_hash if task else "")
        )
        error = result[6] if len(result) > 6 else None

        return PdfConversionResult(
            doc_id=str(doc_id),
            status=status,
            duration_s=duration,
            input_path=str(input_path),
            input_hash=str(input_hash),
            output_path=str(output_path),
            error=None if error is None else str(error),
        )

    # Fallback for unexpected return types
    doc_id = task.doc_id if task else "unknown"
    return PdfConversionResult(
        doc_id=doc_id,
        status="failure",
        duration_s=0.0,
        input_path=str(task.pdf_path if task else ""),
        input_hash=str(task.input_hash if task else ""),
        output_path=str(task.output_path if task else ""),
        error=f"Unsupported result type: {type(result)!r}",
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
    port: int, timeout=2.5
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
    try:
        r = requests.get(url, timeout=timeout)
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
    except Exception as e:
        return None, str(e), None


def probe_metrics(port: int, timeout=2.5) -> Tuple[bool, Optional[int]]:
    """Check whether the vLLM `/metrics` endpoint is healthy.

    Args:
        port: HTTP port where the vLLM server should expose metrics.
        timeout: Seconds to wait for the HTTP response before aborting.

    Returns:
        Tuple of `(is_healthy, status_code)` where `is_healthy` is True when the
        endpoint responds with HTTP 200.
    """
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        r = requests.get(url, timeout=timeout)
        return (r.status_code == 200), r.status_code
    except Exception:
        return (False, None)


def stream_logs(proc: sp.Popen, prefix="[vLLM] "):
    """Continuously stream stdout lines from a child process to the console.

    Args:
        proc: Running subprocess whose stdout should be tailed.
        prefix: Text prefix applied to each emitted log line for readability.

    Returns:
        None: This routine streams output for side effects only.
    """
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        s = line.rstrip()
        if s:
            _LOGGER.info(
                "vLLM stdout",
                extra={
                    "extra_fields": {
                        "source": "vllm",
                        "line": prefix + s,
                    }
                },
            )


def start_vllm(port: int) -> sp.Popen:
    """Launch a vLLM server process on the requested port.

    Args:
        port: Port on which the vLLM HTTP server should listen.

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
        MODEL_PATH,  # /home/paul/hf-cache/granite-docling-258M
        "--port",
        str(port),
        # One flag with multiple names works across versions (nargs='+'):
        "--served-model-name",
        "granite-docling-258M",
        "ibm-granite/granite-docling-258M",
        "--gpu-memory-utilization",
        "0.30",
        # If you do want to cap images, use JSON for newer vLLM:
        # "--limit-mm-per-prompt", '{"image": 1}',
    ]

    env = os.environ.copy()
    env.setdefault("VLLM_LOG_LEVEL", "INFO")  # INFO so we can see useful lines
    _LOGGER.info(
        "Starting vLLM",
        extra={"extra_fields": {"command": cmd, "env_log_level": env.get("VLLM_LOG_LEVEL")}},
    )
    proc = sp.Popen(cmd, env=env, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    # Start log thread
    t = threading.Thread(target=stream_logs, args=(proc,), daemon=True)
    t.start()
    return proc


def wait_for_vllm(port: int, proc: sp.Popen, timeout_s: int = WAIT_TIMEOUT_S):
    """Poll the vLLM server until `/v1/models` responds with success.

    Args:
        port: HTTP port where the server is expected to listen.
        proc: Subprocess handle representing the running vLLM instance.
        timeout_s: Maximum time in seconds to wait for readiness.

    Returns:
        None

    Raises:
        RuntimeError: If the server exits prematurely or fails to become ready
            within the allotted timeout.
    """
    _LOGGER.info(
        "Probing vLLM",
        extra={"extra_fields": {"port": port, "timeout_s": timeout_s}},
    )
    t0 = time.time()
    with tqdm(total=timeout_s, unit="s", desc="vLLM warmup", leave=True) as bar:
        while True:
            # If vLLM crashed, stop early and print the last lines
            if proc.poll() is not None:
                try:
                    leftover = proc.stdout.read() or ""
                    _LOGGER.error(
                        "vLLM exited while waiting",
                        extra={
                            "extra_fields": {
                                "port": port,
                                "tail": leftover[-800:],
                            }
                        },
                    )
                finally:
                    raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
            names, raw, status = probe_models(port)
            if status == 200:
                _LOGGER.info(
                    "vLLM models available",
                    extra={
                        "extra_fields": {
                            "port": port,
                            "models": names,
                        }
                    },
                )
                return
            # (…keep your existing logging, metrics probe, and tqdm update…)
            if int(time.time() - t0) >= timeout_s:
                raise RuntimeError(f"Timed out waiting for vLLM on port {port}")
            time.sleep(1)
            bar.update(1)


def stop_vllm(proc: Optional[sp.Popen], own: bool, grace=10):
    """Terminate a managed vLLM process if this script launched it.

    Args:
        proc: Subprocess handle returned by `start_vllm`, or None.
        own: Indicates whether the caller owns the process lifetime.
        grace: Seconds to wait for graceful shutdown before forcing exit.

    Returns:
        None
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


def ensure_vllm(preferred: int = PREFERRED_PORT) -> Tuple[int, Optional[sp.Popen], bool]:
    """Ensure a vLLM server is available, launching one when necessary.

    Args:
        preferred: Preferred TCP port for the server.

    Returns:
        Tuple containing `(port, process, owns_process)` where `process` is the
        managed subprocess handle (or None if reusing an existing server) and
        `owns_process` indicates whether the caller should terminate it.
    """
    # 1) If preferred is free, start there
    if port_is_free(preferred):
        proc = start_vllm(preferred)
        # ⬇️ pass the process to the waiter so it can bail if vLLM crashes
        wait_for_vllm(preferred, proc)
        return preferred, proc, True

    # 2) If something is already on preferred, reuse if it's vLLM (any models list)
    names, raw, status = probe_models(preferred)
    if status == 200:
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
    proc = start_vllm(alt)
    # ⬇️ same change here
    wait_for_vllm(alt, proc)
    return alt, proc, True


def list_pdfs(root: Path) -> List[Path]:
    """Collect PDF files under a directory recursively.

    Args:
        root: Directory whose subtree should be scanned for PDFs.

    Returns:
        Sorted list of paths to PDF files.
    """
    return sorted([p for p in root.rglob("*.pdf") if p.is_file()])


# -------- Docling worker --------
def convert_one(task: PdfTask) -> PdfConversionResult:
    """Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

    Args:
        task: Description of the conversion request, including paths and port.

    Returns:
        Populated :class:`PdfConversionResult` reporting success, skip, or failure.

    Raises:
        ValueError: Propagated when the underlying conversion libraries raise
            validation errors prior to being caught by this helper.
    """

    start = time.perf_counter()
    pdf_path = task.pdf_path
    out_dir = task.output_dir
    port = task.port
    out_path = task.output_path

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
                model="granite-docling-258M",  # or "ibm-granite/granite-docling-258M"
                max_tokens=4096,  # <-- IMPORTANT for vLLM
                skip_special_tokens=False,
                temperature=0.1,
                stop=["</doctag>", "<|end_of_text|>"],
            ),
            prompt="Convert this page to docling.",
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

        result.document.save_as_doctags(out_path)
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


# -------- Main --------
def main(args: argparse.Namespace | None = None) -> int:
    """Coordinate vLLM startup and parallel DocTags conversion.

    Args:
        args: Optional argument namespace injected during programmatic use.

    Returns:
        Process exit code, where ``0`` indicates success.
    """

    import multiprocessing as mp

    logger = get_logger(__name__)

    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Multiprocessing start method set to 'spawn' for CUDA safety")
    except RuntimeError:
        current_method = mp.get_start_method()
        if current_method != "spawn":
            logger.warning(
                "Could not force spawn mode; current method is '%s'. CUDA operations in workers may fail.",
                current_method,
            )
    logger.info(
        "Multiprocessing method: %s, CPU count: %s",
        mp.get_start_method(),
        os.cpu_count(),
    )
    logger.info(
        "Multiprocessing method established",
        extra={
            "extra_fields": {
                "start_method": mp.get_start_method(),
                "cpu_count": os.cpu_count(),
            }
        },
    )

    parser = build_parser()
    defaults = parser.parse_args([])
    provided = parse_args() if args is None else args
    for key, value in vars(provided).items():
        if value is not None:
            setattr(defaults, key, value)
    args = defaults

    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else DEFAULT_DATA_ROOT
    )

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    data_manifests(resolved_root)

    if args.input == DEFAULT_INPUT and data_root_override is not None:
        input_dir = data_pdfs(resolved_root)
    else:
        input_dir = (args.input or DEFAULT_INPUT).resolve()

    if args.output == DEFAULT_OUTPUT and data_root_override is not None:
        output_dir = data_doctags(resolved_root)
    else:
        output_dir = (args.output or DEFAULT_OUTPUT).resolve()

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "I/O configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "artifacts_cache": ARTIFACTS or "",
            }
        },
    )

    if args.force:
        logger.info("Force mode: reprocessing all documents")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged outputs will be skipped")

    port, proc, owns = ensure_vllm(PREFERRED_PORT)
    logger.info(
        "vLLM server ready",
        extra={
            "extra_fields": {
                "port": port,
                "owns_process": owns,
            }
        },
    )

    try:
        pdfs = list_pdfs(input_dir)
        if not pdfs:
            logger.warning(
                "No PDFs found",
                extra={"extra_fields": {"input_dir": str(input_dir)}},
            )
            return 0

        manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if args.resume else {}

        workers = max(1, int(args.workers))
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
            doc_id = pdf_path.relative_to(input_dir).as_posix()
            out_path = output_dir / (pdf_path.stem + ".doctags")
            input_hash = compute_content_hash(pdf_path)
            manifest_entry = manifest_index.get(doc_id)
            if (
                args.resume
                and not args.force
                and out_path.exists()
                and manifest_entry
                and manifest_entry.get("input_hash") == input_hash
            ):
                logger.info("Skipping %s: output exists and input unchanged", doc_id)
                manifest_append(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    status="skip",
                    duration_s=0.0,
                    schema_version="docparse/1.1.0",
                    input_path=str(pdf_path),
                    input_hash=input_hash,
                    output_path=str(out_path),
                    parse_engine="docling-vlm",
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
            future_map = {ex.submit(convert_one, task): task for task in tasks}
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
                        logger.error(
                            "Conversion failed",
                            extra={
                                "extra_fields": {
                                    "doc_id": result.doc_id,
                                    "error": result.error or "unknown",
                                }
                            },
                        )

                    manifest_append(
                        stage=MANIFEST_STAGE,
                        doc_id=result.doc_id,
                        status=result.status,
                        duration_s=round(result.duration_s, 3),
                        schema_version="docparse/1.1.0",
                        input_path=result.input_path,
                        input_hash=result.input_hash,
                        output_path=result.output_path,
                        error=result.error,
                        parse_engine="docling-vlm",
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


if __name__ == "__main__":
    raise SystemExit(main())
