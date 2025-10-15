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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tqdm import tqdm

from DocsToKG.DocParsing._common import (
    data_doctags,
    data_pdfs,
    detect_data_root,
    find_free_port,
    get_logger,
)


# -------- Paths --------
DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_INPUT = data_pdfs(DEFAULT_DATA_ROOT)
DEFAULT_OUTPUT = data_doctags(DEFAULT_DATA_ROOT)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the DocTags conversion pipeline."""

    parser = argparse.ArgumentParser()
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
    return parser.parse_args()

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
        None
    """
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        s = line.rstrip()
        if s:
            print(prefix + s)


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
        print("ERROR: 'vllm' not found on PATH.", file=sys.stderr)
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
    print("Starting vLLM:", " ".join(cmd))
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
    print(f"Probing vLLM on port {port} for up to {timeout_s}s ...")
    t0 = time.time()
    with tqdm(total=timeout_s, unit="s", desc="vLLM warmup", leave=True) as bar:
        while True:
            # If vLLM crashed, stop early and print the last lines
            if proc.poll() is not None:
                try:
                    leftover = proc.stdout.read() or ""
                    print("[vLLM exit]", leftover[-800:])
                finally:
                    raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
            names, raw, status = probe_models(port)
            if status == 200:
                print(f"  /v1/models -> 200; models reported: {names}")
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
    print("Stopping vLLM...")
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
        print(f"Reusing existing vLLM on {preferred}; models={names}")
        return preferred, None, False

    # 3) Otherwise, pick a new free port
    alt = find_free_port(preferred + 1, PORT_SCAN_SPAN)
    print(f"Port {preferred} busy (not vLLM). Launching on {alt} instead.")
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
def convert_one(args):
    """Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

    Args:
        args: Tuple containing `(pdf_path, output_dir, port)` for the work item.

    Returns:
        Tuple of `(pdf_name, status)` where status is one of `ok`, `skip`, or a
        `fail:<reason>` string describing the conversion issue.
    """
    pdf_path, out_dir, port = args
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
        out_path = out_dir / (pdf_path.stem + ".doctags")
        if out_path.exists():
            return (pdf_path.name, "skip")

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
            return (pdf_path.name, f"fail:{detail}")

        if result.document is None:
            return (pdf_path.name, "fail:empty-document")

        result.document.save_as_doctags(out_path)
        return (pdf_path.name, "ok")
    except Exception as e:
        return (pdf_path.name, f"fail:{e}")


# -------- Main --------
def main():
    """Entrypoint that coordinates vLLM setup and parallel DocTags conversion.

    Args:
        None

    Returns:
        None
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

    args = parse_args()
    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else DEFAULT_DATA_ROOT
    )

    if args.input == DEFAULT_INPUT and data_root_override is not None:
        input_dir = data_pdfs(resolved_root)
    else:
        input_dir = args.input.resolve()

    if args.output == DEFAULT_OUTPUT and data_root_override is not None:
        output_dir = data_doctags(resolved_root)
    else:
        output_dir = args.output.resolve()

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
            return

        logger.info(
            "Launching workers",
            extra={
                "extra_fields": {
                    "pdf_count": len(pdfs),
                    "workers": DEFAULT_WORKERS,
                }
            },
        )
        tasks = [(p, output_dir, port) for p in pdfs]
        ok = fail = skip = 0
        with ProcessPoolExecutor(max_workers=DEFAULT_WORKERS) as ex:
            futures = [ex.submit(convert_one, t) for t in tasks]
            with tqdm(total=len(futures), desc="Converting PDFs", unit="file") as pbar:
                for fut in as_completed(futures):
                    name, status = fut.result()
                    if status == "ok":
                        ok += 1
                    elif status == "skip":
                        skip += 1
                    else:
                        fail += 1
                        logger.error(
                            "Conversion failed",
                            extra={
                                "extra_fields": {
                                    "doc_id": name,
                                    "status": status,
                                }
                            },
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


if __name__ == "__main__":
    main()
