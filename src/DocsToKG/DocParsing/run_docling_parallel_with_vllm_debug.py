#!/usr/bin/env python3
"""
Start (or reuse) a local vLLM server for Granite-Docling, then run parallel Docling conversions.

Improvements:
- Port-smart: reuse healthy vLLM on 8000; else find another free port.
- Rich diagnostics: stream vLLM logs; print HTTP status and bodies from /v1/models and /metrics.
- tqdm progress bars for vLLM warmup and per-PDF conversion progress.
"""

import os
import sys
import time
import argparse
import json
import socket
import shutil
import signal
import threading
import subprocess as sp
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
from tqdm import tqdm


# -------- Paths --------
def find_data_root(start: Path) -> Path:
    """
    Walk up from `start` to find a directory that has Data/PDFs.
    Returns `start/"Data"` if nothing matches above.
    """
    start = start.resolve()  # normalize before walking upward  # docs: pathlib.resolve()  :contentReference[oaicite:1]{index=1}
    for anc in (start, *start.parents):
        candidate = anc / "Data"
        if (candidate / "PDFs").is_dir():
            return candidate
    return start / "Data"


# Optional: env override
ENV_DATA_ROOT = os.getenv(
    "DOCSTOKG_DATA_ROOT"
)  # set to /home/paul/DocsToKG/src/DocsToKG/Data if you like  :contentReference[oaicite:2]{index=2}

# Compute defaults
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(ENV_DATA_ROOT) if ENV_DATA_ROOT else find_data_root(SCRIPT_DIR)
DEFAULT_INPUT = DEFAULT_DATA_ROOT / "PDFs"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "DocTagsFiles"

# CLI (still allows explicit overrides)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=Path, default=DEFAULT_INPUT, help="Folder with PDFs (recurses)."
)
parser.add_argument(
    "--output", type=Path, default=DEFAULT_OUTPUT, help="Folder for Doctags output."
)
args = parser.parse_args()

INPUT_DIR = args.input
OUTPUT_DIR = args.output

print(f"Resolved Data root: {DEFAULT_DATA_ROOT}")
print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) != 0


def probe_models(
    port: int, timeout=2.5
) -> Tuple[Optional[List[str]], Optional[str], Optional[int]]:
    """Return (names, raw_text, status) from GET /v1/models, or (None, raw, status) on failure."""
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
    """Return (ok, status) from /metrics; OK if 200."""
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        r = requests.get(url, timeout=timeout)
        return (r.status_code == 200), r.status_code
    except Exception:
        return (False, None)


def find_free_port(start: int, span: int = PORT_SCAN_SPAN) -> int:
    for p in range(start, start + span):
        if port_is_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # bind-to-0 to get a free ephemeral port


def stream_logs(proc: sp.Popen, prefix="[vLLM] "):
    """Continuously read vLLM stdout and print lines."""
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        s = line.rstrip()
        if s:
            print(prefix + s)


def start_vllm(port: int) -> sp.Popen:
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
    """Return (port, process, owns_process)."""
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
    return sorted([p for p in root.rglob("*.pdf") if p.is_file()])


# -------- Docling worker --------
def convert_one(args):
    pdf_path, out_dir, port = args
    try:
        from pathlib import Path
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.backend.docling_parse_v2_backend import (
            DoclingParseV2DocumentBackend,
        )  # fallback

        # Imports (exact modules)
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
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
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    if ARTIFACTS:
        print(f"Artifacts cache: {ARTIFACTS}")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure/launch vLLM
    print("Ensuring vLLM server...")
    port, proc, owns = ensure_vllm(PREFERRED_PORT)
    print(f"Using vLLM on port {port} (owns_process={owns})")

    try:
        pdfs = list_pdfs(INPUT_DIR)
        if not pdfs:
            print("No PDFs found. Exiting.")
            return

        print(f"Found {len(pdfs)} PDFs. Launching {DEFAULT_WORKERS} workers...")
        tasks = [(p, OUTPUT_DIR, port) for p in pdfs]
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
                        print(f"[FAIL] {name}: {status}")
                    pbar.update(1)

        print(f"\nDone. ok={ok}, skip={skip}, fail={fail}")
    finally:
        stop_vllm(proc, owns, grace=10)
        print("All done.")


if __name__ == "__main__":
    main()
