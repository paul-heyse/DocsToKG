#!/usr/bin/env python3
"""
HTML → DocTags with robust image handling + captioning & classification (Docling).

- Input :  Data/HTML/        (recurses)
- Output:  Data/DocTagsFiles/*.doctags
- Images:  Data/ImagesCache/ (downloaded/decoded assets for stable local references)

Captioning options:
  --caption-model granite     # local Granite Vision 3.3-2b via Transformers (no vLLM)
  --granite-path /path/to/local/model/dir
  --hf-cache /home/paul/hf-cache  --offline

  --caption-model smolvlm     # local tiny captioner

  --caption-model api         # remote OpenAI-style endpoint (e.g., your vLLM)
  --api-url http://localhost:8000/v1/chat/completions --api-model ibm-granite/granite-vision-3.3-2b
  --api-headers '{"Authorization":"Bearer ..."}'

Classification:
  --classify                  # enable DocumentFigureClassifier (charts/diagrams/photos, etc.)

Accelerator:
  --device auto|cpu|cuda|mps  # default: auto
  --threads N                 # CPU worker threads for models

Notes
- This uses Docling’s HTML backend + Simple/Convert pipeline; picture enrichment is configured
  via ConvertPipelineOptions (do_picture_description / do_picture_classification +
  PictureDescription{Vlm,Api}Options). :contentReference[oaicite:1]{index=1}
- HTML is normalized to ensure ALL images are visible to the pipeline (lazy-load, srcset,
  CSS backgrounds, OpenGraph figures, base64 data URIs).
- HF offline/caching envs are honored when --offline/--hf-cache are set. :contentReference[oaicite:2]{index=2}
"""

import os
import re
import json
import base64
import hashlib
import mimetypes
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -------- Docling imports
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling.backend.html_backend import (
    HTMLDocumentBackend,
)  # explicit backend is fine :contentReference[oaicite:3]{index=3}
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
)
from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
    AcceleratorDevice,
)  # :contentReference[oaicite:4]{index=4}

# ---------- Defaults & CLI ----------


def detect_data_root(start: Path) -> Path:
    for anc in (start, *start.parents):
        if (anc / "Data" / "HTML").is_dir():
            return anc / "Data"
    return start / "Data"


ENV_DATA_ROOT = os.getenv("DOCSTOKG_DATA_ROOT")
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(ENV_DATA_ROOT) if ENV_DATA_ROOT else detect_data_root(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=Path, default=DATA_ROOT / "HTML", help="Root folder with HTML files."
)
parser.add_argument(
    "--output",
    type=Path,
    default=DATA_ROOT / "DocTagsFiles",
    help="Destination for .doctags files.",
)
parser.add_argument(
    "--images-dir",
    type=Path,
    default=DATA_ROOT / "ImagesCache",
    help="Cache folder for downloaded images.",
)
parser.add_argument(
    "--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2), help="Process workers."
)

# Enrichments
parser.add_argument(
    "--classify", action="store_true", help="Enable DocumentFigureClassifier (picture classes)."
)
parser.add_argument(
    "--caption-model",
    choices=["granite", "smolvlm", "api"],
    default="granite",
    help="Choose captioning backend: local Granite, local SmolVLM, or remote API.",
)
parser.add_argument(
    "--caption-prompt",
    type=str,
    default="Describe the image in two concise, factual sentences.",
    help="Prompt for caption generation.",
)
parser.add_argument(
    "--min-pic-area",
    type=float,
    default=0.0,
    help="Minimum picture area fraction (0..1) to process; 0.0 captions even small images. "
    "Applied on PictureDescription options. See picture_area_threshold. :contentReference[oaicite:5]{index=5}",
)

# Granite local controls
parser.add_argument(
    "--granite-path",
    type=str,
    default="",
    help="Local directory for ibm-granite/granite-vision-3.3-2b (use this to avoid Hub).",
)
parser.add_argument(
    "--hf-cache",
    type=str,
    default=str(DATA_ROOT.parent / "hf-cache"),
    help="HF cache dir used when loading models.",
)
parser.add_argument(
    "--offline",
    action="store_true",
    help="Set HF offline envs to avoid network access. :contentReference[oaicite:6]{index=6}",
)

# Remote API (vLLM/Ollama/OpenAI-style)
parser.add_argument(
    "--api-url",
    type=str,
    default="http://localhost:8000/v1/chat/completions",
    help="Remote VLM endpoint (OpenAI-compatible).",
)
parser.add_argument("--api-model", type=str, default="", help="Remote model name for the endpoint.")
parser.add_argument(
    "--api-headers",
    type=str,
    default="{}",
    help='JSON dict of HTTP headers, e.g. {"Authorization":"Bearer ..."}',
)
parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout for remote captioning.")

# Accelerator
parser.add_argument(
    "--device",
    choices=["auto", "cpu", "cuda", "mps"],
    default="auto",
    help="Target device for enrichment models (AUTO/CPU/CUDA/MPS). :contentReference[oaicite:7]{index=7}",
)
parser.add_argument(
    "--threads",
    type=int,
    default=max(1, (os.cpu_count() or 8) - 2),
    help="CPU threads budget for models.",
)

# Behavior
parser.add_argument("--overwrite", action="store_true", help="Recreate .doctags even if it exists.")
args = parser.parse_args()

INPUT_DIR: Path = args.input
OUTPUT_DIR: Path = args.output
IMAGES_DIR: Path = args.images_dir
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Respect HF offline/cache preferences (propagates to workers)
if args.offline:
    os.environ["HF_HUB_OFFLINE"] = (
        "1"  # hub offline (no HTTP)  :contentReference[oaicite:8]{index=8}
    )
    os.environ["TRANSFORMERS_OFFLINE"] = (
        "1"  # transformers offline   :contentReference[oaicite:9]{index=9}
    )
if args.hf_cache:
    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_cache)

# Keep CPU libs polite
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------- HTML normalization to surface all images ----------

LAZY_ATTRS = ["data-src", "data-original", "data-lazy", "data-zoom", "data-zoom-image", "data-url"]
SRCSET_ATTRS = ["srcset", "data-srcset"]
STYLE_URL_RE = re.compile(r"url\(['\"]?(?P<url>[^)'\"]+)['\"]?\)", re.IGNORECASE)


def sha1_name(payload: bytes | str, ext_hint: str = "") -> str:
    h = hashlib.sha1(payload if isinstance(payload, bytes) else payload.encode("utf-8")).hexdigest()
    return f"{h}{ext_hint}"


def pick_src_from_srcset(srcset_val: str) -> str:
    best = None
    for candidate in srcset_val.split(","):
        url_part, *desc = candidate.strip().split()
        score = 0.0
        if desc:
            d = desc[0].lower()
            if d.endswith("x"):
                try:
                    score = float(d[:-1])
                except:
                    score = 1.0
            elif d.endswith("w"):
                try:
                    score = float(d[:-1]) / 1000.0
                except:
                    score = 1.0
        best = (score, url_part) if best is None or score > best[0] else best
    return best[1] if best else srcset_val.split(",")[0].strip()


def guess_ext(mime: str) -> str:
    return mimetypes.guess_extension(mime) or (".png" if mime.startswith("image/") else "")


def download_or_decode_image(src: str, base_url: str | None, out_dir: Path, timeout=30) -> str:
    # data: URI
    if src.startswith("data:"):
        header, b64 = src.split(",", 1)
        mime = header.split(";")[0].split(":", 1)[1] if ":" in header else "image/png"
        ext = guess_ext(mime) or ".bin"
        payload = base64.b64decode(b64)
        fn = sha1_name(payload, ext)
        dst = out_dir / fn
        if not dst.exists():
            dst.write_bytes(payload)
        return str(dst)

    # URL (absolute/relative/scheme-less)
    url = urljoin(base_url, src) if base_url else src
    if url.startswith("//"):
        url = "https:" + url

    # Fetch
    ext = Path(urlparse(url).path).suffix or ".png"
    fn = sha1_name(url, ext)
    dst = out_dir / fn
    if not dst.exists():
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return str(dst)


def find_base_url(soup: BeautifulSoup, html_path: Path) -> str | None:
    base_tag = soup.find("base", href=True)
    if base_tag:
        return base_tag["href"]
    link_canon = soup.find("link", rel=lambda v: v and "canonical" in v.lower(), href=True)
    if link_canon:
        return link_canon["href"]
    meta_og = soup.find(
        "meta", property=lambda v: v and v.lower() in ("og:url", "prism.url"), content=True
    )
    if meta_og:
        return meta_og["content"]
    if html_path.exists():
        return f"file://{html_path.parent.as_posix()}/"
    return None


def promote_og_images(soup: BeautifulSoup) -> list[str]:
    urls = []
    for p in ("og:image", "og:image:secure_url"):
        for m in soup.find_all("meta", attrs={"property": p}):
            if m.get("content"):
                urls.append(m["content"])
    return urls


def extract_bg_images(soup: BeautifulSoup) -> list[str]:
    urls = []
    for tag in soup.find_all(style=True):
        for m in STYLE_URL_RE.finditer(tag["style"]):
            urls.append(m.group("url"))
    return urls


def normalize_html_images(html_path: Path, images_dir: Path) -> Path:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    base_url = find_base_url(soup, html_path)

    # Fix <img> lazy-loading + srcset
    for img in soup.find_all("img"):
        if not img.get("src"):
            for la in LAZY_ATTRS:
                if img.get(la):
                    img["src"] = img[la]
                    break
        for sa in SRCSET_ATTRS:
            if img.get(sa):
                img["src"] = pick_src_from_srcset(img[sa])
                break

    # Materialize <picture>/<source>
    for picture in soup.find_all("picture"):
        if picture.find("img"):
            continue
        src_candidate = None
        for sa in SRCSET_ATTRS:
            src_node = picture.find("source", attrs={sa: True})
            if src_node:
                src_candidate = pick_src_from_srcset(src_node[sa])
                break
        if src_candidate:
            ni = soup.new_tag("img")
            ni["src"] = src_candidate
            picture.append(ni)

    # CSS backgrounds → hidden <img>
    for url in extract_bg_images(soup):
        collector = soup.find("div", id="__bg_images__")
        if collector is None and soup.body:
            collector = soup.new_tag("div", id="__bg_images__")
            collector["style"] = "display:none"
            soup.body.append(collector)
        if collector is not None:
            ni = soup.new_tag("img")
            ni["src"] = url
            collector.append(ni)

    # OpenGraph figures → hidden <img>
    og_urls = promote_og_images(soup)
    if og_urls:
        og_div = soup.find(id="__og_images__")
        if og_div is None and soup.body:
            og_div = soup.new_tag("div", id="__og_images__")
            og_div["style"] = "display:none"
            soup.body.append(og_div)
        if og_div is not None:
            for u in og_urls:
                ni = soup.new_tag("img")
                ni["src"] = u
                og_div.append(ni)

    # Download all images and rewrite as file:// URIs
    for img in soup.find_all("img"):
        if not img.get("src"):
            continue
        local = download_or_decode_image(img["src"], base_url, images_dir)
        img["src"] = Path(local).absolute().as_uri()

    norm_path = html_path.with_suffix(".normalized.html")
    norm_path.write_text(str(soup), encoding="utf-8")
    return norm_path


# ---------- Captioning options helpers ----------


def make_accel(device: str, threads: int) -> AcceleratorOptions:
    dev_map = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }
    return AcceleratorOptions(
        num_threads=threads, device=dev_map.get(device, AcceleratorDevice.AUTO)
    )


def make_vlm_options_for_granite(local_path: str, hf_cache: str, prompt: str, area: float):
    # Use local path (preferred) or fall back to HF repo + cache. :contentReference[oaicite:10]{index=10}
    if local_path:
        return PictureDescriptionVlmOptions(
            repo_id=local_path, prompt=prompt, picture_area_threshold=area
        )
    return PictureDescriptionVlmOptions(
        repo_id="ibm-granite/granite-vision-3.3-2b",  # built-in preset points here in current docs
        repo_cache_folder=hf_cache,  # load from your cache dir
        prompt=prompt,
        picture_area_threshold=area,
    )  # :contentReference[oaicite:11]{index=11}


def make_vlm_options_for_smol(prompt: str, area: float):
    return PictureDescriptionVlmOptions(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        prompt=prompt,
        picture_area_threshold=area,
    )  # :contentReference[oaicite:12]{index=12}


def make_api_options(url: str, model: str, headers: dict, prompt: str, area: float, timeout: int):
    return PictureDescriptionApiOptions(
        url=url,
        params={"model": model} if model else {},
        headers=headers or {},
        timeout=timeout,
        prompt=prompt,
        picture_area_threshold=area,
    )  # requires enable_remote_services=True :contentReference[oaicite:13]{index=13}


# ---------- Conversion worker ----------

HTML_EXTS = {".html", ".htm", ".xhtml"}


def list_htmls(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in HTML_EXTS])


def convert_one(
    html_path: Path,
    out_dir: Path,
    caption_model: str,
    classify: bool,
    granite_path: str,
    hf_cache: str,
    offline: bool,
    api_url: str,
    api_model: str,
    api_headers: dict,
    timeout: int,
    device: str,
    threads: int,
    caption_prompt: str,
    min_pic_area: float,
    overwrite: bool,
) -> tuple[str, str]:
    try:
        # Normalize HTML and surface all images
        norm_html = normalize_html_images(html_path, IMAGES_DIR)

        # Pipeline options (Convert pipeline for HTML) :contentReference[oaicite:14]{index=14}
        pl_opts = ConvertPipelineOptions()
        pl_opts.accelerator_options = make_accel(device, threads)
        pl_opts.do_picture_description = True
        pl_opts.do_picture_classification = bool(classify)

        # Captioning backend
        if caption_model == "granite":
            pl_opts.picture_description_options = make_vlm_options_for_granite(
                local_path=granite_path,
                hf_cache=hf_cache,
                prompt=caption_prompt,
                area=min_pic_area,
            )
        elif caption_model == "smolvlm":
            pl_opts.picture_description_options = make_vlm_options_for_smol(
                prompt=caption_prompt, area=min_pic_area
            )
        else:
            pl_opts.enable_remote_services = True
            pl_opts.picture_description_options = make_api_options(
                url=api_url,
                model=api_model,
                headers=api_headers,
                prompt=caption_prompt,
                area=min_pic_area,
                timeout=timeout,
            )  # :contentReference[oaicite:15]{index=15}

        # Build converter for HTML
        converter = DocumentConverter(
            format_options={
                InputFormat.HTML: HTMLFormatOption(
                    backend=HTMLDocumentBackend,  # explicit backend is fine
                    pipeline_options=pl_opts,
                )
            }
        )

        result = converter.convert(norm_html, raises_on_error=False)
        if result.document is None:
            return (html_path.name, "fail: empty-document")

        out_path = out_dir / (html_path.stem + ".doctags")
        if out_path.exists() and not overwrite:
            return (html_path.name, "skip")

        result.document.save_as_doctags(out_path)
        return (html_path.name, "ok")

    except Exception as e:
        return (html_path.name, f"fail: {e}")


# ---------- Main ----------


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images: {IMAGES_DIR}")
    print(f"Workers: {args.workers}")
    print(
        f"Caption model: {args.caption_model}  | classify={args.classify}  | device={args.device}"
    )

    # parse remote headers once
    try:
        api_headers = json.loads(args.api_headers.strip() or "{}")
    except Exception:
        api_headers = {}

    htmls = list_htmls(INPUT_DIR)
    if not htmls:
        print("No HTML files found. Exiting.")
        return

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                convert_one,
                p,
                OUTPUT_DIR,
                args.caption_model,
                args.classify,
                args.granite_path,
                args.hf_cache,
                args.offline,
                args.api_url,
                args.api_model,
                api_headers,
                args.timeout,
                args.device,
                args.threads,
                args.caption_prompt,
                args.min_pic_area,
                args.overwrite,
            )
            for p in htmls
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), unit="file", desc="HTML → DocTags"
        ):
            name, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[FAIL] {name}: {status}")

    print(f"\nDone. ok={ok}, skip={skip}, fail={fail}")


if __name__ == "__main__":
    main()
