#!/usr/bin/env python3
"""
HTML → DocTags with robust picture handling + Docling captioning & classification.

- Input :  Data/HTML  (recurses)
- Output:  Data/DocTagsFiles/*.doctags
- Models:  Built-in DocumentFigureClassifier + (choose) Granite Vision, SmolVLM, or a remote vLLM API
- Notes :
    * We normalize HTML to surface every kind of image representation (img/src, data-src, srcset,
      CSS background-image, og:image, data: URIs), download to a local images dir, and rewrite the HTML.
    * For captioning, select one of:
        --caption-model granite|smolvlm|api
      When using --caption-model api, set --api-url and optional --api-headers/--api-model.
"""

import os, re, base64, hashlib, mimetypes, argparse, json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- Docling imports
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PictureDescriptionApiOptions,
    granite_picture_description,  # Granite Vision
    smolvlm_picture_description,  # SmolVLM
)
# Optional: control export image embedding when exporting MD/HTML later
# from docling_core.types.doc import ImageRefMode

# ---------- CLI ----------


def default_data_root(start: Path) -> Path:
    for anc in (start, *start.parents):
        if (anc / "Data" / "HTML").is_dir():
            return anc / "Data"
    return start / "Data"


ENV_DATA_ROOT = os.getenv("DOCSTOKG_DATA_ROOT")
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(ENV_DATA_ROOT) if ENV_DATA_ROOT else default_data_root(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=DATA_ROOT / "HTML")
parser.add_argument("--output", type=Path, default=DATA_ROOT / "DocTagsFiles")
parser.add_argument("--images-dir", type=Path, default=DATA_ROOT / "ImagesCache")
parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
parser.add_argument(
    "--caption-model",
    choices=["granite", "smolvlm", "api"],
    default="granite",
    help="Granite Vision (local), SmolVLM (local), or a remote API (e.g., vLLM).",
)
parser.add_argument(
    "--api-url",
    type=str,
    default="http://localhost:8000/v1/chat/completions",
    help="Remote vision model endpoint (if --caption-model api).",
)
parser.add_argument(
    "--api-model",
    type=str,
    default="",
    help="Remote model name for vLLM/OpenAI-compatible servers (if --caption-model api).",
)
parser.add_argument(
    "--api-headers",
    type=str,
    default="{}",
    help='JSON string of HTTP headers for remote API (e.g., {"Authorization":"Bearer ..."}).',
)
parser.add_argument(
    "--classify",
    action="store_true",
    help="Enable DocumentFigureClassifier-based picture classes (charts/diagrams/photos/etc.).",
)
parser.add_argument("--timeout", type=int, default=90)
args = parser.parse_args()

INPUT_DIR: Path = args.input
OUTPUT_DIR: Path = args.output
IMAGES_DIR: Path = args.images_dir
WORKERS = args.workers

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep CPU libraries well-behaved per worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------- HTML normalization utilities ----------

LAZY_ATTRS = ["data-src", "data-original", "data-lazy", "data-zoom", "data-zoom-image", "data-url"]
SRCSET_ATTRS = ["srcset", "data-srcset"]
STYLE_URL_RE = re.compile(r"url\(['\"]?(?P<url>[^)'\"]+)['\"]?\)", re.IGNORECASE)


def sha1_name(url_or_bytes: bytes | str, ext_hint: str = "") -> str:
    h = hashlib.sha1(
        url_or_bytes if isinstance(url_or_bytes, bytes) else url_or_bytes.encode("utf-8")
    ).hexdigest()
    return f"{h}{ext_hint}"


def pick_src_from_srcset(srcset_val: str) -> str:
    # choose the highest density/width candidate
    best = None
    for candidate in srcset_val.split(","):
        url_part, *desc = candidate.strip().split()
        score = 0
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
        best = (score, url_part) if (best is None or score > best[0]) else best
    return best[1] if best else srcset_val.split(",")[0].strip()


def guess_ext_from_mime(m: str) -> str:
    return mimetypes.guess_extension(m) or (".png" if m.startswith("image/") else "")


def download_or_decode_image(src: str, base_url: str | None, out_dir: Path, timeout=30) -> str:
    """
    Returns a local file path (str) to the downloaded/decoded image.
    Handles: absolute/relative URLs, data: URIs.
    """
    # data: URI
    if src.startswith("data:"):
        # data:image/png;base64,<payload>
        header, b64 = src.split(",", 1)
        mime = header.split(";")[0].split(":", 1)[1] if ":" in header else "image/png"
        ext = guess_ext_from_mime(mime) or ".bin"
        payload = base64.b64decode(b64)
        fn = sha1_name(payload, ext)
        dst = out_dir / fn
        if not dst.exists():
            dst.write_bytes(payload)
        return str(dst)

    # URL (absolute or relative)
    url = urljoin(base_url, src) if base_url else src
    # Some HTMLs may carry scheme-less //host/path
    if url.startswith("//"):
        url = "https:" + url

    # fetch
    fn = sha1_name(url, Path(urlparse(url).path).suffix or ".png")
    dst = out_dir / fn
    if not dst.exists():
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return str(dst)


def find_base_url(soup: BeautifulSoup, html_path: Path) -> str | None:
    # 1) explicit <base href="">
    base_tag = soup.find("base", href=True)
    if base_tag:
        return base_tag["href"]

    # 2) canonical or og:url on many publisher pages
    link_canon = soup.find("link", rel=lambda v: v and "canonical" in v.lower(), href=True)
    if link_canon:
        return link_canon["href"]
    meta_og = soup.find(
        "meta", property=lambda v: v and v.lower() in ("og:url", "prism.url"), content=True
    )
    if meta_og:
        return meta_og["content"]

    # 3) fallback to file location (no trailing filename)
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

    # <img> with lazy-loading attributes → set .src
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            for la in LAZY_ATTRS:
                if img.get(la):
                    img["src"] = img[la]
                    break
        # srcset preference
        for sa in SRCSET_ATTRS:
            if img.get(sa):
                img["src"] = pick_src_from_srcset(img[sa])
                break

    # <picture>/<source> → materialize an <img> with best candidate
    for picture in soup.find_all("picture"):
        # If it already contains an <img src="">, skip.
        img = picture.find("img")
        src_candidate = None
        for srcset_attr in SRCSET_ATTRS:
            for src_node in picture.find_all("source", attrs={srcset_attr: True}):
                src_candidate = pick_src_from_srcset(src_node[srcset_attr])
                break
            if src_candidate:
                break
        if (img is None) and src_candidate:
            new_img = soup.new_tag("img")
            new_img["src"] = src_candidate
            picture.append(new_img)

    # CSS background-image → append a sibling <img> so HTML backend can see it
    for url in extract_bg_images(soup):
        # Create a hidden collector at end of body for background images
        collector = soup.find("div", id="__bg_images__")
        if collector is None and soup.body:
            collector = soup.new_tag("div", id="__bg_images__")
            collector["style"] = "display:none"
            soup.body.append(collector)
        if collector is not None:
            ni = soup.new_tag("img")
            ni["src"] = url
            collector.append(ni)

    # OpenGraph figures (e.g., journal sites) → ensure they exist as <img>
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

    # Download/replace every src (including data:) with local file path
    for img in soup.find_all("img"):
        if not img.get("src"):
            continue
        local_path = download_or_decode_image(img["src"], base_url, images_dir)
        img["src"] = Path(local_path).absolute().as_uri()

    # Write normalized file next to original
    norm_path = html_path.with_suffix(".normalized.html")
    norm_path.write_text(str(soup), encoding="utf-8")
    return norm_path


# ---------- Worker ----------


def convert_one(
    html_path: Path,
    out_dir: Path,
    caption_model: str,
    use_classes: bool,
    api_url: str,
    api_model: str,
    api_headers: dict,
    timeout: int,
) -> tuple[str, str]:
    try:
        norm_html = normalize_html_images(html_path, IMAGES_DIR)

        # Configure ConvertPipelineOptions (simple pipeline is used for HTML backends)
        # - do_picture_description: enable VLM captioning
        # - do_picture_classification: enable document figure classifier
        # - picture_description_options: choose local Granite, SmolVLM, or a remote API (e.g., vLLM)
        # Doc: Enrichment features & picture models; ConvertPipelineOptions fields.  (citations in main text)
        pl_opts = ConvertPipelineOptions()
        pl_opts.do_picture_description = True
        if caption_model == "granite":
            pl_opts.picture_description_options = granite_picture_description
        elif caption_model == "smolvlm":
            pl_opts.picture_description_options = smolvlm_picture_description
        else:
            pl_opts.enable_remote_services = True
            headers = api_headers or {}
            pl_opts.picture_description_options = PictureDescriptionApiOptions(
                url=api_url,
                params={"model": api_model} if api_model else {},
                headers=headers,
                timeout=timeout,
                prompt="Describe the image in two concise sentences.",
            )
        if use_classes:
            pl_opts.do_picture_classification = True

        converter = DocumentConverter(
            format_options={InputFormat.HTML: HTMLFormatOption(pipeline_options=pl_opts)}
        )

        result = converter.convert(norm_html, raises_on_error=False)
        if result.document is None:
            return (html_path.name, "fail: empty-document")

        out_path = out_dir / (html_path.stem + ".doctags")
        result.document.save_as_doctags(out_path)
        return (html_path.name, "ok")
    except Exception as e:
        return (html_path.name, f"fail: {e}")


# ---------- Main ----------


def main():
    htmls = sorted([p for p in INPUT_DIR.rglob("*.html") if p.is_file()])
    if not htmls:
        print(f"No HTML files found under {INPUT_DIR}")
        return

    # Parse remote headers JSON once
    try:
        api_headers = json.loads(args.api_headers.strip() or "{}")
    except Exception:
        api_headers = {}

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [
            ex.submit(
                convert_one,
                p,
                OUTPUT_DIR,
                args.caption_model,
                args.classify,
                args.api_url,
                args.api_model,
                api_headers,
                args.timeout,
            )
            for p in htmls
        ]
        for fut in as_completed(futures):
            name, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {name}: {status}")

    print(f"Done. ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()
