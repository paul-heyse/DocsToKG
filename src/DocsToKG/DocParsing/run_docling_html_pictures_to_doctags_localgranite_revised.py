#!/usr/bin/env python3
"""
HTML → DocTags with robust image handling + captioning & classification (Docling)
and Hugging Face authentication (env or token file).

- Input :  Data/HTML/        (recurses; excludes *.normalized.html)
- Output:  Data/DocTagsFiles/*.doctags
- Images:  Data/ImagesCache/ (downloaded/decoded assets)

Captioning:
  --caption-model granite|smolvlm|api
  --hf-cache /home/paul/hf-cache   --offline
  --api-url/--api-model/--api-headers for remote OpenAI-style endpoints

Classification:
  --classify

Accelerator:
  --device auto|cpu|cuda|mps  --threads N
"""

import os, re, json, base64, hashlib, mimetypes, argparse, warnings
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# -------------------- CLI & environment --------------------


def detect_data_root(start: Path) -> Path:
    for anc in (start, *start.parents):
        if (anc / "Data" / "HTML").is_dir():
            return anc / "Data"
    return start / "Data"


ENV_DATA_ROOT = os.getenv("DOCSTOKG_DATA_ROOT")
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(ENV_DATA_ROOT) if ENV_DATA_ROOT else detect_data_root(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=DATA_ROOT / "HTML")
parser.add_argument("--output", type=Path, default=DATA_ROOT / "DocTagsFiles")
parser.add_argument("--images-dir", type=Path, default=DATA_ROOT / "ImagesCache")
parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))

parser.add_argument("--classify", action="store_true")
parser.add_argument("--caption-model", choices=["granite", "smolvlm", "api"], default="granite")
parser.add_argument(
    "--caption-prompt", type=str, default="Describe the image in two concise, factual sentences."
)
parser.add_argument("--min-pic-area", type=float, default=0.0)

# HF cache + auth
parser.add_argument("--hf-cache", type=str, default=str(DATA_ROOT.parent / "hf-cache"))
parser.add_argument(
    "--hf-token-file",
    type=Path,
    default=Path("/home/paul/hf-cache/token"),
    help="Plain-text HF token fallback path (used if env var not set).",
)
parser.add_argument("--offline", action="store_true")

# Remote API (vLLM/Ollama/OpenAI-style)
parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions")
parser.add_argument("--api-model", type=str, default="")
parser.add_argument("--api-headers", type=str, default="{}")
parser.add_argument("--timeout", type=int, default=90)

# Accelerator
parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))

parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

INPUT_DIR: Path = args.input
OUTPUT_DIR: Path = args.output
IMAGES_DIR: Path = args.images_dir
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HF cache location (must be set before saving token)
if args.hf_cache:
    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_cache)


def ensure_hf_auth(token_file: Path):
    """
    Load HF token from env or token file and:
      - set common env vars (HUGGING_FACE_HUB_TOKEN, HF_TOKEN, HUGGINGFACEHUB_API_TOKEN)
      - persist via HfFolder.save_token() under HF_HOME
    """
    tok = (
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not tok and token_file and token_file.exists():
        try:
            tok = token_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            warnings.warn(f"Could not read token file {token_file}: {e}")

    if tok:
        # Set all common env keys so every library path sees it
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = tok
        try:
            from huggingface_hub import HfFolder

            HfFolder.save_token(tok)  # writes to HF_HOME/token
        except Exception as e:
            warnings.warn(f"Could not persist HF token via HfFolder: {e}")


# Load token before any worker processes are spawned
ensure_hf_auth(args.hf_token_file)

# Offline mode (after token so we can still seed when needed by toggling this off)
if args.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Keep CPU libs polite
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# -------------------- HTTP session with retries --------------------


def make_http_session():
    s = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=0.3,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = make_http_session()

# -------------------- URL helpers & normalization --------------------

LAZY_ATTRS = ["data-src", "data-original", "data-lazy", "data-zoom", "data-zoom-image", "data-url"]
SRCSET_ATTRS = ["srcset", "data-srcset"]
STYLE_URL_RE = re.compile(r"url\(['\"]?(?P<url>[^)'\"]+)['\"]?\)", re.IGNORECASE)

AD_HOST_PATTERNS = (
    "doubleclick.net",
    "googlesyndication.com",
    "googletagmanager.com",
    "gampad",
    "adservice",
    "adsystem",
    "scorecardresearch.com",
)


def is_ad_or_tracker(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in AD_HOST_PATTERNS)


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


def find_base_url(soup: BeautifulSoup, html_path: Path) -> str | None:
    # Prefer declared base/canonical/og:url
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
    # Fallback to filesystem folder
    if html_path.exists():
        return f"file://{html_path.parent.as_posix()}/"
    return None


def extract_bg_images(soup: BeautifulSoup) -> list[str]:
    urls = []
    for tag in soup.find_all(style=True):
        for m in STYLE_URL_RE.finditer(tag["style"]):
            urls.append(m.group("url"))
    return urls


def promote_og_images(soup: BeautifulSoup) -> list[str]:
    urls = []
    for p in ("og:image", "og:image:secure_url"):
        for m in soup.find_all("meta", attrs={"property": p}):
            if m.get("content"):
                urls.append(m["content"])
    return urls


def list_htmls(root: Path) -> list[Path]:
    out = []
    for pat in ("*.html", "*.htm", "*.xhtml"):
        out.extend(root.rglob(pat))
    # exclude our normalized files
    return sorted([p for p in out if p.is_file() and not p.name.endswith(".normalized.html")])


# -------------------- fetch/resolve images --------------------


def resolve_candidate_url(src: str, base_url: str | None) -> str | None:
    s = src.strip()
    if not s:
        return None

    # data: URI or http(s)
    if s.startswith("data:") or s.startswith("http://") or s.startswith("https://"):
        return s

    # scheme-relative
    if s.startswith("//"):
        return "https:" + s

    # file:// absolute
    if s.startswith("file://"):
        # If someone embedded domain as file://example.com/... → treat as https://example.com/...
        if re.match(r"^file://[a-zA-Z0-9\.\-]+/", s):
            return "https://" + s[len("file://") :]
        return s

    # root-relative ("/static/…")
    if s.startswith("/"):
        if base_url and base_url.startswith(("http://", "https://")):
            pr = urlparse(base_url)
            origin = f"{pr.scheme}://{pr.netloc}"
            return origin + s
        # Unknown origin (e.g., base is file://) → skip
        return None

    # relative path
    if base_url:
        return urljoin(base_url, s)

    # No base; we can't resolve
    return None


def download_or_decode_image(
    src: str, base_url: str | None, out_dir: Path, timeout=30
) -> str | None:
    """
    Returns absolute filesystem path (string) to the local copy, or None if skipped.
    """
    # data: URI
    if src.startswith("data:"):
        try:
            header, b64 = src.split(",", 1)
            mime = header.split(";")[0].split(":", 1)[1] if ":" in header else "image/png"
            ext = guess_ext(mime) or ".bin"
            payload = base64.b64decode(b64)
            fn = sha1_name(payload, ext)
            dst = out_dir / fn
            if not dst.exists():
                dst.write_bytes(payload)
            return str(dst.absolute())
        except Exception as e:
            warnings.warn(f"data: image decode failed: {e}")
            return None

    url = resolve_candidate_url(src, base_url)
    if not url:
        return None

    # local filesystem path (from earlier runs or local references)
    if url.startswith("file://"):
        p = Path(urlparse(url).path)
        if p.exists():
            # copy into cache (dedupe by sha1 of path string)
            ext = p.suffix or ".png"
            fn = sha1_name(str(p), ext)
            dst = out_dir / fn
            if not dst.exists():
                try:
                    dst.write_bytes(p.read_bytes())
                except Exception as e:
                    warnings.warn(f"file:// read failed {p}: {e}")
                    return None
            return str(dst.absolute())
        return None

    # http(s)
    if is_ad_or_tracker(url):
        return None
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code >= 400:
            return None
        # guess extension from Content-Type if available
        ct = r.headers.get("Content-Type", "").split(";")[0].strip()
        ext = guess_ext(ct) or Path(urlparse(url).path).suffix or ".png"
        fn = sha1_name(url, ext)
        dst = out_dir / fn
        if not dst.exists():
            dst.write_bytes(r.content)
        return str(dst.absolute())
    except Exception as e:
        warnings.warn(f"HTTP image fetch failed {url}: {e}")
        return None


# -------------------- normalization pass --------------------


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
    collector = None
    for url in extract_bg_images(soup):
        if collector is None and soup.body:
            collector = soup.new_tag("div", id="__bg_images__", style="display:none")
            soup.body.append(collector)
        if collector is not None:
            ni = soup.new_tag("img")
            ni["src"] = url
            collector.append(ni)

    # OpenGraph figures → hidden <img>
    og_urls = promote_og_images(soup)
    if og_urls and soup.body:
        og_div = soup.find(id="__og_images__")
        if og_div is None:
            og_div = soup.new_tag("div", id="__og_images__", style="display:none")
            soup.body.append(og_div)
        for u in og_urls:
            ni = soup.new_tag("img")
            ni["src"] = u
            og_div.append(ni)

    # Download/replace every src with local filesystem path (NO file:// scheme)
    for img in soup.find_all("img"):
        s = img.get("src")
        if not s:
            continue
        local = download_or_decode_image(s, base_url, images_dir)
        if local:
            img["src"] = str(Path(local).absolute())

    norm_path = html_path.with_suffix(".normalized.html")
    norm_path.write_text(str(soup), encoding="utf-8")
    return norm_path


# -------------------- VLM / pipeline options --------------------


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


def make_vlm_options_for_granite(hf_cache: str, prompt: str, area: float):
    # Keep repo_id as Hub ID and point to your cache; seed once online, then you can use --offline.
    return PictureDescriptionVlmOptions(
        repo_id="ibm-granite/granite-vision-3.3-2b",
        repo_cache_folder=hf_cache,
        prompt=prompt,
        picture_area_threshold=area,
    )


def make_vlm_options_for_smol(prompt: str, area: float):
    return PictureDescriptionVlmOptions(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        prompt=prompt,
        picture_area_threshold=area,
    )


def make_api_options(url: str, model: str, headers: dict, prompt: str, area: float, timeout: int):
    return PictureDescriptionApiOptions(
        url=url,
        params={"model": model} if model else {},
        headers=headers or {},
        timeout=timeout,
        prompt=prompt,
        picture_area_threshold=area,
    )


# -------------------- Convert worker --------------------

HTML_EXTS = {".html", ".htm", ".xhtml"}


def convert_one(
    html_path: Path,
    out_dir: Path,
    caption_model: str,
    classify: bool,
    hf_cache: str,
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
        norm_html = normalize_html_images(html_path, IMAGES_DIR)

        pl_opts = ConvertPipelineOptions()
        pl_opts.accelerator_options = make_accel(device, threads)
        pl_opts.do_picture_description = True
        pl_opts.do_picture_classification = bool(classify)

        if caption_model == "granite":
            pl_opts.picture_description_options = make_vlm_options_for_granite(
                hf_cache=hf_cache, prompt=caption_prompt, area=min_pic_area
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
            )

        converter = DocumentConverter(
            format_options={
                InputFormat.HTML: HTMLFormatOption(
                    backend=HTMLDocumentBackend, pipeline_options=pl_opts
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


# -------------------- Main --------------------


def main():
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images: {IMAGES_DIR}")
    print(f"Workers: {args.workers}")
    print(
        f"Caption model: {args.caption_model}  | classify={args.classify}  | device={args.device}"
    )
    print(f"HF cache: {args.hf_cache} | Offline: {bool(os.environ.get('HF_HUB_OFFLINE'))}")

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
                args.hf_cache,
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
