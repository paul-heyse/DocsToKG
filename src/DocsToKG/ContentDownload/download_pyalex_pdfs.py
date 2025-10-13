#!/usr/bin/env python3
import argparse
import os
import re
import time
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import requests

import pyalex
from pyalex import Works, Topics, config as oa_config


# ---------- helpers ----------
def slugify(text: str, keep: int = 80) -> str:
    text = re.sub(r"[^\w\s\-]+", "", text or "")
    text = re.sub(r"\s+", "_", text.strip())
    return text[:keep] or "untitled"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pdf_candidates(work: Dict) -> List[str]:
    urls = []
    for key in ("best_oa_location", "primary_location"):
        loc = work.get(key) or {}
        url = loc.get("pdf_url")
        if url:
            urls.append(url)
    # full location sweep (can add more pdf_url hits)
    for loc in work.get("locations", []) or []:
        url = (loc or {}).get("pdf_url")
        if url:
            urls.append(url)
    # Fallback: an OA landing page might actually be a direct PDF
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        urls.append(oa["oa_url"])
    # de-dup preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


MAX_SNIFF_BYTES = 64 * 1024  # sufficient to decide PDF vs HTML


def classify_payload(head_bytes: bytes, content_type: str, url: str) -> Optional[str]:
    """
    Return 'pdf', 'html', or None if undecided based on current evidence.
    """
    ctype = (content_type or "").lower()
    stripped = head_bytes.lstrip() if head_bytes else b""
    prefix = stripped[:64].lower()

    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        return "html"
    if prefix.startswith(b"<head") or prefix.startswith(b"<body"):
        return "html"

    if stripped.startswith(b"%PDF") or b"%PDF" in head_bytes[:2048]:
        return "pdf"

    if "html" in ctype:
        return "html"
    if "pdf" in ctype:
        return "pdf"

    if url.lower().endswith(".pdf"):
        return "pdf"

    return None


def download_pdf(
    session: requests.Session,
    url: str,
    base_stem: str,
    pdf_dir: Path,
    html_dir: Path,
    timeout: float = 30.0,
) -> Tuple[str, Optional[Path]]:
    try:
        # Try HEAD to detect obvious non-PDFs; not all hosts support HEAD, so tolerate failures.
        ctype_hint = ""
        try:
            h = session.head(url, allow_redirects=True, timeout=timeout)
            ctype = h.headers.get("Content-Type", "")
            ctype_hint = ctype
            if h.status_code == 200 and ("pdf" in ctype.lower() or url.lower().endswith(".pdf")):
                pass  # looks like a PDF
            # If HEAD is unhelpful, we’ll still try GET below.
        except requests.RequestException:
            pass

        with session.get(url, stream=True, allow_redirects=True, timeout=timeout) as r:
            if r.status_code != 200:
                return "http_error", None

            content_type = r.headers.get("Content-Type", "") or ctype_hint
            sniff_buffer = bytearray()
            detected: Optional[str] = None
            flagged_unknown = False
            dest_path: Optional[Path] = None
            handle = None

            try:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if not chunk:
                        continue
                    if detected is None:
                        sniff_buffer.extend(chunk)
                        detected = classify_payload(bytes(sniff_buffer), content_type, url)
                        if detected is None and len(sniff_buffer) >= MAX_SNIFF_BYTES:
                            detected = "pdf"
                            flagged_unknown = True

                        if detected is not None:
                            if detected == "html":
                                dest_path = html_dir / f"{base_stem}.html"
                                ensure_dir(dest_path.parent)
                            else:
                                dest_path = pdf_dir / f"{base_stem}.pdf"
                                ensure_dir(dest_path.parent)
                            handle = open(dest_path, "wb")
                            handle.write(sniff_buffer)
                            sniff_buffer.clear()
                            continue
                    else:
                        assert handle is not None
                        handle.write(chunk)

                if detected is None:
                    # No content received at all.
                    return "empty", None

                assert handle is not None and dest_path is not None
            finally:
                if handle is not None:
                    handle.close()

            if flagged_unknown:
                return "pdf_unknown", dest_path
            return detected, dest_path
    except requests.RequestException:
        return "request_error", None


def build_query(args) -> Works:
    q = Works()
    # Text search or strict Topic ID filter
    if args.topic_id:
        q = q.filter(topics={"id": args.topic_id})
    else:
        # Fulltext/title/abstract search across works
        q = q.search(args.topic)
    # Year range (OpenAlex supports "YYYY-YYYY" on publication_year)
    q = q.filter(publication_year=f"{args.year_start}-{args.year_end}")
    # Optional: open-access only
    if args.oa_only:
        q = q.filter(open_access={"is_oa": True})
    # Select top-level fields that carry PDF links
    q = q.select(
        [
            "id",
            "title",
            "publication_year",
            "open_access",
            "best_oa_location",
            "primary_location",
            "locations",
        ]
    )
    # Sort newest to oldest so recent stuff lands first (optional)
    q = q.sort(publication_date="desc")
    return q


def resolve_topic_id_if_needed(topic_text: Optional[str]) -> Optional[str]:
    if not topic_text:
        return None
    # Resolve the best matching OpenAlex Topic for stricter filtering
    hits = Topics().search(topic_text).get()
    if not hits:
        return None
    # Choose the first hit (you can add smarter disambiguation if needed)
    return hits[0]["id"]


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Download OpenAlex PDFs for a topic and year range using pyalex."
    )

    parser.add_argument(
        "--topic", type=str, help="Free-text topic (search across title/abstract/fulltext)."
    )
    parser.add_argument(
        "--topic-id",
        type=str,
        help="OpenAlex Topic ID (e.g., https://openalex.org/T12345). Overrides --topic.",
    )
    parser.add_argument("--year-start", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--year-end", type=int, required=True, help="End year (inclusive).")
    parser.add_argument("--out", type=Path, default=Path("./pdfs"), help="Output folder.")
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Folder for HTML responses (default: sibling 'HTML').",
    )
    parser.add_argument(
        "--mailto", type=str, default=None, help="Your email for the OpenAlex polite pool."
    )
    parser.add_argument("--per-page", type=int, default=200, help="Results per page (1-200).")
    parser.add_argument(
        "--max", type=int, default=None, help="Max works to process (default: all)."
    )
    parser.add_argument("--oa-only", action="store_true", help="Only consider open-access works.")
    parser.add_argument(
        "--sleep", type=float, default=0.05, help="Sleep seconds between downloads to be polite."
    )
    args = parser.parse_args()

    if not args.topic and not args.topic_id:
        parser.error("Provide --topic or --topic-id.")

    # Join OpenAlex polite pool for better reliability
    if args.mailto:
        oa_config.email = args.mailto

    # If user provided text topic but wants strict Topic semantics, resolve an ID (optional pattern)
    topic_id = args.topic_id
    if not topic_id and args.topic:
        # Comment out the next 3 lines if you prefer plain text search only
        topic_id = None  # leave as None by default; flip to resolve by uncommenting:
        # topic_id = resolve_topic_id_if_needed(args.topic)

    query = build_query(
        argparse.Namespace(
            topic=args.topic,
            topic_id=topic_id,
            year_start=args.year_start,
            year_end=args.year_end,
            oa_only=args.oa_only,
        )
    )

    pdf_dir = args.out
    html_dir = args.html_out or (pdf_dir.parent / "HTML")

    ensure_dir(pdf_dir)
    ensure_dir(html_dir)

    # Cursor pagination; per_page up to 200; n_max=None => all results
    pager = query.paginate(per_page=args.per_page, n_max=None)

    s = requests.Session()
    s.headers["User-Agent"] = (
        f"pyalex-pdf-downloader (mailto:{args.mailto})" if args.mailto else "pyalex-pdf-downloader"
    )

    processed = 0
    saved = 0
    for page in pager:
        for w in page:
            processed += 1
            if args.max and processed > args.max:
                print(f"Reached --max={args.max}. Stopping.")
                return

            year = w.get("publication_year", "")
            title = w.get("title") or w.get("display_name") or ""
            work_id = (w.get("id") or "W").rsplit("/", 1)[-1]
            base = f"{year}__{slugify(title)}__{work_id}".strip("_")
            pdf_path = pdf_dir / f"{base}.pdf"
            if pdf_path.exists():
                print(f"[skip] {pdf_path.name} (exists)")
                continue

            ok = False
            pdf_warn = False
            html_saved: Optional[Path] = None
            pdf_file_path: Optional[Path] = None
            for url in pdf_candidates(w):
                status, artifact_path = download_pdf(s, url, base, pdf_dir, html_dir)
                if status == "pdf":
                    ok = True
                    pdf_file_path = artifact_path
                    break
                if status == "pdf_unknown":
                    ok = True
                    pdf_warn = True
                    pdf_file_path = artifact_path
                    break
                if status == "html" and artifact_path is not None:
                    if html_saved is None:
                        html_saved = artifact_path
                    # keep searching other URLs for a true PDF
                    continue
                # otherwise try next candidate (http errors, empty bodies, etc.)
            if ok:
                saved += 1
                display_name = pdf_file_path.name if pdf_file_path is not None else pdf_path.name
                if pdf_warn:
                    print(f"[ok?] {display_name} (saved but could not confirm PDF signature)")
                else:
                    print(f"[ok]   {display_name}")
            else:
                if html_saved is not None:
                    try:
                        rel_html = html_saved.relative_to(html_dir)
                    except ValueError:
                        rel_html = html_saved
                    print(f"[miss] No PDF for {work_id} — saved HTML to {rel_html}")
                else:
                    print(f"[miss] No PDF found for {work_id} — {title[:80]}")
            time.sleep(args.sleep)

    print(f"\nDone. Processed {processed} works, saved {saved} PDFs into {args.out}")


if __name__ == "__main__":
    main()
