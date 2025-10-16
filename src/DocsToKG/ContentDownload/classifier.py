"""Payload classification helpers shared across the content download toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

from DocsToKG.ContentDownload.classifications import Classification


def classify_payload(head_bytes: bytes, content_type: Optional[str], url: str) -> Classification:
    """Classify a payload as ``Classification.PDF``/``Classification.HTML`` or ``UNKNOWN``."""

    ctype = (content_type or "").lower()
    stripped = head_bytes.lstrip() if head_bytes else b""
    prefix = stripped[:64].lower()

    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        return Classification.HTML
    if prefix.startswith(b"<head") or prefix.startswith(b"<body"):
        return Classification.HTML

    if stripped.startswith(b"%PDF") or (head_bytes or b"")[:2048].find(b"%PDF") != -1:
        return Classification.PDF

    if "html" in ctype:
        return Classification.HTML
    if "pdf" in ctype:
        return Classification.PDF
    if ctype == "application/octet-stream":
        return Classification.UNKNOWN

    if url.lower().endswith(".pdf"):
        return Classification.PDF

    return Classification.UNKNOWN


def _extract_filename_from_disposition(disposition: Optional[str]) -> Optional[str]:
    """Return the filename component from a Content-Disposition header."""

    if not disposition:
        return None
    parts = [segment.strip() for segment in disposition.split(";") if segment.strip()]
    for part in parts:
        lower = part.lower()
        if lower.startswith("filename*="):
            try:
                value = part.split("=", 1)[1].strip()
            except IndexError:
                continue
            _, _, encoded = value.partition("''")
            candidate = unquote(encoded or value)
            candidate = candidate.strip('"')
            if candidate:
                return candidate
        if lower.startswith("filename="):
            try:
                candidate = part.split("=", 1)[1].strip()
            except IndexError:
                continue
            candidate = candidate.strip('"')
            if candidate:
                return candidate
    return None


def _infer_suffix(
    url: str,
    content_type: Optional[str],
    disposition: Optional[str],
    classification: Classification | str,
    default_suffix: str,
) -> str:
    """Infer a destination suffix from HTTP hints and classification heuristics."""

    classification_code = Classification.from_wire(classification)
    classification_text = (
        classification_code.value if classification_code is not None else str(classification)
    )

    filename = _extract_filename_from_disposition(disposition)
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix.lower()

    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype == "application/pdf" or (
        ctype.endswith("pdf") and classification_text.startswith("pdf")
    ):
        return ".pdf"
    if ctype in {"text/html", "application/xhtml+xml"} or "html" in ctype:
        return ".html"

    if classification_text.startswith("pdf"):
        path_suffix = Path(urlsplit(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        return default_suffix

    if classification_code is Classification.HTML:
        path_suffix = Path(urlsplit(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        return default_suffix

    return default_suffix


def update_tail_buffer(buffer: bytearray, chunk: bytes, *, limit: int = 1024) -> None:
    """Maintain a sliding window of the trailing ``limit`` bytes."""

    if not chunk:
        return
    buffer.extend(chunk)
    if len(buffer) > limit:
        del buffer[:-limit]


def has_pdf_eof(path: Path, *, window_bytes: int = 2048) -> bool:
    """Return ``True`` when the PDF at ``path`` ends with ``%%EOF`` marker."""

    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            window = max(int(window_bytes), 0)
            offset = max(size - window, 0)
            handle.seek(offset)
            tail = handle.read().decode(errors="ignore")
            return "%%EOF" in tail
    except OSError:
        return False


def tail_contains_html(tail: Optional[bytes]) -> bool:
    """Heuristic to detect HTML signatures in the trailing payload bytes."""

    if not tail:
        return False
    lowered = tail.lower()
    return any(marker in lowered for marker in (b"</html", b"</body", b"</script", b"<html"))


__all__ = (
    "classify_payload",
    "_extract_filename_from_disposition",
    "_infer_suffix",
    "update_tail_buffer",
    "has_pdf_eof",
    "tail_contains_html",
)
