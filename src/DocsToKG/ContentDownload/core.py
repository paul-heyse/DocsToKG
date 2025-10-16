"""
Core primitives for the DocsToKG content download pipeline.

This module consolidates the shared taxonomy enums, payload classification
heuristics, and identifier normalisation helpers that were previously spread
across ``classifications``, ``classifier``, and ``utils``. Co-locating these
utilities keeps the public surface that other modules consume in one place,
simplifying imports for both the CLI and resolver pipeline.
"""

from __future__ import annotations

import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

__all__ = (
    "Classification",
    "PDF_LIKE",
    "ReasonCode",
    "DEFAULT_SNIFF_BYTES",
    "DEFAULT_MIN_PDF_BYTES",
    "DEFAULT_TAIL_CHECK_BYTES",
    "WorkArtifact",
    "atomic_write",
    "atomic_write_bytes",
    "atomic_write_text",
    "classify_payload",
    "_extract_filename_from_disposition",
    "_infer_suffix",
    "update_tail_buffer",
    "has_pdf_eof",
    "tail_contains_html",
    "normalize_doi",
    "normalize_pmcid",
    "strip_prefix",
    "dedupe",
    "normalize_pmid",
    "normalize_arxiv",
    "slugify",
    "normalize_url",
    "parse_size",
)


# ---------------------------------------------------------------------------
# Shared heuristics


DEFAULT_SNIFF_BYTES = 64 * 1024
DEFAULT_MIN_PDF_BYTES = 1024
DEFAULT_TAIL_CHECK_BYTES = 2048

_SIZE_SUFFIXES = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
}


def atomic_write(
    path: Path,
    chunks: Iterable[bytes],
    *,
    hasher: Optional[Any] = None,
    temp_suffix: str = ".part",
) -> int:
    """Atomically write ``chunks`` to ``path`` and return the byte count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = temp_suffix if temp_suffix.startswith(".") else f".{temp_suffix}"
    temp_name = f"{path.name}{suffix}.{uuid.uuid4().hex}"
    temp_path = path.with_name(temp_name)
    written = 0
    replaced = False
    try:
        with temp_path.open("wb") as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if hasher is not None:
                    hasher.update(chunk)
        os.replace(temp_path, path)
        replaced = True
        return written
    finally:
        if not replaced:
            with suppress(FileNotFoundError):
                temp_path.unlink()


def atomic_write_bytes(
    path: Path,
    chunks: Iterable[bytes],
    *,
    hasher: Optional[Any] = None,
) -> int:
    """Backward-compatible wrapper for :func:`atomic_write`."""

    return atomic_write(path, chunks, hasher=hasher)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically write ``text`` to ``path`` using :func:`atomic_write`."""

    atomic_write(path, [text.encode(encoding)])


@dataclass
class WorkArtifact:
    """Normalized artifact describing an OpenAlex work to process."""

    work_id: str
    title: str
    publication_year: Optional[int]
    doi: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    arxiv_id: Optional[str]
    landing_urls: List[str]
    pdf_urls: List[str]
    open_access_url: Optional[str]
    source_display_names: List[str]
    base_stem: str
    pdf_dir: Path
    html_dir: Path
    failed_pdf_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.namespaces: Dict[str, Path] = {"pdf": self.pdf_dir, "html": self.html_dir}


# ---------------------------------------------------------------------------
# Classification taxonomy


class Classification(Enum):
    """Canonical classification codes for download outcomes."""

    PDF = "pdf"
    HTML = "html"
    MISS = "miss"
    UNKNOWN = "unknown"
    HTTP_ERROR = "http_error"
    CACHED = "cached"
    SKIPPED = "skipped"
    HTML_TOO_LARGE = "html_too_large"
    PAYLOAD_TOO_LARGE = "payload_too_large"

    @classmethod
    def from_wire(cls, value: Union[str, "Classification", None]) -> "Classification":
        """Return the enum member when ``value`` matches a known code."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        if not text:
            return cls.UNKNOWN

        legacy_map = {
            "pdf_unknown": cls.PDF,
            "pdf_corrupt": cls.MISS,
            "request_error": cls.HTTP_ERROR,
            "exists": cls.CACHED,
        }
        if text in legacy_map:
            return legacy_map[text]

        for member in cls:
            if member.value == text:
                return member
        return cls.UNKNOWN


PDF_LIKE = frozenset({Classification.PDF, Classification.CACHED})


class ReasonCode(Enum):
    """Machine-readable reason taxonomy for download outcomes."""

    UNKNOWN = "unknown"
    PDF_SNIFF_UNKNOWN = "pdf_sniff_unknown"
    PDF_TOO_SMALL = "pdf_too_small"
    HTML_TAIL_DETECTED = "html_tail_detected"
    PDF_EOF_MISSING = "pdf_eof_missing"
    ROBOTS_DISALLOWED = "robots_disallowed"
    UNEXPECTED_304 = "unexpected_304"
    RESUME_COMPLETE = "resume_complete"
    ALREADY_DOWNLOADED = "already_downloaded"
    CONDITIONAL_CACHE_INVALID = "conditional_cache_invalid"
    CONDITIONAL_NOT_MODIFIED = "conditional_not_modified"
    MAX_BYTES_HEADER = "max_bytes_header"
    MAX_BYTES_STREAM = "max_bytes_stream"
    HEAD_PRECHECK_FAILED = "head_precheck_failed"
    HTTP_STATUS = "http_status"
    REQUEST_EXCEPTION = "request_exception"
    RESOLVER_MISSING = "resolver_missing"
    RESOLVER_DISABLED = "resolver_disabled"
    RESOLVER_NOT_APPLICABLE = "resolver_not_applicable"
    DUPLICATE_URL = "duplicate_url"
    DUPLICATE_URL_GLOBAL = "duplicate_url_global"
    LIST_ONLY = "list_only"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    RESOLVER_BREAKER_OPEN = "resolver_breaker_open"
    DOMAIN_BREAKER_OPEN = "domain_breaker_open"
    DOMAIN_BYTES_BUDGET = "domain_bytes_budget"
    DOMAIN_MAX_BYTES = "domain_max_bytes"
    DOMAIN_DISALLOWED_MIME = "domain_disallowed_mime"
    BUDGET_EXHAUSTED = "budget_exhausted"

    @classmethod
    def from_wire(cls, value: Union[str, "ReasonCode", None]) -> "ReasonCode":
        """Return the matching enum member or ``UNKNOWN``."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        if not text:
            return cls.UNKNOWN
        for member in cls:
            if member.value == text:
                return member
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# Payload classification helpers


def classify_payload(head_bytes: bytes, content_type: Optional[str], url: str) -> Classification:
    """Classify a payload as ``Classification.PDF``/``Classification.HTML`` or ``Classification.UNKNOWN``."""

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


def parse_size(value: str) -> int:
    """Parse human-friendly size strings like ``10GB`` into byte counts."""

    text = (value or "").strip().lower().replace(",", "").replace("_", "")
    if not text:
        raise ValueError("size value cannot be empty")
    match = re.fullmatch(r"(?P<amount>\d+(?:\.\d+)?)(?P<suffix>[kmgt]?b)?", text)
    if not match:
        raise ValueError(f"invalid size specification: {value!r}")
    amount_text = match.group("amount")
    suffix = match.group("suffix") or "b"
    try:
        amount = float(amount_text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid numeric amount in {value!r}") from exc
    factor = _SIZE_SUFFIXES.get(suffix)
    if factor is None:
        raise ValueError(f"unsupported size suffix {suffix!r}")
    bytes_value = int(amount * factor)
    if bytes_value <= 0:
        raise ValueError("size value must be positive")
    return bytes_value


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


# ---------------------------------------------------------------------------
# Identifier / string normalisation helpers


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI identifiers by stripping common prefixes and whitespace."""

    if not doi:
        return None
    value = doi.strip()
    lower = value.lower()
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            value = value[len(prefix) :]
            lower = value.lower()
            break
    if lower.startswith("doi:"):
        value = value[len("doi:") :]
    return value.strip() or None


def normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
    """Normalize PMCID values ensuring a canonical PMC prefix."""

    if not pmcid:
        return None
    pmcid = pmcid.strip()
    match = re.search(r"(?:PMC)?(\d+)", pmcid, flags=re.I)
    if match:
        return f"PMC{match.group(1)}"
    return None


def strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    """Strip a case-insensitive prefix from a string when present."""

    if not value:
        return None
    value = value.strip()
    if value.lower().startswith(prefix.lower()):
        return value[len(prefix) :]
    return value


def dedupe(items: List[str]) -> List[str]:
    """Remove duplicates while preserving the first occurrence order."""

    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def normalize_pmid(pmid: Optional[str]) -> Optional[str]:
    """Extract the numeric PubMed identifier from the supplied string."""

    if not pmid:
        return None
    pmid = pmid.strip()
    match = re.search(r"(\d+)", pmid)
    return match.group(1) if match else None


def normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    """Normalize arXiv identifiers by removing common prefixes and whitespace."""

    if not arxiv_id:
        return None
    value = strip_prefix(arxiv_id, "arxiv:") or arxiv_id
    value = value.replace("https://arxiv.org/abs/", "")
    return value.strip() or None


def slugify(text: str, keep: int = 80) -> str:
    """Create a filesystem-friendly slug from the provided text."""

    text = re.sub(r"[^\w\s]+", "", text or "")
    text = re.sub(r"\s+", "_", text.strip())
    return text[:keep] or "untitled"


def normalize_url(url: str) -> str:
    """Return a canonicalised version of ``url`` suitable for deduplication."""

    parts = urlsplit(url)
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            urlencode(query_pairs, doseq=True),
            "",
        )
    )
