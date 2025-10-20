"""Core primitives and shared utilities for DocsToKG content downloads.

Responsibilities
----------------
- Define canonical enums, dataclasses, and constants (e.g., ``Classification``,
  ``ReasonCode``, ``DEFAULT_*`` thresholds) that the rest of the stack depends on.
- Provide payload inspection helpers (header sniffing, tail analysis, EOF checks)
  used to classify resolver responses and enforce size/format safeguards.
- Offer persistent-friendly helpers such as :func:`slugify`,
  :func:`atomic_write`, and identifier normalisation routines for DOI/PMCID/etc.
- Centralise URL/metadata normalisation so telemetry, manifest generation, and
  resumable caches operate over consistent keys.

Design Notes
------------
- The abstractions here are intentionally side-effect free; file IO utilities
  expose explicit arguments for atomic writes so higher layers can control
  storage paths.
- Classification heuristics operate on simple buffers/strings, making them easy
  to exercise in isolation from the downloader.
"""

from __future__ import annotations

import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Union
from urllib.parse import unquote, urlsplit

from DocsToKG.ContentDownload.urls import canonical_for_index

__all__ = (
    "Classification",
    "PDF_LIKE",
    "ReasonCode",
    "DEFAULT_SNIFF_BYTES",
    "DEFAULT_MIN_PDF_BYTES",
    "DEFAULT_TAIL_CHECK_BYTES",
    "WorkArtifact",
    "DownloadContext",
    "atomic_write",
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
    "normalize_classification",
    "normalize_reason",
    "parse_size",
)

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.download import RobotsCache


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
    keep_partial_on_error: bool = False,
) -> int:
    """Atomically write ``chunks`` to ``path`` and return the byte count.

    Performance Note:
        When ``hasher`` is provided, uses an optimized code path that avoids
        conditional checks in the hot loop for better throughput on large files.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = temp_suffix if temp_suffix.startswith(".") else f".{temp_suffix}"
    temp_name = f"{path.name}{suffix}.{uuid.uuid4().hex}"
    temp_path = path.with_name(temp_name)
    written = 0
    replaced = False
    partial_kept = False
    try:
        with temp_path.open("wb") as handle:
            # Optimization: Use separate code paths to avoid hasher None check in hot loop
            if hasher is not None:
                for chunk in chunks:
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    hasher.update(chunk)
            else:
                for chunk in chunks:
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
        os.replace(temp_path, path)
        replaced = True
        return written
    except Exception:
        if keep_partial_on_error:
            partial_path = path.with_suffix(path.suffix + suffix)
            try:
                partial_path.parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            with suppress(FileNotFoundError):
                partial_path.unlink()
            try:
                os.replace(temp_path, partial_path)
                partial_kept = True
            except FileNotFoundError:
                partial_kept = False
            except OSError:
                partial_kept = False
                with suppress(FileNotFoundError):
                    temp_path.unlink()
        else:
            with suppress(FileNotFoundError):
                temp_path.unlink()
        raise
    finally:
        if not replaced and not partial_kept:
            with suppress(FileNotFoundError):
                temp_path.unlink()


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
    xml_dir: Path
    failed_pdf_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.namespaces: Dict[str, Path] = {
            "pdf": self.pdf_dir,
            "html": self.html_dir,
            "xml": self.xml_dir,
        }


@dataclass
class DownloadContext:
    """Typed execution context shared by the CLI and resolver pipeline."""

    resolver_order: List[str] = field(default_factory=list)
    dry_run: bool = False
    list_only: bool = False
    extract_html_text: bool = False
    previous: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_manifest_index: Any = field(default_factory=dict)
    sniff_bytes: int = DEFAULT_SNIFF_BYTES
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    domain_content_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    host_accept_overrides: Dict[str, Any] = field(default_factory=dict)
    progress_callback: Optional[Callable[[int, Optional[int], str], None]] = None
    enable_range_resume: bool = False
    robots_checker: Optional["RobotsCache"] = None
    skip_head_precheck: bool = False
    head_precheck_passed: bool = False
    content_addressed: bool = False
    skip_large_downloads: bool = False
    size_warning_threshold: Optional[int] = None
    chunk_size: Optional[int] = None
    stream_retry_attempts: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    provided_fields: Set[str] = field(default_factory=set, init=False, repr=False)
    verify_cache_digest: bool = False

    def __post_init__(self) -> None:
        self.resolver_order = self._normalize_sequence(self.resolver_order)
        self.previous = self._normalize_mapping(self.previous)
        if not hasattr(self.global_manifest_index, "get"):
            self.global_manifest_index = self._normalize_mapping(self.global_manifest_index)
        self.domain_content_rules = self._normalize_mapping(self.domain_content_rules)
        self.host_accept_overrides = self._normalize_mapping(self.host_accept_overrides)
        self.extra = self._normalize_mapping(self.extra)

        self.dry_run = bool(self.dry_run)
        self.list_only = bool(self.list_only)
        self.extract_html_text = bool(self.extract_html_text)
        self.enable_range_resume = bool(self.enable_range_resume)
        self.skip_head_precheck = bool(self.skip_head_precheck)
        self.head_precheck_passed = bool(self.head_precheck_passed)
        self.content_addressed = bool(self.content_addressed)
        self.skip_large_downloads = bool(self.skip_large_downloads)
        self.verify_cache_digest = bool(self.verify_cache_digest)

        self.size_warning_threshold = self._coerce_optional_positive(self.size_warning_threshold)
        self.chunk_size = self._coerce_optional_positive(self.chunk_size)
        self.stream_retry_attempts = max(int(self.stream_retry_attempts or 0), 0)

        self.sniff_bytes = self._coerce_non_negative(self.sniff_bytes, DEFAULT_SNIFF_BYTES)
        self.min_pdf_bytes = self._coerce_non_negative(self.min_pdf_bytes, DEFAULT_MIN_PDF_BYTES)
        self.tail_check_bytes = self._coerce_non_negative(
            self.tail_check_bytes, DEFAULT_TAIL_CHECK_BYTES
        )
        self.provided_fields = {str(field) for field in self.provided_fields if field}

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]] = None) -> "DownloadContext":
        """Construct a context instance from a mapping-based payload."""

        if isinstance(data, cls):
            return data

        mapping = dict(data or {})
        provided: Set[str] = set()

        def _pop(name: str, default: Any = None, *, canonical: Optional[str] = None) -> Any:
            key = canonical or name
            if name in mapping:
                provided.add(key)
            return mapping.pop(name, default)

        resolver_order_value = _pop("resolver_order", None, canonical="resolver_order")
        override_value = _pop("resolver_order_override", None, canonical="resolver_order")
        if resolver_order_value is None:
            resolver_order_value = override_value or []
        stream_attempts = _pop(
            "_stream_retry_attempts",
            _pop("stream_retry_attempts", 0, canonical="stream_retry_attempts"),
            canonical="stream_retry_attempts",
        )
        context = cls(
            resolver_order=resolver_order_value,
            dry_run=_pop("dry_run", False),
            list_only=_pop("list_only", False),
            extract_html_text=_pop("extract_html_text", False),
            previous=_pop("previous", {}),
            global_manifest_index=_pop("global_manifest_index", {}),
            sniff_bytes=_pop("sniff_bytes", DEFAULT_SNIFF_BYTES),
            min_pdf_bytes=_pop("min_pdf_bytes", DEFAULT_MIN_PDF_BYTES),
            tail_check_bytes=_pop("tail_check_bytes", DEFAULT_TAIL_CHECK_BYTES),
            domain_content_rules=_pop("domain_content_rules", {}),
            host_accept_overrides=_pop("host_accept_overrides", {}),
            progress_callback=_pop("progress_callback", None),
            enable_range_resume=_pop("enable_range_resume", False),
            robots_checker=_pop("robots_checker", None),
            skip_head_precheck=_pop("skip_head_precheck", False),
            head_precheck_passed=_pop("head_precheck_passed", False),
            content_addressed=_pop("content_addressed", False),
            skip_large_downloads=_pop("skip_large_downloads", False),
            verify_cache_digest=_pop("verify_cache_digest", False),
            size_warning_threshold=_pop("size_warning_threshold", None),
            chunk_size=_pop("chunk_size", None),
            stream_retry_attempts=stream_attempts,
        )
        context.provided_fields.update(provided)
        if mapping:
            context.extra.update(mapping)
        return context

    def mark_explicit(self, *fields: str) -> None:
        """Record that the given fields were explicitly provided by the caller."""

        self.provided_fields.update(field for field in fields if field)

    def is_explicit(self, field: str) -> bool:
        """Return ``True`` when ``field`` was explicitly provided by the caller."""

        return field in self.provided_fields

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the context to a mapping for legacy integrations."""

        payload: Dict[str, Any] = {
            "resolver_order": list(self.resolver_order),
            "dry_run": self.dry_run,
            "list_only": self.list_only,
            "extract_html_text": self.extract_html_text,
            "previous": self.previous,
            "sniff_bytes": self.sniff_bytes,
            "min_pdf_bytes": self.min_pdf_bytes,
            "tail_check_bytes": self.tail_check_bytes,
            "domain_content_rules": self.domain_content_rules,
            "host_accept_overrides": self.host_accept_overrides,
            "progress_callback": self.progress_callback,
            "enable_range_resume": self.enable_range_resume,
            "robots_checker": self.robots_checker,
            "skip_head_precheck": self.skip_head_precheck,
            "head_precheck_passed": self.head_precheck_passed,
            "content_addressed": self.content_addressed,
            "skip_large_downloads": self.skip_large_downloads,
            "size_warning_threshold": self.size_warning_threshold,
            "chunk_size": self.chunk_size,
            "verify_cache_digest": self.verify_cache_digest,
        }
        if self.stream_retry_attempts:
            payload["stream_retry_attempts"] = self.stream_retry_attempts
        if self.extra:
            payload.update(self.extra)
        return payload

    def clone_for_download(self) -> "DownloadContext":
        """Return a shallow clone suitable for per-download mutation."""

        clone = DownloadContext.from_mapping(self.to_dict())
        clone.stream_retry_attempts = 0
        clone.provided_fields = set(self.provided_fields)
        clone.global_manifest_index = self.global_manifest_index
        return clone

    @staticmethod
    def _normalize_sequence(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        else:
            try:
                items = list(value)
            except TypeError:
                items = [value]
        return [str(item) for item in items if item]

    @staticmethod
    def _normalize_mapping(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _coerce_optional_positive(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _coerce_non_negative(value: Any, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return number if number >= 0 else max(default, 0)


# ---------------------------------------------------------------------------
# Classification taxonomy


class Classification(Enum):
    """Canonical classification codes for download outcomes."""

    PDF = "pdf"
    HTML = "html"
    XML = "xml"
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
    RATE_LIMITED = "rate_limited"
    RESOLVER_BREAKER_OPEN = "resolver_breaker_open"
    DOMAIN_BREAKER_OPEN = "domain_breaker_open"
    DOMAIN_DISALLOWED_MIME = "domain_disallowed_mime"
    SKIP_LARGE_DOWNLOAD = "skip_large_download"
    WORKER_EXCEPTION = "worker_exception"

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


def normalize_classification(value: Union[str, Classification, None]) -> Union[Classification, str]:
    """Return a normalized classification token preserving unknown custom codes."""

    if isinstance(value, Classification):
        return value
    if value is None:
        return Classification.UNKNOWN
    text = str(value).strip()
    if not text:
        return Classification.UNKNOWN
    candidate = Classification.from_wire(text)
    if isinstance(value, str):
        lowered = text.lower()
        if candidate is Classification.UNKNOWN and lowered not in {
            member.value for member in Classification
        }:
            return value
    return candidate


def normalize_reason(value: Optional[Union[str, ReasonCode]]) -> Optional[Union[ReasonCode, str]]:
    """Return a normalized reason token preserving unknown custom codes."""

    if isinstance(value, ReasonCode):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("-", "_")
    candidate = ReasonCode.from_wire(normalized)
    if candidate is not ReasonCode.UNKNOWN or normalized == ReasonCode.UNKNOWN.value:
        return candidate
    return text


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

    if prefix.startswith(b"<?xml") or prefix.startswith(b"<rss") or prefix.startswith(b"<feed"):
        return Classification.XML

    if stripped.startswith(b"%PDF") or (head_bytes or b"")[:2048].find(b"%PDF") != -1:
        return Classification.PDF

    if "html" in ctype:
        return Classification.HTML
    if "xml" in ctype:
        return Classification.XML
    if "pdf" in ctype:
        return Classification.PDF
    if ctype == "application/octet-stream":
        return Classification.UNKNOWN

    lowered_url = url.lower()
    if lowered_url.endswith(".pdf"):
        return Classification.PDF
    if lowered_url.endswith(".xml"):
        return Classification.XML

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
    if ctype in {"application/xml", "text/xml"} or "xml" in ctype:
        return ".xml"

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
    if classification_code is Classification.XML:
        path_suffix = Path(urlsplit(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        return ".xml"

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
    """Deprecated alias for :func:`DocsToKG.ContentDownload.urls.canonical_for_index`."""

    return canonical_for_index(url)
