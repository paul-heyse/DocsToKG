"""Networking utilities for DocsToKG ontology downloads."""

from __future__ import annotations

import logging
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set, TypeVar

import pooch
import psutil
import requests
from urllib.parse import urlparse

from . import ratelimit
from .config import DownloadConfiguration
from .errors import DownloadFailure, OntologyDownloadError, PolicyError
from .io_safe import sanitize_filename, sha256_file, validate_url_security

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from .pipeline import validate_manifest_dict

T = TypeVar("T")

TokenBucket = ratelimit.TokenBucket


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retryable: Callable[[BaseException], bool],
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.5,
    callback: Optional[Callable[[int, BaseException, float], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute ``func`` with exponential backoff until it succeeds."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except BaseException as exc:  # pragma: no cover - behaviour verified via callers
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            if callback is not None:
                try:
                    callback(attempt, exc, delay)
                except Exception:  # pragma: no cover - defensive against callbacks
                    pass
            sleep(max(delay, 0.0))


def log_memory_usage(
    logger: logging.Logger,
    *,
    stage: str,
    event: str,
    validator: Optional[str] = None,
) -> None:
    """Emit debug-level memory usage snapshots when enabled."""
    is_enabled = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled):
        enabled = is_enabled(logging.DEBUG)
    else:  # pragma: no cover - fallback for stub loggers
        enabled = False
    if not enabled:
        return
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    extra: Dict[str, object] = {"stage": stage, "event": event, "memory_mb": round(memory_mb, 2)}
    if validator:
        extra["validator"] = validator
    logger.debug("memory usage", extra=extra)


@dataclass(slots=True)
class DownloadResult:
    """Result metadata for a completed download operation.

    Attributes:
        path: Final file path where the ontology document was stored.
        status: Download status (`fresh`, `updated`, or `cached`).
        sha256: SHA-256 checksum of the downloaded artifact.
        etag: HTTP ETag returned by the upstream server, when available.
        last_modified: Upstream last-modified header value if provided.
        content_type: Reported MIME type when available (HEAD or GET).
        content_length: Reported content length when available.

    Examples:
        >>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None, None, None)
        >>> result.status
        'fresh'
    """

    path: Path
    status: str
    sha256: str
    etag: Optional[str]
    last_modified: Optional[str]
    content_type: Optional[str]
    content_length: Optional[int]


_RETRYABLE_HTTP_STATUSES = {403, 408, 425, 429, 500, 502, 503, 504}


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        seconds = float(candidate)
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(candidate)
        except (TypeError, ValueError, IndexError):
            return None
        if retry_time is None:
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = (retry_time - now).total_seconds()
    return max(seconds, 0.0)


def _is_retryable_status(status_code: Optional[int]) -> bool:
    if status_code is None:
        return True
    if status_code >= 500:
        return True
    return status_code in _RETRYABLE_HTTP_STATUSES


_RDF_FORMAT_LABELS = {
    "application/rdf+xml": "RDF/XML",
    "text/turtle": "Turtle",
    "application/n-triples": "N-Triples",
    "application/trig": "TriG",
    "application/ld+json": "JSON-LD",
}
_RDF_ALIAS_GROUPS = {
    "application/rdf+xml": {"application/rdf+xml", "application/xml", "text/xml"},
    "text/turtle": {"text/turtle", "application/x-turtle"},
    "application/n-triples": {"application/n-triples", "text/plain"},
    "application/trig": {"application/trig"},
    "application/ld+json": {"application/ld+json"},
}
RDF_MIME_ALIASES: Set[str] = set()
RDF_MIME_FORMAT_LABELS: Dict[str, str] = {}
for canonical, aliases in _RDF_ALIAS_GROUPS.items():
    label = _RDF_FORMAT_LABELS[canonical]
    for alias in aliases:
        RDF_MIME_ALIASES.add(alias)
        RDF_MIME_FORMAT_LABELS[alias] = label


class StreamingDownloader(pooch.HTTPDownloader):
    """Custom downloader supporting HEAD validation, conditional requests, resume, and caching.

    The downloader shares a :mod:`requests` session so it can issue a HEAD probe
    prior to streaming content, verifies Content-Type and Content-Length against
    expectations, and persists ETag/Last-Modified headers for cache-friendly
    revalidation.

    Attributes:
        destination: Final location where the ontology will be stored.
        custom_headers: HTTP headers supplied by the resolver.
        http_config: Download configuration governing retries and limits.
        previous_manifest: Manifest from prior runs used for caching.
        logger: Logger used for structured telemetry.
        status: Final download status (`fresh`, `updated`, or `cached`).
        response_etag: ETag returned by the upstream server, if present.
        response_last_modified: Last-modified timestamp provided by the server.
        expected_media_type: MIME type provided by the resolver for validation.

    Examples:
        >>> from pathlib import Path
        >>> from DocsToKG.OntologyDownload import DownloadConfiguration
        >>> downloader = StreamingDownloader(
        ...     destination=Path("/tmp/ontology.owl"),
        ...     headers={},
        ...     http_config=DownloadConfiguration(),
        ...     previous_manifest={},
        ...     logger=logging.getLogger("test"),
        ... )
        >>> downloader.status
        'fresh'
    """

    def __init__(
        self,
        *,
        destination: Path,
        headers: Dict[str, str],
        http_config: DownloadConfiguration,
        previous_manifest: Optional[Dict[str, object]],
        logger: logging.Logger,
        expected_media_type: Optional[str] = None,
        service: Optional[str] = None,
        origin_host: Optional[str] = None,
    ) -> None:
        super().__init__(headers={}, progressbar=False, timeout=http_config.timeout_sec)
        self.destination = destination
        self.custom_headers = headers
        self.http_config = http_config
        self.previous_manifest = previous_manifest or {}
        self.logger = logger
        self.status = "fresh"
        self.response_etag: Optional[str] = None
        self.response_last_modified: Optional[str] = None
        self.expected_media_type = expected_media_type
        self.head_content_type: Optional[str] = None
        self.head_content_length: Optional[int] = None
        self.response_content_type: Optional[str] = None
        self.response_content_length: Optional[int] = None
        self.service = service
        self.origin_host = origin_host

    def _preliminary_head_check(
        self, url: str, session: requests.Session
    ) -> tuple[Optional[str], Optional[int]]:
        """Probe the origin with HEAD to audit media type and size before downloading.

        The HEAD probe allows the pipeline to abort before streaming large
        payloads that exceed configured limits and to log early warnings for
        mismatched Content-Type headers reported by the origin.

        Args:
            url: Fully qualified download URL resolved by the planner.
            session: Prepared requests session used for outbound calls.

        Returns:
            Tuple ``(content_type, content_length)`` extracted from response
            headers. Each element is ``None`` when the origin omits it.

        Raises:
            PolicyError: If the origin reports a payload larger than the
                configured ``max_download_size_gb`` limit.
        """

        try:
            response = session.head(
                url,
                headers=self.custom_headers,
                timeout=self.http_config.timeout_sec,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            self.logger.debug(
                "HEAD request exception, proceeding with GET",
                extra={"stage": "download", "error": str(exc), "url": url},
            )
            return None, None

        if response.status_code >= 400:
            self.logger.debug(
                "HEAD request failed, proceeding with GET",
                extra={
                    "stage": "download",
                    "method": "HEAD",
                    "status_code": response.status_code,
                    "url": url,
                },
            )
            return None, None

        content_type = response.headers.get("Content-Type")
        content_length_header = response.headers.get("Content-Length")
        content_length = int(content_length_header) if content_length_header else None

        if content_length:
            max_bytes = self.http_config.max_download_size_gb * (1024**3)
            if content_length > max_bytes:
                self.logger.error(
                    "file exceeds size limit (HEAD check)",
                    extra={
                        "stage": "download",
                        "content_length": content_length,
                        "limit_bytes": max_bytes,
                        "url": url,
                    },
                )
                raise PolicyError(
                    "File size {size} bytes exceeds limit of {limit} GB (detected via HEAD)".format(
                        size=content_length,
                        limit=self.http_config.max_download_size_gb,
                    )
                )

        return content_type, content_length

    def _validate_media_type(
        self,
        actual_content_type: Optional[str],
        expected_media_type: Optional[str],
        url: str,
    ) -> None:
        """Validate that the received ``Content-Type`` header is acceptable, tolerating aliases.

        RDF endpoints often return generic XML or Turtle aliases, so the
        validator accepts a small set of known MIME variants while still
        surfacing actionable warnings for unexpected types.

        Args:
            actual_content_type: Raw header value reported by the origin server.
            expected_media_type: MIME type declared by resolver metadata.
            url: Download URL logged when mismatches occur.

        Returns:
            None
        """

        if not self.http_config.validate_media_type:
            return
        if not expected_media_type:
            return
        if not actual_content_type:
            self.logger.warning(
                "server did not provide Content-Type header",
                extra={
                    "stage": "download",
                    "expected_media_type": expected_media_type,
                    "url": url,
                },
            )
            return

        actual_mime = actual_content_type.split(";")[0].strip().lower()
        expected_mime = expected_media_type.strip().lower()
        if actual_mime == expected_mime:
            return

        expected_label = RDF_MIME_FORMAT_LABELS.get(expected_mime)
        actual_label = RDF_MIME_FORMAT_LABELS.get(actual_mime)
        if expected_label and actual_label:
            if expected_label == actual_label:
                if actual_mime != expected_mime:
                    self.logger.info(
                        "acceptable media type variation",
                        extra={
                            "stage": "download",
                            "expected": expected_mime,
                            "actual": actual_mime,
                            "label": expected_label,
                            "url": url,
                        },
                    )
                return
            variation_hint = {
                "stage": "download",
                "expected_media_type": expected_mime,
                "expected_label": expected_label,
                "actual_media_type": actual_mime,
                "actual_label": actual_label,
                "url": url,
            }
            self.logger.warning(
                "media type mismatch detected",
                extra={
                    **variation_hint,
                    "action": "proceeding with download",
                    "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
                },
            )
            return

        self.logger.warning(
            "media type mismatch detected",
            extra={
                "stage": "download",
                "expected_media_type": expected_mime,
                "actual_media_type": actual_mime,
                "url": url,
                "action": "proceeding with download",
                "override_hint": "Set defaults.http.validate_media_type: false to disable validation",
            },
        )

    def __call__(self, url: str, output_file: str, pooch_logger: logging.Logger) -> None:  # type: ignore[override]
        """Stream ontology content to disk while enforcing download policies.

        Args:
            url: Secure download URL resolved by the planner.
            output_file: Temporary filename managed by pooch during download.
            pooch_logger: Logger instance supplied by pooch (unused).

        Raises:
            PolicyError: If download limits are exceeded.
            OntologyDownloadError: If filesystem errors occur.
            requests.HTTPError: Propagated when HTTP status codes indicate failure.

        Returns:
            None
        """
        manifest_headers: Dict[str, str] = {}
        if "etag" in self.previous_manifest:
            manifest_headers["If-None-Match"] = self.previous_manifest["etag"]
        if "last_modified" in self.previous_manifest:
            manifest_headers["If-Modified-Since"] = self.previous_manifest["last_modified"]
        request_headers = {**self.custom_headers, **manifest_headers}
        part_path = Path(output_file + ".part")
        destination_part_path = Path(str(self.destination) + ".part")
        if not part_path.exists() and destination_part_path.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination_part_path, part_path)
        resume_position = part_path.stat().st_size if part_path.exists() else 0
        if resume_position:
            request_headers["Range"] = f"bytes={resume_position}-"
        attempt = 0
        session = requests.Session()
        self.head_content_type = None
        self.head_content_length = None
        self.response_content_type = None
        self.response_content_length = None
        head_content_type, head_content_length = self._preliminary_head_check(url, session)
        self.head_content_type = head_content_type
        self.head_content_length = head_content_length
        if head_content_type:
            self._validate_media_type(head_content_type, self.expected_media_type, url)
        try:
            while True:
                attempt += 1
                try:
                    with session.get(
                        url,
                        headers=request_headers,
                        stream=True,
                        timeout=self.http_config.download_timeout_sec,
                        allow_redirects=True,
                    ) as response:
                        if response.status_code == 304 and Path(self.destination).exists():
                            self.status = "cached"
                            self.response_etag = response.headers.get(
                                "ETag"
                            ) or self.previous_manifest.get("etag")
                            self.response_last_modified = response.headers.get(
                                "Last-Modified"
                            ) or self.previous_manifest.get("last_modified")
                            manifest_type = self.previous_manifest.get("content_type")
                            self.response_content_type = (
                                manifest_type if isinstance(manifest_type, str) else None
                            )
                            manifest_length = self.previous_manifest.get("content_length")
                            try:
                                self.response_content_length = (
                                    int(manifest_length) if manifest_length is not None else None
                                )
                            except (TypeError, ValueError):
                                self.response_content_length = None
                            part_path.unlink(missing_ok=True)
                            return
                        if response.status_code in {429, 503}:
                            retry_after_header = response.headers.get("Retry-After")
                            retry_after_delay = _parse_retry_after(retry_after_header)
                            if retry_after_delay is not None:
                                self.logger.warning(
                                    "download retry-after",
                                    extra={
                                        "stage": "download",
                                        "status_code": response.status_code,
                                        "retry_after_sec": round(retry_after_delay, 2),
                                        "service": self.service,
                                        "host": self.origin_host,
                                    },
                                )
                                ratelimit.apply_retry_after(
                                    http_config=self.http_config,
                                    service=self.service,
                                    host=self.origin_host,
                                    delay=retry_after_delay,
                                )
                                attempt = max(attempt - 1, 0)
                                time.sleep(retry_after_delay)
                                continue
                        if response.status_code == 206:
                            self.status = "updated"
                        response.raise_for_status()
                        self._validate_media_type(
                            response.headers.get("Content-Type"),
                            self.expected_media_type,
                            url,
                        )
                        self.response_content_type = response.headers.get("Content-Type")
                        length_header = response.headers.get("Content-Length")
                        total_bytes: Optional[int] = None
                        next_progress: Optional[float] = 0.1
                        parsed_length: Optional[int] = None
                        if length_header:
                            try:
                                parsed_length = int(length_header)
                            except ValueError:
                                parsed_length = None
                        self.response_content_length = parsed_length
                        if parsed_length is not None:
                            total_bytes = parsed_length
                        max_bytes = self.http_config.max_download_size_gb * (1024**3)
                        if total_bytes is not None and total_bytes > max_bytes:
                            self.logger.error(
                                "file exceeds size limit",
                                extra={
                                    "stage": "download",
                                    "size": total_bytes,
                                    "limit": max_bytes,
                                },
                            )
                            raise PolicyError(
                                f"File size {total_bytes} exceeds configured limit of "
                                f"{self.http_config.max_download_size_gb} GB"
                            )
                        if total_bytes:
                            completed_fraction = resume_position / total_bytes
                            if completed_fraction >= 1:
                                next_progress = None
                            else:
                                next_progress = ((int(completed_fraction * 10)) + 1) / 10
                        self.response_etag = response.headers.get("ETag")
                        self.response_last_modified = response.headers.get("Last-Modified")
                        mode = "ab" if resume_position else "wb"
                        bytes_downloaded = resume_position
                        part_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with part_path.open(mode) as fh:
                                for chunk in response.iter_content(chunk_size=1 << 20):
                                    if not chunk:
                                        continue
                                    fh.write(chunk)
                                    bytes_downloaded += len(chunk)
                                    if total_bytes and next_progress:
                                        progress = bytes_downloaded / total_bytes
                                        while next_progress and progress >= next_progress:
                                            self.logger.info(
                                                "download progress",
                                                extra={
                                                    "stage": "download",
                                                    "status": "in-progress",
                                                    "progress": {
                                                        "percent": round(
                                                            min(progress, 1.0) * 100, 1
                                                        )
                                                    },
                                                },
                                            )
                                            next_progress += 0.1
                                            if next_progress > 1:
                                                next_progress = None
                                                break
                                    if bytes_downloaded > self.http_config.max_download_size_gb * (
                                        1024**3
                                    ):
                                        self.logger.error(
                                            "download exceeded size limit",
                                            extra={
                                                "stage": "download",
                                                "size": bytes_downloaded,
                                                "limit": self.http_config.max_download_size_gb
                                                * (1024**3),
                                            },
                                        )
                                        raise PolicyError(
                                            "Download exceeded maximum configured size while streaming"
                                        )
                        except OSError as exc:
                            part_path.unlink(missing_ok=True)
                            self.logger.error(
                                "filesystem error during download",
                                extra={"stage": "download", "error": str(exc)},
                            )
                            if "No space left" in str(exc):
                                raise OntologyDownloadError(
                                    "No space left on device while writing download"
                                ) from exc
                            raise OntologyDownloadError(f"Failed to write download: {exc}") from exc
                        break
                except (
                    requests.ConnectionError,
                    requests.Timeout,
                    requests.HTTPError,
                    requests.exceptions.SSLError,
                ) as exc:
                    if attempt > self.http_config.max_retries:
                        raise
                    sleep_time = self.http_config.backoff_factor * (2 ** (attempt - 1))
                    self.logger.warning(
                        "download retry",
                        extra={
                            "stage": "download",
                            "attempt": attempt,
                            "sleep_sec": sleep_time,
                            "error": str(exc),
                        },
                    )
                    time.sleep(sleep_time)
            part_path.replace(Path(output_file))
            destination_part_path.unlink(missing_ok=True)
        finally:
            session.close()


def download_stream(
    *,
    url: str,
    destination: Path,
    headers: Dict[str, str],
    previous_manifest: Optional[Dict[str, object]],
    http_config: DownloadConfiguration,
    cache_dir: Path,
    logger: logging.Logger,
    expected_media_type: Optional[str] = None,
    service: Optional[str] = None,
) -> DownloadResult:
    """Download ontology content with HEAD validation, rate limiting, caching, retries, and hash checks.

    Args:
        url: URL of the ontology document to download.
        destination: Target file path for the downloaded content.
        headers: HTTP headers forwarded to the download request.
        previous_manifest: Manifest metadata from a prior run, used for caching.
        http_config: Download configuration containing timeouts, limits, and rate controls.
        cache_dir: Directory where intermediary cached files are stored.
        logger: Logger adapter for structured download telemetry.
        expected_media_type: Expected Content-Type for validation, if known.
        service: Logical service identifier for per-service rate limiting.

    Returns:
        DownloadResult describing the final artifact and metadata.

    Raises:
        PolicyError: If policy validation fails or limits are exceeded.
        OntologyDownloadError: If retryable download mechanisms exhaust or IO fails.
    """
    secure_url = validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    host = parsed.hostname
    bucket = ratelimit.get_bucket(http_config=http_config, host=host, service=service)
    bucket.consume()

    start_time = time.monotonic()
    log_memory_usage(logger, stage="download", event="before")
    downloader = StreamingDownloader(
        destination=destination,
        headers=headers,
        http_config=http_config,
        previous_manifest=previous_manifest,
        logger=logger,
        expected_media_type=expected_media_type,
        service=service,
        origin_host=host,
    )

    def _resolved_content_metadata() -> tuple[Optional[str], Optional[int]]:
        content_type = downloader.response_content_type or downloader.head_content_type
        if content_type is None and previous_manifest:
            manifest_type = previous_manifest.get("content_type")
            if isinstance(manifest_type, str):
                content_type = manifest_type

        content_length = downloader.response_content_length
        if content_length is None and downloader.head_content_length is not None:
            content_length = downloader.head_content_length
        if content_length is None and previous_manifest:
            manifest_length = previous_manifest.get("content_length")
            try:
                content_length = int(manifest_length) if manifest_length is not None else None
            except (TypeError, ValueError):
                content_length = None
        return content_type, content_length

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(destination.name)
    try:
        cached_path = Path(
            pooch.retrieve(
                secure_url,
                path=cache_dir,
                fname=safe_name,
                known_hash=None,
                downloader=downloader,
                progressbar=False,
            )
        )
    except requests.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        message = f"HTTP error while downloading {secure_url}: {exc}"
        retryable = _is_retryable_status(status_code)
        logger.error(
            "download request failed",
            extra={
                "stage": "download",
                "url": secure_url,
                "error": str(exc),
                "status_code": status_code,
            },
        )
        raise DownloadFailure(message, status_code=status_code, retryable=retryable) from exc
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.SSLError,
    ) as exc:
        logger.error(
            "download request failed",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise DownloadFailure(
            f"HTTP error while downloading {secure_url}: {exc}", retryable=True
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive catch for pooch errors
        logger.error(
            "pooch download error",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise OntologyDownloadError(f"Download failed for {secure_url}: {exc}") from exc
    if downloader.status == "cached":
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            "cache hit",
            extra={"stage": "download", "status": "cached", "elapsed_ms": round(elapsed, 2)},
        )
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, destination)
        sha256 = sha256_file(destination)
        log_memory_usage(logger, stage="download", event="after")
        content_type, content_length = _resolved_content_metadata()
        return DownloadResult(
            path=destination,
            status="cached",
            sha256=sha256,
            etag=downloader.response_etag,
            last_modified=downloader.response_last_modified,
            content_type=content_type,
            content_length=content_length,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)
    sha256 = sha256_file(destination)
    expected_hash = previous_manifest.get("sha256") if previous_manifest else None
    if expected_hash and expected_hash != sha256:
        logger.error(
            "sha256 mismatch detected",
            extra={
                "stage": "download",
                "expected": expected_hash,
                "actual": sha256,
                "url": secure_url,
            },
        )
        destination.unlink(missing_ok=True)
        cached_path.unlink(missing_ok=True)
        return download_stream(
            url=url,
            destination=destination,
            headers=headers,
            previous_manifest=None,
            http_config=http_config,
            cache_dir=cache_dir,
            logger=logger,
            expected_media_type=expected_media_type,
            service=service,
        )
    elapsed = (time.monotonic() - start_time) * 1000
    logger.info(
        "download complete",
        extra={
            "stage": "download",
            "status": downloader.status,
            "elapsed_ms": round(elapsed, 2),
            "sha256": sha256,
        },
    )
    log_memory_usage(logger, stage="download", event="after")
    content_type, content_length = _resolved_content_metadata()
    return DownloadResult(
        path=destination,
        status=downloader.status,
        sha256=sha256,
        etag=downloader.response_etag,
        last_modified=downloader.response_last_modified,
        content_type=content_type,
        content_length=content_length,
    )


__all__ = [
    "retry_with_backoff",
    "DownloadResult",
    "DownloadFailure",
    "TokenBucket",
    "RDF_MIME_ALIASES",
    "RDF_MIME_FORMAT_LABELS",
    "StreamingDownloader",
    "download_stream",
    "log_memory_usage",
    "validate_manifest_dict",
]


def __getattr__(name: str):
    """Lazily proxy pipeline helpers without incurring import cycles."""

    if name == "validate_manifest_dict":
        from .pipeline import validate_manifest_dict as _validate_manifest_dict

        return _validate_manifest_dict
    raise AttributeError(name)
