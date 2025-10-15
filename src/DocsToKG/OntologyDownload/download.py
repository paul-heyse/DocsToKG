"""
Ontology Download Utilities

This module houses secure download helpers that implement rate limiting,
content validation, and resumable transfers for ontology documents. It works
in concert with resolver planning to ensure that downloaded artifacts respect
size limits and are safe for downstream document processing.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import shutil
import socket
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import pooch
import psutil
import requests

from .config import ConfigError, DownloadConfiguration


def _log_memory(logger: logging.Logger, event: str) -> None:
    is_enabled = getattr(logger, "isEnabledFor", None)
    if callable(is_enabled):
        enabled = is_enabled(logging.DEBUG)
    else:  # pragma: no cover - fallback for stub loggers
        enabled = False
    if not enabled:
        return
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    logger.debug(
        "memory usage",
        extra={"stage": "download", "event": event, "memory_mb": round(memory_mb, 2)},
    )


@dataclass(slots=True)
class DownloadResult:
    """Result metadata for a completed download operation.

    Attributes:
        path: Final file path where the ontology document was stored.
        status: Download status (`fresh`, `updated`, or `cached`).
        sha256: SHA-256 checksum of the downloaded artifact.
        etag: HTTP ETag returned by the upstream server, when available.
        last_modified: Upstream last-modified header value if provided.
    """

    path: Path
    status: str
    sha256: str
    etag: Optional[str]
    last_modified: Optional[str]


class TokenBucket:
    """Simple token bucket implementation for per-host rate limiting."""

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity or rate_per_sec
        self.tokens = self.capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> None:
        """Consume tokens from the bucket, sleeping until capacity is available.

        Args:
            tokens: Number of tokens required for the current download request.

        Returns:
            None
        """
        while True:
            with self.lock:
                now = time.monotonic()
                delta = now - self.timestamp
                self.timestamp = now
                self.tokens = min(self.capacity, self.tokens + delta * self.rate)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
            time.sleep(max(needed / self.rate, 0.0))


_TOKEN_BUCKETS: Dict[str, TokenBucket] = {}


def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to prevent directory traversal and unsafe characters.

    Args:
        filename: Candidate filename provided by an upstream service.

    Returns:
        Safe filename compatible with local filesystem storage.
    """

    original = filename
    safe = filename.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe)
    safe = safe.strip("._") or "ontology"
    if len(safe) > 255:
        safe = safe[:255]
    if safe != original:
        logging.getLogger("DocsToKG.OntologyDownload").warning(
            "sanitized unsafe filename",
            extra={"stage": "sanitize", "original": original, "sanitized": safe},
        )
    return safe


def validate_url_security(url: str) -> str:
    """Validate URLs to avoid SSRF and insecure schemes.

    Args:
        url: URL returned by a resolver for ontology download.

    Returns:
        HTTPS URL safe for downstream download operations.

    Raises:
        ConfigError: If the URL uses an insecure scheme or resolves to private addresses.
    """

    parsed = urlparse(url)
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    if parsed.scheme == "http":
        logger.warning(
            "upgrading http url to https",
            extra={"stage": "download", "original_url": url},
        )
        parsed = parsed._replace(scheme="https")
    if parsed.scheme != "https":
        raise ConfigError("Only HTTPS URLs are allowed for ontology downloads")
    host = parsed.hostname
    if not host:
        raise ConfigError("URL must include hostname")
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    if is_ip:
        address = ipaddress.ip_address(host)
        if address.is_private or address.is_loopback or address.is_reserved or address.is_multicast:
            raise ConfigError(f"Refusing to download from private address {host}")
    else:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise ConfigError(f"Unable to resolve hostname {host}") from exc
        for info in infos:
            candidate_ip = ipaddress.ip_address(info[4][0])
            if (
                candidate_ip.is_private
                or candidate_ip.is_loopback
                or candidate_ip.is_reserved
                or candidate_ip.is_multicast
            ):
                raise ConfigError(f"Refusing to download from private address resolved for {host}")
    return urlunparse(parsed)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for the provided file.

    Args:
        path: Path to the file whose digest should be calculated.

    Returns:
        Hexadecimal SHA-256 checksum string.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_zip_safe(
    zip_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract a ZIP archive while preventing path traversal.

    Args:
        zip_path: Path to the ZIP file to extract.
        destination: Directory where extracted files should be stored.
        logger: Optional logger for emitting extraction telemetry.

    Returns:
        List of extracted file paths.

    Raises:
        ConfigError: If the archive contains unsafe paths or is missing.
    """

    if not zip_path.exists():
        raise ConfigError(f"ZIP archive not found: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ConfigError(f"Unsafe path detected in archive: {member.filename}")
            target_path = destination / member_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                continue
            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(target_path)
    if logger:
        logger.info(
            "extracted zip archive",
            extra={"stage": "extract", "archive": str(zip_path), "files": len(extracted)},
        )
    return extracted


class StreamingDownloader(pooch.HTTPDownloader):
    """Custom downloader to support conditional requests, resume, and caching."""

    def __init__(
        self,
        *,
        destination: Path,
        headers: Dict[str, str],
        http_config: DownloadConfiguration,
        previous_manifest: Optional[Dict[str, object]],
        logger: logging.Logger,
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

    def __call__(self, url: str, output_file: str, pooch_logger: logging.Logger) -> None:  # type: ignore[override]
        """Stream ontology content to disk while enforcing download policies.

        Args:
            url: Secure download URL resolved by the planner.
            output_file: Temporary filename managed by pooch during download.
            pooch_logger: Logger instance supplied by pooch (unused).

        Raises:
            ConfigError: If download limits are exceeded or filesystem errors occur.
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
        resume_position = part_path.stat().st_size if part_path.exists() else 0
        if resume_position:
            request_headers["Range"] = f"bytes={resume_position}-"
        attempt = 0
        session = requests.Session()
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
                        part_path.unlink(missing_ok=True)
                        return
                    if response.status_code == 206:
                        self.status = "updated"
                    response.raise_for_status()
                    length_header = response.headers.get("Content-Length")
                    total_bytes: Optional[int] = None
                    next_progress: Optional[float] = 0.1
                    if length_header:
                        try:
                            total_bytes = int(length_header)
                        except ValueError:
                            total_bytes = None
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
                            raise ConfigError(
                                f"File size {total_bytes} exceeds configured limit of {self.http_config.max_download_size_gb} GB"
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
                                                    "percent": round(min(progress, 1.0) * 100, 1)
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
                                    raise ConfigError(
                                        "Download exceeded maximum configured size while streaming"
                                    )
                    except OSError as exc:
                        part_path.unlink(missing_ok=True)
                        self.logger.error(
                            "filesystem error during download",
                            extra={"stage": "download", "error": str(exc)},
                        )
                        if "No space left" in str(exc):
                            raise ConfigError(
                                "No space left on device while writing download"
                            ) from exc
                        raise ConfigError(f"Failed to write download: {exc}") from exc
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


def _get_bucket(host: str, http_config: DownloadConfiguration) -> TokenBucket:
    bucket = _TOKEN_BUCKETS.get(host)
    if bucket is None:
        bucket = TokenBucket(
            http_config.rate_limit_per_second(), capacity=http_config.rate_limit_per_second()
        )
        _TOKEN_BUCKETS[host] = bucket
    return bucket


def download_stream(
    *,
    url: str,
    destination: Path,
    headers: Dict[str, str],
    previous_manifest: Optional[Dict[str, object]],
    http_config: DownloadConfiguration,
    cache_dir: Path,
    logger: logging.Logger,
) -> DownloadResult:
    """Download ontology content with caching, retries, and hash validation.

    Args:
        url: URL of the ontology document to download.
        destination: Target file path for the downloaded content.
        headers: HTTP headers forwarded to the download request.
        previous_manifest: Manifest metadata from a prior run, used for caching.
        http_config: Download configuration containing timeouts and limits.
        cache_dir: Directory where intermediary cached files are stored.
        logger: Logger adapter for structured download telemetry.

    Returns:
        DownloadResult describing the final artifact and metadata.

    Raises:
        ConfigError: If validation fails, limits are exceeded, or HTTP errors occur.
    """
    secure_url = validate_url_security(url)
    parsed = urlparse(secure_url)
    bucket = _get_bucket(parsed.hostname or "default", http_config)
    bucket.consume()

    start_time = time.monotonic()
    _log_memory(logger, "before")
    downloader = StreamingDownloader(
        destination=destination,
        headers=headers,
        http_config=http_config,
        previous_manifest=previous_manifest,
        logger=logger,
    )
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
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.HTTPError,
        requests.exceptions.SSLError,
    ) as exc:
        logger.error(
            "download request failed",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise ConfigError(f"HTTP error while downloading {secure_url}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive catch for pooch errors
        logger.error(
            "pooch download error",
            extra={"stage": "download", "url": secure_url, "error": str(exc)},
        )
        raise ConfigError(f"Download failed for {secure_url}: {exc}") from exc
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
        _log_memory(logger, "after")
        return DownloadResult(
            path=destination,
            status="cached",
            sha256=sha256,
            etag=downloader.response_etag,
            last_modified=downloader.response_last_modified,
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
    _log_memory(logger, "after")
    return DownloadResult(
        path=destination,
        status=downloader.status,
        sha256=sha256,
        etag=downloader.response_etag,
        last_modified=downloader.response_last_modified,
    )


__all__ = [
    "DownloadResult",
    "extract_zip_safe",
    "download_stream",
    "sanitize_filename",
    "sha256_file",
    "validate_url_security",
]
