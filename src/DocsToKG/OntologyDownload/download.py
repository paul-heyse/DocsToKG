"""HTTP download utilities for ontology artifacts."""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse

import pooch
import requests
import socket

from .config import ConfigError, DownloadConfiguration


@dataclass(slots=True)
class DownloadResult:
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
    """Sanitize filenames to prevent directory traversal and unsafe characters."""

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
    """Validate URLs to avoid SSRF and insecure schemes."""

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
            if candidate_ip.is_private or candidate_ip.is_loopback or candidate_ip.is_reserved or candidate_ip.is_multicast:
                raise ConfigError(f"Refusing to download from private address resolved for {host}")
    return urlunparse(parsed)


def sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class StreamingDownloader(pooch.HTTPDownloader):
    """Custom downloader to support conditional requests and resume."""

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
                        self.response_etag = response.headers.get("ETag") or self.previous_manifest.get("etag")
                        self.response_last_modified = response.headers.get("Last-Modified") or self.previous_manifest.get(
                            "last_modified"
                        )
                        part_path.unlink(missing_ok=True)
                        return
                    if response.status_code == 206:
                        self.status = "updated"
                    response.raise_for_status()
                    length_header = response.headers.get("Content-Length")
                    if length_header:
                        max_bytes = self.http_config.max_download_size_gb * (1024 ** 3)
                        if int(length_header) > max_bytes:
                            self.logger.error(
                                "file exceeds size limit",
                                extra={
                                    "stage": "download",
                                    "size": int(length_header),
                                    "limit": max_bytes,
                                },
                            )
                            raise ConfigError(
                                f"File size {int(length_header)} exceeds configured limit of {self.http_config.max_download_size_gb} GB"
                            )
                    self.response_etag = response.headers.get("ETag")
                    self.response_last_modified = response.headers.get("Last-Modified")
                    mode = "ab" if resume_position else "wb"
                    bytes_downloaded = resume_position
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    with part_path.open(mode) as fh:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            bytes_downloaded += len(chunk)
                            if bytes_downloaded > self.http_config.max_download_size_gb * (1024 ** 3):
                                self.logger.error(
                                    "download exceeded size limit",
                                    extra={
                                        "stage": "download",
                                        "size": bytes_downloaded,
                                        "limit": self.http_config.max_download_size_gb * (1024 ** 3),
                                    },
                                )
                                raise ConfigError(
                                    "Download exceeded maximum configured size while streaming"
                                )
                    break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError, requests.exceptions.SSLError) as exc:
                if attempt > self.http_config.max_retries:
                    raise
                sleep_time = self.http_config.backoff_factor * (2 ** (attempt - 1))
                self.logger.warning(
                    "download retry", extra={"stage": "download", "attempt": attempt, "sleep_sec": sleep_time, "error": str(exc)}
                )
                time.sleep(sleep_time)
        part_path.replace(Path(output_file))


def _get_bucket(host: str, http_config: DownloadConfiguration) -> TokenBucket:
    bucket = _TOKEN_BUCKETS.get(host)
    if bucket is None:
        bucket = TokenBucket(http_config.rate_limit_per_second(), capacity=http_config.rate_limit_per_second())
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
    secure_url = validate_url_security(url)
    parsed = urlparse(secure_url)
    bucket = _get_bucket(parsed.hostname or "default", http_config)
    bucket.consume()

    start_time = time.monotonic()
    downloader = StreamingDownloader(
        destination=destination,
        headers=headers,
        http_config=http_config,
        previous_manifest=previous_manifest,
        logger=logger,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(destination.name)
    cached_path = Path(
        pooch.retrieve(
            secure_url,
            path=cache_dir,
            fname=safe_name,
            downloader=downloader,
            progressbar=False,
        )
    )
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
    return DownloadResult(
        path=destination,
        status=downloader.status,
        sha256=sha256,
        etag=downloader.response_etag,
        last_modified=downloader.response_last_modified,
    )


__all__ = [
    "DownloadResult",
    "download_stream",
    "sanitize_filename",
    "sha256_file",
    "validate_url_security",
]
