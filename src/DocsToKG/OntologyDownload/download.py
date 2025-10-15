"""
Ontology Download Utilities

This module houses secure download helpers that implement per-host and
per-service rate limiting, Content-Type and size validation via preliminary
HEAD requests, internationalized domain name hardening, and resumable
transfers for ontology documents. It works in concert with resolver planning
to ensure that downloaded artifacts respect size limits, host allowlists, and
are safe for downstream processing, including guarded archive extraction.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import shutil
import socket
import tarfile
import threading
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set
from urllib.parse import ParseResult, urlparse, urlunparse

import pooch
import psutil
import requests

from .config import ConfigError, DownloadConfiguration


def _log_memory(logger: logging.Logger, event: str) -> None:
    """Emit debug-level memory usage snapshots when enabled.

    Args:
        logger: Logger instance controlling verbosity for download telemetry.
        event: Short label describing the lifecycle point (e.g., ``before``).

    Returns:
        None
    """
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

    Examples:
        >>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None)
        >>> result.status
        'fresh'
    """

    path: Path
    status: str
    sha256: str
    etag: Optional[str]
    last_modified: Optional[str]


class DownloadFailure(ConfigError):
    """Raised when an HTTP download attempt fails.

    Attributes:
        status_code: Optional HTTP status code returned by the upstream service.
        retryable: Whether the failure is safe to retry with an alternate resolver.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class TokenBucket:
    """Token bucket used to enforce per-host and per-service rate limits.

    Each unique combination of host and logical service identifier receives
    its own bucket so resolvers can honour provider-specific throttling
    guidance without starving other endpoints.

    Attributes:
        rate: Token replenishment rate per second.
        capacity: Maximum number of tokens the bucket may hold.
        tokens: Current token balance available for consumption.
        timestamp: Monotonic timestamp of the last refill.
        lock: Threading lock protecting bucket state.

    Examples:
        >>> bucket = TokenBucket(rate_per_sec=2.0, capacity=4.0)
        >>> bucket.consume(1.0)  # consumes immediately
        >>> isinstance(bucket.tokens, float)
        True
    """

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


_RETRYABLE_HTTP_STATUSES = {403, 408, 425, 429, 500, 502, 503, 504}


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


def _enforce_idn_safety(host: str) -> None:
    """Validate internationalized hostnames and reject suspicious patterns.

    Args:
        host: Hostname component extracted from the download URL.

    Returns:
        None

    Raises:
        ConfigError: If the hostname mixes multiple scripts or contains invisible characters.
    """

    if all(ord(char) < 128 for char in host):
        return

    scripts = set()
    for char in host:
        if ord(char) < 128:
            continue

        category = unicodedata.category(char)
        if category in {"Mn", "Me", "Cf"}:
            raise ConfigError("Internationalized host contains invisible characters")

        try:
            name = unicodedata.name(char)
        except ValueError as exc:
            raise ConfigError("Internationalized host contains unknown characters") from exc

        for script in ("LATIN", "CYRILLIC", "GREEK"):
            if script in name:
                scripts.add(script)
                break

    if len(scripts) > 1:
        raise ConfigError("Internationalized host mixes multiple scripts")


def _rebuild_netloc(parsed: ParseResult, ascii_host: str) -> str:
    """Reconstruct URL netloc with a normalized hostname.

    Args:
        parsed: Parsed URL components produced by :func:`urllib.parse.urlparse`.
        ascii_host: ASCII-normalized hostname (potentially IPv6).

    Returns:
        String suitable for use as the netloc portion of a URL.
    """

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        auth = f"{auth}@"

    host_component = ascii_host
    if ":" in host_component and not host_component.startswith("["):
        host_component = f"[{host_component}]"

    port = f":{parsed.port}" if parsed.port else ""
    return f"{auth}{host_component}{port}"


def validate_url_security(url: str, http_config: Optional[DownloadConfiguration] = None) -> str:
    """Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

    Hostnames are converted to punycode before resolution, and both direct IP
    addresses and DNS results are rejected when they target private or loopback
    ranges to prevent server-side request forgery.

    Args:
        url: URL returned by a resolver for ontology download.
        http_config: Download configuration providing optional host allowlist.

    Returns:
        HTTPS URL safe for downstream download operations.

    Raises:
        ConfigError: If the URL violates security requirements or allowlists.
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

    ascii_host = host.lower()
    if not is_ip:
        _enforce_idn_safety(host)
        try:
            ascii_host = host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ConfigError(f"Invalid internationalized hostname: {host}") from exc

    parsed = parsed._replace(netloc=_rebuild_netloc(parsed, ascii_host))

    allowed = http_config.normalized_allowed_hosts() if http_config else None
    if allowed:
        exact, suffixes = allowed
        if ascii_host not in exact and not any(
            ascii_host == suffix or ascii_host.endswith(f".{suffix}") for suffix in suffixes
        ):
            raise ConfigError(f"Host {host} not in allowlist")

    if is_ip:
        address = ipaddress.ip_address(ascii_host)
        if address.is_private or address.is_loopback or address.is_reserved or address.is_multicast:
            raise ConfigError(f"Refusing to download from private address {host}")
        return urlunparse(parsed)

    try:
        infos = socket.getaddrinfo(ascii_host, None)
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


_MAX_COMPRESSION_RATIO = 10.0


def _validate_member_path(member_name: str) -> Path:
    """Validate archive member paths to prevent traversal attacks.

    Args:
        member_name: Path declared within the archive.

    Returns:
        Sanitised relative path safe for extraction on the local filesystem.

    Raises:
        ConfigError: If the member path is absolute or contains traversal segments.
    """

    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute():
        raise ConfigError(f"Unsafe absolute path detected in archive: {member_name}")
    if not relative.parts:
        raise ConfigError(f"Empty path detected in archive: {member_name}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ConfigError(f"Unsafe path detected in archive: {member_name}")
    return Path(*relative.parts)


def _check_compression_ratio(
    *,
    total_uncompressed: int,
    compressed_size: int,
    archive: Path,
    logger: Optional[logging.Logger],
    archive_type: str,
) -> None:
    """Ensure compressed archives do not expand beyond the permitted ratio.

    Args:
        total_uncompressed: Sum of file sizes within the archive.
        compressed_size: Archive file size on disk (or sum of compressed entries).
        archive: Path to the archive on disk.
        logger: Optional logger for emitting diagnostic messages.
        archive_type: Human readable label for error messages (ZIP/TAR).

    Raises:
        ConfigError: If the archive exceeds the allowed expansion ratio.
    """

    if compressed_size <= 0:
        return
    ratio = total_uncompressed / float(compressed_size)
    if ratio > _MAX_COMPRESSION_RATIO:
        if logger:
            logger.error(
                "archive compression ratio too high",
                extra={
                    "stage": "extract",
                    "archive": str(archive),
                    "ratio": round(ratio, 2),
                    "compressed_bytes": compressed_size,
                    "uncompressed_bytes": total_uncompressed,
                    "limit": _MAX_COMPRESSION_RATIO,
                },
            )
        raise ConfigError(
            f"{archive_type} archive {archive} expands to {total_uncompressed} bytes, "
            f"exceeding {_MAX_COMPRESSION_RATIO}:1 compression ratio"
        )


def extract_zip_safe(
    zip_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract a ZIP archive while preventing traversal and compression bombs.

    Args:
        zip_path: Path to the ZIP file to extract.
        destination: Directory where extracted files should be stored.
        logger: Optional logger for emitting extraction telemetry.

    Returns:
        List of extracted file paths.

    Raises:
        ConfigError: If the archive contains unsafe paths, compression bombs, or is missing.
    """

    if not zip_path.exists():
        raise ConfigError(f"ZIP archive not found: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        safe_members: List[tuple[zipfile.ZipInfo, Path]] = []
        total_uncompressed = 0
        for member in members:
            member_path = _validate_member_path(member.filename)
            if member.is_dir():
                safe_members.append((member, member_path))
                continue
            total_uncompressed += int(member.file_size)
            safe_members.append((member, member_path))
        compressed_size = max(
            zip_path.stat().st_size,
            sum(int(member.compress_size) for member in members) or 0,
        )
        _check_compression_ratio(
            total_uncompressed=total_uncompressed,
            compressed_size=compressed_size,
            archive=zip_path,
            logger=logger,
            archive_type="ZIP",
        )
        for member, member_path in safe_members:
            target_path = destination / member_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(target_path)
    if logger:
        logger.info(
            "extracted zip archive",
            extra={"stage": "extract", "archive": str(zip_path), "files": len(extracted)},
        )
    return extracted


def extract_tar_safe(
    tar_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

    Args:
        tar_path: Path to the tar archive (tar, tar.gz, tar.xz).
        destination: Directory where extracted files should be stored.
        logger: Optional logger for emitting extraction telemetry.

    Returns:
        List of extracted file paths.

    Raises:
        ConfigError: If the archive is missing, unsafe, or exceeds compression limits.
    """

    if not tar_path.exists():
        raise ConfigError(f"TAR archive not found: {tar_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    try:
        with tarfile.open(tar_path, mode="r:*") as archive:
            members = archive.getmembers()
            safe_members: List[tuple[tarfile.TarInfo, Path]] = []
            total_uncompressed = 0
            for member in members:
                member_path = _validate_member_path(member.name)
                if member.isdir():
                    safe_members.append((member, member_path))
                    continue
                if member.islnk() or member.issym():
                    raise ConfigError(f"Unsafe link detected in archive: {member.name}")
                if member.isdev():
                    raise ConfigError(
                        f"Unsupported special file detected in archive: {member.name}"
                    )
                if not member.isfile():
                    raise ConfigError(f"Unsupported tar member type encountered: {member.name}")
                total_uncompressed += int(member.size)
                safe_members.append((member, member_path))
            compressed_size = tar_path.stat().st_size
            _check_compression_ratio(
                total_uncompressed=total_uncompressed,
                compressed_size=compressed_size,
                archive=tar_path,
                logger=logger,
                archive_type="TAR",
            )
            for member, member_path in safe_members:
                if member.isdir():
                    (destination / member_path).mkdir(parents=True, exist_ok=True)
                    continue
                target_path = destination / member_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_file = archive.extractfile(member)
                if extracted_file is None:
                    raise ConfigError(f"Failed to extract member: {member.name}")
                with extracted_file as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted.append(target_path)
    except tarfile.TarError as exc:
        raise ConfigError(f"Failed to extract tar archive {tar_path}: {exc}") from exc
    if logger:
        logger.info(
            "extracted tar archive",
            extra={"stage": "extract", "archive": str(tar_path), "files": len(extracted)},
        )
    return extracted


_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")


def extract_archive_safe(
    archive_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Extract archives by dispatching to the appropriate safe handler."""

    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        return extract_zip_safe(archive_path, destination, logger=logger)
    if any(lower_name.endswith(suffix) for suffix in _TAR_SUFFIXES):
        return extract_tar_safe(archive_path, destination, logger=logger)
    raise ConfigError(f"Unsupported archive format: {archive_path}")


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
        >>> from DocsToKG.OntologyDownload.config import DownloadConfiguration
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
            ConfigError: If the origin reports a payload larger than the
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
                raise ConfigError(
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
        head_content_type, _ = self._preliminary_head_check(url, session)
        if head_content_type:
            self._validate_media_type(head_content_type, self.expected_media_type, url)
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
                    self._validate_media_type(
                        response.headers.get("Content-Type"),
                        self.expected_media_type,
                        url,
                    )
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


def _get_bucket(
    host: str, http_config: DownloadConfiguration, service: Optional[str] = None
) -> TokenBucket:
    """Return a token bucket keyed by host and optional service name.

    Args:
        host: Hostname extracted from the download URL.
        http_config: Download configuration providing base rate limits.
        service: Logical service identifier enabling per-service overrides.

    Returns:
        TokenBucket instance shared across downloads for throttling, seeded
        with either per-host defaults or service-specific overrides.
    """
    key = f"{service}:{host}" if service else host
    bucket = _TOKEN_BUCKETS.get(key)
    if bucket is None:
        rate = http_config.rate_limit_per_second()
        if service:
            service_rate = http_config.parse_service_rate_limit(service)
            if service_rate:
                rate = service_rate
        bucket = TokenBucket(rate_per_sec=rate, capacity=rate)
        _TOKEN_BUCKETS[key] = bucket
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
        ConfigError: If validation fails, limits are exceeded, or HTTP errors occur.
    """
    secure_url = validate_url_security(url, http_config)
    parsed = urlparse(secure_url)
    bucket = _get_bucket(parsed.hostname or "default", http_config, service)
    bucket.consume()

    start_time = time.monotonic()
    _log_memory(logger, "before")
    downloader = StreamingDownloader(
        destination=destination,
        headers=headers,
        http_config=http_config,
        previous_manifest=previous_manifest,
        logger=logger,
        expected_media_type=expected_media_type,
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
        raise DownloadFailure(
            message, status_code=status_code, retryable=retryable
        ) from exc
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
    _log_memory(logger, "after")
    return DownloadResult(
        path=destination,
        status=downloader.status,
        sha256=sha256,
        etag=downloader.response_etag,
        last_modified=downloader.response_last_modified,
    )


__all__ = [
    "DownloadFailure",
    "DownloadResult",
    "RDF_MIME_ALIASES",
    "RDF_MIME_FORMAT_LABELS",
    "extract_archive_safe",
    "extract_zip_safe",
    "download_stream",
    "sanitize_filename",
    "sha256_file",
    "validate_url_security",
]
