# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io_safe",
#   "purpose": "IO safety helpers for the ontology downloader",
#   "sections": [
#     {
#       "id": "sanitize-filename",
#       "name": "sanitize_filename",
#       "anchor": "function-sanitize-filename",
#       "kind": "function"
#     },
#     {
#       "id": "generate-correlation-id",
#       "name": "generate_correlation_id",
#       "anchor": "function-generate-correlation-id",
#       "kind": "function"
#     },
#     {
#       "id": "mask-sensitive-data",
#       "name": "mask_sensitive_data",
#       "anchor": "function-mask-sensitive-data",
#       "kind": "function"
#     },
#     {
#       "id": "enforce-idn-safety",
#       "name": "_enforce_idn_safety",
#       "anchor": "function-enforce-idn-safety",
#       "kind": "function"
#     },
#     {
#       "id": "rebuild-netloc",
#       "name": "_rebuild_netloc",
#       "anchor": "function-rebuild-netloc",
#       "kind": "function"
#     },
#     {
#       "id": "validate-url-security",
#       "name": "validate_url_security",
#       "anchor": "function-validate-url-security",
#       "kind": "function"
#     },
#     {
#       "id": "sha256-file",
#       "name": "sha256_file",
#       "anchor": "function-sha256-file",
#       "kind": "function"
#     },
#     {
#       "id": "validate-member-path",
#       "name": "_validate_member_path",
#       "anchor": "function-validate-member-path",
#       "kind": "function"
#     },
#     {
#       "id": "check-compression-ratio",
#       "name": "_check_compression_ratio",
#       "anchor": "function-check-compression-ratio",
#       "kind": "function"
#     },
#     {
#       "id": "extract-zip-safe",
#       "name": "extract_zip_safe",
#       "anchor": "function-extract-zip-safe",
#       "kind": "function"
#     },
#     {
#       "id": "extract-tar-safe",
#       "name": "extract_tar_safe",
#       "anchor": "function-extract-tar-safe",
#       "kind": "function"
#     },
#     {
#       "id": "extract-archive-safe",
#       "name": "extract_archive_safe",
#       "anchor": "function-extract-archive-safe",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Filesystem and payload safety utilities for the ontology downloader."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import shutil
import socket
import stat
import tarfile
import unicodedata
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional
from urllib.parse import ParseResult, urlparse, urlunparse

from .config import ConfigError, DownloadConfiguration

__all__ = [
    "sanitize_filename",
    "generate_correlation_id",
    "mask_sensitive_data",
    "validate_url_security",
    "sha256_file",
    "extract_zip_safe",
    "extract_tar_safe",
    "extract_archive_safe",
]


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename derived from ``filename``."""

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


def generate_correlation_id() -> str:
    """Return a short-lived identifier that links related log entries."""

    return uuid.uuid4().hex[:12]


def mask_sensitive_data(payload: Dict[str, object]) -> Dict[str, object]:
    """Return a copy of ``payload`` with common secret fields masked."""

    sensitive_keys = {"authorization", "api_key", "apikey", "token", "secret", "password"}
    token_pattern = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")

    def _mask_value(value: object, key_hint: Optional[str] = None) -> object:
        if isinstance(value, dict):
            return {
                sub_key: _mask_value(sub_value, sub_key.lower())
                for sub_key, sub_value in value.items()
            }
        if isinstance(value, list):
            return [_mask_value(item, key_hint) for item in value]
        if isinstance(value, tuple):
            return tuple(_mask_value(item, key_hint) for item in value)
        if isinstance(value, set):
            return {_mask_value(item, key_hint) for item in value}
        if isinstance(value, str):
            lowered = value.lower()
            if key_hint in sensitive_keys:
                return "***masked***"
            if "apikey" in lowered:
                return "***masked***"
            if key_hint == "authorization":
                token = value.strip()
                if "bearer " in lowered or token_pattern.match(token):
                    return "***masked***"
            if "bearer " in lowered:
                return "***masked***"
        return value

    masked: Dict[str, object] = {}
    for key, value in payload.items():
        lower = key.lower()
        if lower in sensitive_keys:
            masked[key] = "***masked***"
        else:
            masked[key] = _mask_value(value, lower)
    return masked


def _enforce_idn_safety(host: str) -> None:
    """Validate internationalized hostnames and reject suspicious patterns."""

    if all(ord(char) < 128 for char in host):
        return

    scripts = set()
    for char in host:
        if ord(char) < 128:
            if char.isalpha():
                scripts.add("LATIN")
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
    """Reconstruct URL netloc with a normalized hostname."""

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
    """Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists."""

    parsed = urlparse(url)
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ConfigError("Only HTTP(S) URLs are allowed for ontology downloads")

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
    allow_private = False
    if allowed:
        exact, suffixes = allowed
        if ascii_host in exact or any(
            ascii_host == suffix or ascii_host.endswith(f".{suffix}") for suffix in suffixes
        ):
            allow_private = True
        else:
            raise ConfigError(f"Host {host} not in allowlist")

    if scheme == "http":
        if allow_private:
            logger.warning(
                "allowing http url for explicit allowlist host",
                extra={"stage": "download", "original_url": url},
            )
        else:
            logger.warning(
                "upgrading http url to https",
                extra={"stage": "download", "original_url": url},
            )
            parsed = parsed._replace(scheme="https")
            scheme = "https"

    if scheme != "https" and not allow_private:
        raise ConfigError("Only HTTPS URLs are allowed for ontology downloads")

    if is_ip:
        address = ipaddress.ip_address(ascii_host)
        if not allow_private and (
            address.is_private or address.is_loopback or address.is_reserved or address.is_multicast
        ):
            raise ConfigError(f"Refusing to download from private address {host}")
        return urlunparse(parsed)

    try:
        infos = socket.getaddrinfo(ascii_host, None)
    except socket.gaierror as exc:
        logger.warning(
            "dns resolution failed",
            extra={"stage": "download", "hostname": host, "error": str(exc)},
        )
        return urlunparse(parsed)

    for info in infos:
        candidate_ip = ipaddress.ip_address(info[4][0])
        if not allow_private and (
            candidate_ip.is_private
            or candidate_ip.is_loopback
            or candidate_ip.is_reserved
            or candidate_ip.is_multicast
        ):
            raise ConfigError(f"Refusing to download from private address resolved for {host}")

    return urlunparse(parsed)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for the provided file."""

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_MAX_COMPRESSION_RATIO = 10.0


def _validate_member_path(member_name: str) -> Path:
    """Validate archive member paths to prevent traversal attacks."""

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
    """Ensure compressed archives do not expand beyond the permitted ratio."""

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
    """Extract a ZIP archive while preventing traversal and compression bombs."""

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
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ConfigError(f"Unsafe link detected in archive: {member.filename}")
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
    """Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks."""

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
