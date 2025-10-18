"""Filesystem helpers for ontology downloads."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import stat
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from ..errors import ConfigError, DownloadFailure
from ..settings import get_default_config

_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")
_MAX_COMPRESSION_RATIO = 10.0

def _resolve_max_uncompressed_bytes(limit: Optional[int]) -> Optional[int]:
    """Return the effective archive expansion limit, honoring runtime overrides."""

    if limit is not None:
        return limit
    return get_default_config().defaults.http.max_uncompressed_bytes()

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

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for the provided file."""

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _compute_file_hash(path: Path, algorithm: str) -> str:
    """Compute ``algorithm`` digest for ``path``."""

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:  # pragma: no cover - defensive guard for unsupported algs
        raise ValueError(f"Unsupported checksum algorithm '{algorithm}'") from exc
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

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

def _enforce_uncompressed_ceiling(
    *,
    total_uncompressed: int,
    limit_bytes: Optional[int],
    archive: Path,
    logger: Optional[logging.Logger],
    archive_type: str,
) -> None:
    """Ensure uncompressed payload stays within configured limits."""

    if limit_bytes is None or limit_bytes <= 0:
        return
    if total_uncompressed <= limit_bytes:
        return
    if logger:
        logger.error(
            "archive uncompressed size exceeds limit",
            extra={
                "stage": "extract",
                "archive": str(archive),
                "uncompressed_bytes": total_uncompressed,
                "limit_bytes": limit_bytes,
                "archive_type": archive_type,
            },
        )
    raise ConfigError(
        f"{archive_type} archive {archive} expands to {total_uncompressed} bytes, "
        f"exceeding configured ceiling of {limit_bytes} bytes"
    )

def extract_zip_safe(
    zip_path: Path,
    destination: Path,
    *,
    logger: Optional[logging.Logger] = None,
    max_uncompressed_bytes: Optional[int] = None,
) -> List[Path]:
    """Extract a ZIP archive while preventing traversal and compression bombs."""

    if not zip_path.exists():
        raise ConfigError(f"ZIP archive not found: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    limit_bytes = _resolve_max_uncompressed_bytes(max_uncompressed_bytes)
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
        _enforce_uncompressed_ceiling(
            total_uncompressed=total_uncompressed,
            limit_bytes=limit_bytes,
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
    tar_path: Path,
    destination: Path,
    *,
    logger: Optional[logging.Logger] = None,
    max_uncompressed_bytes: Optional[int] = None,
) -> List[Path]:
    """Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks."""

    if not tar_path.exists():
        raise ConfigError(f"TAR archive not found: {tar_path}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    limit_bytes = _resolve_max_uncompressed_bytes(max_uncompressed_bytes)
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
            _enforce_uncompressed_ceiling(
                total_uncompressed=total_uncompressed,
                limit_bytes=limit_bytes,
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

def extract_archive_safe(
    archive_path: Path,
    destination: Path,
    *,
    logger: Optional[logging.Logger] = None,
    max_uncompressed_bytes: Optional[int] = None,
) -> List[Path]:
    """Extract archives by dispatching to the appropriate safe handler."""

    lower_name = archive_path.name.lower()
    limit_bytes = _resolve_max_uncompressed_bytes(max_uncompressed_bytes)
    if lower_name.endswith(".zip"):
        return extract_zip_safe(
            archive_path,
            destination,
            logger=logger,
            max_uncompressed_bytes=limit_bytes,
        )
    if any(lower_name.endswith(suffix) for suffix in _TAR_SUFFIXES):
        return extract_tar_safe(
            archive_path,
            destination,
            logger=logger,
            max_uncompressed_bytes=limit_bytes,
        )
    raise ConfigError(f"Unsupported archive format: {archive_path}")

def _materialize_cached_file(source: Path, destination: Path) -> tuple[Path, Path]:
    """Link or move ``source`` into ``destination`` without redundant copies.

    Returns a tuple ``(artifact_path, cache_path)`` where ``artifact_path`` is the final
    destination path and ``cache_path`` points to the retained cache file (which may be
    identical to ``artifact_path`` when the cache entry is moved instead of linked).
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination, destination
    if destination.exists():
        try:
            if destination.samefile(source):
                return destination, source
        except (FileNotFoundError, OSError):
            pass
    temp_path = destination.with_suffix(destination.suffix + ".tmpdownload")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.unlink(missing_ok=True)
    try:
        os.link(source, temp_path)
        os.replace(temp_path, destination)
        return destination, source
    except OSError:
        shutil.move(str(source), str(temp_path))
        os.replace(temp_path, destination)
        return destination, destination
    finally:
        temp_path.unlink(missing_ok=True)

def format_bytes(num: int) -> str:
    """Return a human-readable representation for ``num`` bytes."""

    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"
