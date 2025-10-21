# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.filesystem",
#   "purpose": "Provide filesystem utilities for sanitisation, hashing, masking, and archive extraction",
#   "sections": [
#     {"id": "limits", "name": "Archive Limits & Helpers", "anchor": "LIM", "kind": "helpers"},
#     {"id": "sanitisation", "name": "Filename Sanitisation & Identifiers", "anchor": "SAN", "kind": "helpers"},
#     {"id": "masking", "name": "Sensitive Data Masking", "anchor": "MSK", "kind": "helpers"},
#     {"id": "hashing", "name": "Hashing Utilities", "anchor": "HAS", "kind": "helpers"},
#     {"id": "archives", "name": "Archive Extraction Utilities", "anchor": "ARC", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Filesystem helpers for ontology downloads.

Responsibilities include sanitising filenames, generating correlation IDs for
log entries, masking sensitive payloads, computing file hashes, and safely
extracting archives while enforcing expansion limits.  These utilities are
shared by both the planner and validator stages to ensure consistent handling of
artefacts on disk and to support content-addressable mirroring via the storage
backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import libarchive

from ..errors import ConfigError
from ..policy.errors import PolicyReject
from ..policy.gates import extraction_gate, filesystem_gate
from ..settings import get_default_config
from .extraction_constraints import ExtractionGuardian, PreScanValidator
from .extraction_policy import ExtractionPolicy, ExtractionSettings, safe_defaults
from .extraction_telemetry import (
    ExtractionErrorCode,
    ExtractionMetrics,
    ExtractionTelemetryEvent,
    error_message,
)

_MAX_COMPRESSION_RATIO = 10.0


def _compute_config_hash(policy: ExtractionPolicy) -> str:
    """Compute a deterministic hash of extraction policy for provenance tracking."""
    policy_str = json.dumps(
        {
            "encapsulate": policy.encapsulate,
            "encapsulation_name": policy.encapsulation_name,
            "max_depth": policy.max_depth,
            "max_components_len": policy.max_components_len,
            "max_path_len": policy.max_path_len,
            "normalize_unicode": policy.normalize_unicode,
            "casefold_collision_policy": policy.casefold_collision_policy,
            "max_entries": policy.max_entries,
            "max_file_size_bytes": policy.max_file_size_bytes,
            "max_entry_ratio": policy.max_entry_ratio,
            "allow_symlinks": policy.allow_symlinks,
            "allow_hardlinks": policy.allow_hardlinks,
        },
        sort_keys=True,
    )
    return hashlib.sha256(policy_str.encode("utf-8")).hexdigest()[:16]


def _write_audit_manifest(
    extract_root: Path,
    archive_path: Path,
    policy: ExtractionPolicy,
    entries_metadata: List[
        tuple[str, Path, int, Optional[str]]
    ],  # (orig_path, normalized_path, size, sha256)
    telemetry: "ExtractionTelemetryEvent",
    metrics: "ExtractionMetrics",
) -> None:
    """Write deterministic audit JSON manifest for extraction.

    Args:
        extract_root: Root directory where files were extracted
        archive_path: Path to the source archive
        policy: ExtractionPolicy used for extraction
        entries_metadata: List of (original_path, normalized_path, size, sha256_hash)
        telemetry: Telemetry event with extraction metadata
        metrics: Extraction metrics with duration and counts
    """
    try:
        # Compute archive SHA256
        archive_sha256 = hashlib.sha256()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                archive_sha256.update(chunk)

        # Build manifest
        manifest = {
            "schema_version": "1.0",
            "run_id": telemetry.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "archive_path": str(archive_path),
            "archive_sha256": archive_sha256.hexdigest(),
            "archive_size_bytes": archive_path.stat().st_size,
            "encapsulation_root": str(extract_root),
            "config_hash": _compute_config_hash(policy),
            "policy": policy.model_dump(),  # Gap 3: Full policy snapshot for complete provenance
            "metrics": {
                "entries_total": metrics.total_entries,
                "entries_allowed": metrics.entries_allowed,
                "entries_extracted": metrics.entries_extracted,
                "bytes_declared": metrics.total_bytes,
                "bytes_written": telemetry.bytes_written,
                "duration_ms": round(metrics.duration_ms, 2),
                "format": telemetry.format_name or "unknown",
                "filters": telemetry.filters or [],
            },
            "entries": [
                {
                    "path_rel": str(norm_path.relative_to(extract_root)),
                    "path_original": orig_path,
                    "size": size,
                    "sha256": sha256 if sha256 else "",
                    "scan_index": idx,
                }
                for idx, (orig_path, norm_path, size, sha256) in enumerate(entries_metadata)
            ],
        }

        # Write atomically
        manifest_path = extract_root / policy.manifest_filename
        manifest_tmp = manifest_path.with_suffix(".tmp")

        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        manifest_tmp.replace(manifest_path)

    except Exception as e:
        # Log but don't fail extraction if audit manifest fails
        logger = logging.getLogger("DocsToKG.OntologyDownload")
        logger.warning(
            "failed to write audit manifest",
            extra={
                "stage": "extract",
                "error": str(e),
                "manifest_path": str(extract_root / policy.manifest_filename),
            },
        )


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

    def _mask_header_pair(item: object) -> Optional[object]:
        """Mask values for header-like key/value tuples."""

        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            key_lower = item[0].lower()
            return (item[0], _mask_value(item[1], key_lower))
        return None

    def _mask_value(value: object, key_hint: Optional[str] = None) -> object:
        if isinstance(value, dict):
            return {
                sub_key: _mask_value(sub_value, sub_key.lower())
                for sub_key, sub_value in value.items()
            }
        if isinstance(value, list):
            masked_list = []
            for item in value:
                masked_pair = _mask_header_pair(item)
                if masked_pair is not None:
                    masked_list.append(masked_pair)
                else:
                    masked_list.append(_mask_value(item, key_hint))
            return masked_list
        if isinstance(value, tuple):
            masked_pair = _mask_header_pair(value)
            if masked_pair is not None:
                return masked_pair
            return tuple(_mask_header_pair(item) or _mask_value(item, key_hint) for item in value)
        if isinstance(value, set):
            masked_set = set()
            for item in value:
                masked_pair = _mask_header_pair(item)
                if masked_pair is not None:
                    masked_set.add(masked_pair)
                else:
                    masked_set.add(_mask_value(item, key_hint))
            return masked_set
        if isinstance(value, str):
            lowered = value.lower()
            if key_hint == "authorization":
                return "***masked***"
            if key_hint in sensitive_keys:
                return "***masked***"
            if "apikey" in lowered:
                return "***masked***"
            if "bearer " in lowered:
                return "***masked***"
            if token_pattern.fullmatch(value):
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

    # GATE 4: Filesystem Security (Path Traversal Prevention)
    try:
        fs_result = filesystem_gate(
            root_path="/",
            entry_paths=[member_name],
            allow_symlinks=False,
        )
        if isinstance(fs_result, PolicyReject):
            raise ConfigError(f"Filesystem policy violation: {fs_result.error_code}")
    except ConfigError:
        raise

    return Path(*relative.parts)


def _compute_archive_sha256(archive_path: Path) -> str:
    """Compute SHA-256 digest of archive for deterministic encapsulation naming."""
    return sha256_file(archive_path)


def _generate_encapsulation_root_name(
    archive_path: Path,
    policy: ExtractionPolicy,
) -> str:
    """Generate encapsulation root name based on policy.

    Args:
        archive_path: Path to the archive
        policy: Extraction policy (determines naming strategy)

    Returns:
        Directory name for the encapsulation root
    """
    if policy.encapsulation_name == "sha256":
        digest = _compute_archive_sha256(archive_path)
        return f"{digest[:12]}.d"
    elif policy.encapsulation_name == "basename":
        return f"{archive_path.stem}.d"
    else:
        raise ConfigError(f"Unknown encapsulation naming policy: {policy.encapsulation_name}")


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


def extract_archive_safe(
    archive_path: Path,
    destination: Path,
    *,
    logger: Optional[logging.Logger] = None,
    max_uncompressed_bytes: Optional[int] = None,
    extraction_policy: Optional[ExtractionPolicy] = None,
) -> List[Path]:
    """Extract archives safely using libarchive with validation and compression checks.

    This function uses libarchive for automatic format and compression detection, eliminating
    the need for format-specific branching. It implements a two-phase extraction strategy:

    Phase 1 (pre-scan): Validates all entries without writing to disk
      - Checks for path traversal attempts (absolute paths, `..`, escaping destination)
      - Rejects unsafe entry types (symlinks, hardlinks, devices, FIFOs, sockets)
      - Accumulates uncompressed sizes for compression ratio validation
      - Enforces zip-bomb guard (~10:1 ratio)

    Phase 2 (extract): Performs actual extraction only if Phase 1 passes
      - Creates directories as needed under the validated destination
      - Writes regular files to pre-validated target paths
      - Returns list of extracted file paths in header order

    Phase 1 (with encapsulation): Creates a deterministic subdirectory
      - Prevents tar-bomb-style extraction into sibling directories
      - Uses SHA256 or basename to name the root (configurable)
      - Enables DirFD + openat semantics for race-free operations (Phase 1)

    Args:
        archive_path: Path to the archive file (any libarchive-supported format)
        destination: Target directory for extraction
        logger: Optional logger for structured logging with stage="extract" key
        max_uncompressed_bytes: Maximum uncompressed size limit (uses config default if None)
        extraction_policy: ExtractionPolicy instance (uses safe defaults if None)

    Returns:
        List of Path objects for regular files extracted, in header order

    Raises:
        ConfigError: If archive is unsupported, corrupted, or violates security policy
    """
    if not archive_path.exists():
        raise ConfigError(f"Archive not found: {archive_path}")

    policy = extraction_policy or safe_defaults()
    # Pydantic v2 validates automatically on initialization; re-validate to provide clear error messages
    try:
        # Re-validate policy using Pydantic v2's validation
        _ = ExtractionSettings.model_validate(policy.model_dump())
    except Exception as e:
        raise ConfigError(f"Invalid extraction policy: {str(e)}")

    destination.mkdir(parents=True, exist_ok=True)

    # Initialize telemetry
    metrics = ExtractionMetrics()
    telemetry = ExtractionTelemetryEvent(archive=str(archive_path))

    try:
        # Determine final extraction root (with encapsulation if enabled)
        extract_root = destination
        if policy.encapsulate:
            root_name = _generate_encapsulation_root_name(archive_path, policy)
            extract_root = destination / root_name
            telemetry.encapsulated_root = str(extract_root)
            telemetry.encapsulation_policy = policy.encapsulation_name

            # Check if encapsulation root already exists
            if extract_root.exists():
                raise ConfigError(
                    error_message(
                        ExtractionErrorCode.OVERWRITE_ROOT,
                        f"Encapsulation root already exists: {extract_root}",
                    )
                )

        extract_root.mkdir(parents=True, exist_ok=True)

        # Compute and store config hash for provenance
        telemetry.config_hash = _compute_config_hash(policy)

        # Phase 1: Pre-scan validation without writing
        entries_to_extract: List[tuple[str, Path, bool]] = []  # (orig_name, validated_path, is_dir)
        total_uncompressed = 0

        # Initialize Phase 2 validator for comprehensive security checks
        prescan_validator = PreScanValidator(policy)

        # Initialize Phase 3-4 guardian for disk space and permissions
        guardian = ExtractionGuardian(policy)

        # GATE 3: Extraction Policy (Pre-Scan Validation)
        # Peek into archive to get total counts before extraction
        try:
            # Do a quick scan to get archive stats
            entries_total = 0
            bytes_declared = 0
            with libarchive.file_reader(str(archive_path)) as archive:
                for entry in archive:
                    entries_total += 1
                    if entry.size and entry.size > 0:
                        bytes_declared += entry.size

            # Invoke extraction gate with pre-scan data
            extraction_result = extraction_gate(
                entries_total=entries_total,
                bytes_declared=bytes_declared,
                max_total_ratio=_MAX_COMPRESSION_RATIO,
            )
            if isinstance(extraction_result, PolicyReject):
                log = logger or logging.getLogger(__name__)
                log.error(f"extraction gate rejected: {extraction_result.error_code}")
                raise ConfigError(f"Archive policy violation: {extraction_result.error_code}")

            if logger:
                logger.debug(
                    f"extraction gate passed ({extraction_result.elapsed_ms:.2f}ms)",
                    extra={"stage": "extract"},
                )

        except ConfigError:
            raise

        with libarchive.file_reader(str(archive_path)) as archive:
            for entry in archive:
                # Collect entry metadata
                orig_pathname = entry.pathname
                is_dir = entry.isdir
                entry_size = entry.size if entry.size is not None else 0

                # Phase 1: Validate format and filters (right after opening)
                if telemetry.format_name is None:  # First iteration
                    from .extraction_integrity import validate_archive_format

                    format_name = getattr(archive, "format_name", None)
                    filters = getattr(archive, "filters", None) or []

                    # Convert bytes to string if necessary
                    if isinstance(format_name, bytes):
                        format_name = format_name.decode("utf-8", errors="replace")
                    if isinstance(filters, (list, tuple)):
                        filters = [
                            f.decode("utf-8", errors="replace") if isinstance(f, bytes) else f
                            for f in filters
                        ]

                    telemetry.format_name = format_name
                    telemetry.filters = filters if isinstance(filters, list) else []

                    is_valid, error_msg = validate_archive_format(format_name, filters, policy)
                    if not is_valid:
                        raise ConfigError(
                            error_message(
                                ExtractionErrorCode.FORMAT_NOT_ALLOWED,
                                f"Archive format/filter validation failed: {error_msg}",
                            )
                        )

                # Phase 2: Validate against all security policies (via PreScanValidator)
                # This includes:
                # - Entry type checking (symlinks, hardlinks, devices, FIFOs, sockets)
                # - Path normalization and constraints (depth, length, unicode)
                # - Case-fold collision detection
                # - Entry count budget
                # - Per-file size limits
                # - Per-entry compression ratio

                # Get compressed size if available (ZIP archives expose this)
                entry_compressed_size = None
                try:
                    # For ZIP and some other formats, libarchive may provide compressed size
                    if hasattr(entry, "compressed_size") and entry.compressed_size:
                        entry_compressed_size = entry.compressed_size
                except (AttributeError, TypeError):
                    pass  # Fallback: compressed_size not available (e.g., TAR formats)

                prescan_validator.validate_entry(
                    original_path=orig_pathname,
                    is_dir=is_dir,
                    is_symlink=entry.issym,
                    is_hardlink=entry.islnk,
                    is_fifo=entry.isfifo,
                    is_block_dev=entry.isblk,
                    is_char_dev=entry.ischr,
                    is_socket=entry.issock,
                    uncompressed_size=entry_size if entry_size > 0 else None,
                    compressed_size=entry_compressed_size,  # Now passing actual compressed size (ZIP-only)
                )

                # Validate and normalize the path
                try:
                    validated_path = _validate_member_path(orig_pathname)
                except ConfigError as e:
                    raise ConfigError(f"Path validation failed for archive entry: {e}") from e

                # Check containment: ensure path stays within extraction root
                try:
                    target = extract_root / validated_path
                    target.resolve().relative_to(extract_root.resolve())
                except ValueError:
                    raise ConfigError(
                        error_message(
                            ExtractionErrorCode.TRAVERSAL,
                            f"Path escapes extraction root: {orig_pathname}",
                        )
                    )

                # Accumulate size for bomb check (directories typically have size 0)
                if not is_dir:
                    total_uncompressed += entry_size
                    metrics.total_bytes += entry_size

                entries_to_extract.append((orig_pathname, validated_path, is_dir))
                metrics.total_entries += 1

        metrics.entries_allowed = len(entries_to_extract)
        telemetry.entries_total = metrics.total_entries
        telemetry.entries_allowed = metrics.entries_allowed
        telemetry.bytes_declared = total_uncompressed

        # Apply deterministic ordering if configured (Gap 2: deterministic_order setting)
        if policy.deterministic_order == "path_asc":
            # Sort entries lexicographically by normalized path for reproducibility
            entries_to_extract.sort(key=lambda x: str(x[1]))  # x[1] is validated_path

        # Phase 3: Verify disk space before extraction (Phase 3-4)
        guardian.verify_space_available(total_uncompressed, extract_root)

        # Phase 4: Extract (only if pre-scan passed)
        extracted_files: List[Path] = []
        extracted_dirs: List[Path] = []
        entries_metadata: List[tuple[str, Path, int, Optional[str]]] = []  # For audit manifest

        with libarchive.file_reader(str(archive_path)) as archive:
            for entry, (orig_pathname, validated_path, is_dir) in zip(
                archive, entries_to_extract, strict=False
            ):
                target_path = extract_root / validated_path

                if is_dir:
                    target_path.mkdir(parents=True, exist_ok=True)
                    extracted_dirs.append(target_path)
                else:
                    # Ensure parent directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Stream the file content to the target using atomic write discipline
                    # (temp file → write → fsync → rename → mtime → periodic dirfsync)
                    temp_path = target_path.with_suffix(
                        f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    )
                    file_hasher = hashlib.sha256() if policy.hash_enable else None

                    try:
                        with temp_path.open("wb") as temp_file:
                            bytes_written = 0
                            for block in entry.get_blocks():
                                temp_file.write(block)
                                bytes_written += len(block)
                                if file_hasher:
                                    file_hasher.update(block)

                            # Sync file to disk before rename
                            if policy.atomic and hasattr(temp_file, "fileno"):
                                try:
                                    os.fsync(temp_file.fileno())
                                except (OSError, AttributeError):
                                    pass  # fsync not supported on this platform

                            telemetry.bytes_written += bytes_written

                        # Atomic rename
                        temp_path.replace(target_path)

                        # Set mtime per policy
                        if policy.timestamp_mode == "preserve" and hasattr(entry, "mtime"):
                            try:
                                if entry.mtime is not None:
                                    os.utime(str(target_path), (entry.mtime, entry.mtime))
                            except (OSError, TypeError):
                                pass  # mtime not available or not supported

                        # Periodic directory fsync for durability
                        if policy.atomic and len(extracted_files) % policy.group_fsync == 0:
                            try:
                                parent_fd = os.open(str(target_path.parent), os.O_DIRECTORY)
                                try:
                                    os.fsync(parent_fd)
                                finally:
                                    os.close(parent_fd)
                            except (OSError, AttributeError):
                                pass  # fsync not supported

                    except Exception:
                        # Clean up temp file on error
                        if temp_path.exists():
                            temp_path.unlink()
                        raise

                    # Record metadata for audit manifest
                    file_hash = file_hasher.hexdigest() if file_hasher else None
                    entries_metadata.append((orig_pathname, target_path, bytes_written, file_hash))
                    extracted_files.append(target_path)

        metrics.entries_extracted = len(extracted_files)
        metrics.finalize()
        telemetry.duration_ms = metrics.duration_ms

        # Phase 4: Apply default permissions and finalize (Phase 3-4)
        guardian.finalize_extraction(
            extracted_files=extracted_files,
            extracted_dirs=extracted_dirs,
        )

        # Write audit JSON manifest (after successful extraction)
        if policy.manifest_emit:
            _write_audit_manifest(
                extract_root=extract_root,
                archive_path=archive_path,
                policy=policy,
                entries_metadata=entries_metadata,
                telemetry=telemetry,
                metrics=metrics,
            )

        if logger:
            logger.info(
                "extracted archive",
                extra={
                    "stage": "extract",
                    "archive": str(archive_path),
                    "files": len(extracted_files),
                    "encapsulated_root": str(extract_root) if policy.encapsulate else None,
                    "duration_ms": round(telemetry.duration_ms, 2),
                },
            )

        return extracted_files

    except libarchive.ArchiveError as exc:
        raise ConfigError(f"Failed to extract archive {archive_path}: {exc}") from exc
    except ConfigError:
        metrics.finalize()
        raise


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
        shutil.copy2(source, temp_path)
        try:
            with temp_path.open("rb") as temp_file:
                try:
                    os.fsync(temp_file.fileno())
                except OSError:
                    pass
        except OSError:
            pass
        os.replace(temp_path, destination)
        return destination, source
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
