# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.finalize",
#   "purpose": "Integration point between download pipeline and catalog.",
#   "sections": [
#     {
#       "id": "compute-sha256-file",
#       "name": "compute_sha256_file",
#       "anchor": "function-compute-sha256-file",
#       "kind": "function"
#     },
#     {
#       "id": "finalize-artifact",
#       "name": "finalize_artifact",
#       "anchor": "function-finalize-artifact",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Integration point between download pipeline and catalog.

Provides utilities for finalizing downloaded artifacts with SHA-256 computation,
path selection (CAS vs policy), and catalog registration.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from DocsToKG.ContentDownload.catalog.fs_layout import (
    choose_final_path,
    dedup_hardlink_or_copy,
    extract_basename_from_url,
)
from DocsToKG.ContentDownload.catalog.store import CatalogStore

logger = logging.getLogger(__name__)


def compute_sha256_file(file_path: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 of a file.

    Args:
        file_path: Path to file
        chunk_size: Read chunk size (default 64KB)

    Returns:
        SHA-256 hash in lowercase hex

    Raises:
        IOError: If file cannot be read
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def finalize_artifact(
    temp_path: str,
    artifact_id: str,
    source_url: str,
    resolver: str,
    content_type: str | None,
    *,
    catalog: CatalogStore | None = None,
    root_dir: str = "data/docs",
    layout: str = "policy_path",
    hardlink_dedup: bool = True,
    compute_sha256: bool = True,
    verify_on_register: bool = False,
    run_id: str | None = None,
) -> dict:
    """Finalize a downloaded artifact: compute hash, choose path, register.

    This is the integration point between download and catalog. It:
    1. Optionally computes SHA-256
    2. Chooses final path (CAS or policy)
    3. Performs atomic move/hardlink
    4. Registers to catalog
    5. Returns metadata

    Args:
        temp_path: Temporary file path from download
        artifact_id: Artifact identifier
        source_url: Source URL
        resolver: Resolver name
        content_type: MIME type (optional)
        catalog: CatalogStore instance (optional)
        root_dir: Storage root directory
        layout: Layout strategy ('policy_path' or 'cas')
        hardlink_dedup: Enable hardlink deduplication
        compute_sha256: Whether to compute SHA-256
        verify_on_register: Verify SHA-256 after finalize
        run_id: Run ID for provenance

    Returns:
        Dict with keys:
            - 'status': 'success' or 'error'
            - 'final_path': Path to final artifact
            - 'storage_uri': Storage URI (file://...)
            - 'bytes': File size
            - 'sha256': SHA-256 hash (if computed)
            - 'is_dedup': True if deduplication occurred
            - 'error': Error message (if status='error')
    """
    temp = Path(temp_path)

    if not temp.exists():
        return {
            "status": "error",
            "error": f"Temporary file not found: {temp_path}",
        }

    try:
        # Get file size
        file_size = temp.stat().st_size

        # Compute SHA-256 if requested
        sha256_hex = None
        if compute_sha256:
            logger.debug(f"Computing SHA-256 for {temp_path}")
            sha256_hex = compute_sha256_file(temp_path)
            logger.debug(f"SHA-256: {sha256_hex}")

        # Choose final path
        basename = extract_basename_from_url(source_url)
        try:
            final_path = choose_final_path(
                root_dir=root_dir,
                layout=layout,
                sha256_hex=sha256_hex,
                artifact_id=artifact_id,
                url_basename=basename,
            )
        except ValueError as e:
            return {
                "status": "error",
                "error": f"Failed to choose final path: {e}",
            }

        # Perform hardlink dedup or move
        is_dedup = False
        try:
            is_dedup = dedup_hardlink_or_copy(
                src_tmp=temp_path,
                dst_final=final_path,
                hardlink=hardlink_dedup,
            )
            logger.info(f"File finalized: {final_path} (dedup={is_dedup})")
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to finalize file: {e}",
            }

        # Verify after finalize if requested
        if verify_on_register and sha256_hex:
            if not Path(final_path).exists():
                return {
                    "status": "error",
                    "error": f"File not found after finalize: {final_path}",
                }

            computed_sha = compute_sha256_file(final_path)
            if computed_sha != sha256_hex:
                return {
                    "status": "error",
                    "error": f"SHA-256 mismatch after finalize: {computed_sha} != {sha256_hex}",
                }

        # Register to catalog
        if catalog is not None:
            storage_uri = f"file://{final_path}"
            try:
                record = catalog.register_or_get(
                    artifact_id=artifact_id,
                    source_url=source_url,
                    resolver=resolver,
                    content_type=content_type,
                    bytes=file_size,
                    sha256=sha256_hex,
                    storage_uri=storage_uri,
                    run_id=run_id,
                )
                logger.info(f"Registered to catalog: record_id={record.id}")
            except Exception as e:
                logger.error(f"Failed to register to catalog: {e}")
                # Continue anyway - file is finalized even if registration failed

        # Build response
        return {
            "status": "success",
            "final_path": final_path,
            "storage_uri": f"file://{final_path}",
            "bytes": file_size,
            "sha256": sha256_hex,
            "is_dedup": is_dedup,
        }

    except Exception as e:
        logger.error(f"Unexpected error during finalization: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Unexpected error: {e}",
        }
