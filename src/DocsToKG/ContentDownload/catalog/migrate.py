"""Migration helpers for backfilling catalog from existing manifests.

Provides utilities to import records from manifest.jsonl files created
by older runs into the new catalog system.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from DocsToKG.ContentDownload.catalog.store import CatalogStore

logger = logging.getLogger(__name__)


def parse_manifest_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single line from manifest.jsonl.
    
    Args:
        line: JSON line from manifest.jsonl
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse manifest line: {e}")
        return None


def extract_catalog_fields(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract catalog-relevant fields from manifest record.
    
    Maps legacy manifest format to catalog fields.
    
    Args:
        record: Manifest record (dict)
        
    Returns:
        Dict with catalog fields or None if validation fails
        
    Manifest record expected format:
        {
            "artifact_id": "doi:10.1234/abc",
            "source_url": "https://example.com/paper.pdf",
            "resolver": "unpaywall",
            "content_type": "application/pdf",
            "bytes": 1234567,
            "sha256": "e3b0c44298fc1c14...",
            "status": "success"
        }
    """
    # Only import successful attempts
    if record.get("status") != "success" and record.get("record_type") != "success":
        return None
    
    artifact_id = record.get("artifact_id") or record.get("work_id")
    source_url = record.get("source_url") or record.get("url")
    resolver = record.get("resolver")
    
    if not all([artifact_id, source_url, resolver]):
        logger.debug(f"Missing required fields in manifest record: {record}")
        return None
    
    # Build storage_uri from path if available, else from source_url
    storage_uri = record.get("storage_uri")
    if not storage_uri:
        if "path" in record:
            storage_uri = f"file://{record['path']}"
        else:
            # Fallback: encode metadata as URI
            storage_uri = f"file://unknown/{artifact_id}"
    
    return {
        "artifact_id": artifact_id,
        "source_url": source_url,
        "resolver": resolver,
        "content_type": record.get("content_type"),
        "bytes": record.get("bytes", 0),
        "sha256": record.get("sha256"),
        "storage_uri": storage_uri,
        "run_id": record.get("run_id"),
    }


def compute_sha256_from_file(file_path: str, chunk_size: int = 65536) -> Optional[str]:
    """Compute SHA-256 hash of a file.
    
    Args:
        file_path: Path to file
        chunk_size: Read chunk size (default 64KB)
        
    Returns:
        SHA-256 hash in lowercase hex, or None if file not found/readable
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.debug(f"File not found for hashing: {file_path}")
            return None
        
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute SHA-256 for {file_path}: {e}")
        return None


def import_manifest(
    catalog: CatalogStore,
    manifest_path: str,
    compute_missing_sha256: bool = False,
    dry_run: bool = False,
) -> int:
    """Import records from manifest.jsonl into catalog.
    
    Args:
        catalog: CatalogStore instance
        manifest_path: Path to manifest.jsonl
        compute_missing_sha256: If True, compute SHA-256 for records without it
        dry_run: If True, don't actually register records
        
    Returns:
        Number of records imported
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    imported_count = 0
    skipped_count = 0
    error_count = 0
    
    try:
        with open(manifest_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = parse_manifest_line(line)
                    if not record:
                        skipped_count += 1
                        continue
                    
                    fields = extract_catalog_fields(record)
                    if not fields:
                        skipped_count += 1
                        continue
                    
                    # Optionally compute missing SHA-256
                    if compute_missing_sha256 and not fields.get("sha256"):
                        if fields["storage_uri"].startswith("file://"):
                            path = fields["storage_uri"][7:]
                            sha = compute_sha256_from_file(path)
                            if sha:
                                fields["sha256"] = sha
                    
                    # Register to catalog
                    if dry_run:
                        logger.debug(f"[DRY-RUN] Would register: {fields}")
                    else:
                        catalog.register_or_get(**fields)
                    
                    imported_count += 1
                    
                    if imported_count % 100 == 0:
                        logger.info(f"Imported {imported_count} records...")
                
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    error_count += 1
    
    except Exception as e:
        logger.error(f"Failed to import manifest: {e}")
        raise
    
    logger.info(
        f"Import complete: {imported_count} imported, "
        f"{skipped_count} skipped, {error_count} errors"
    )
    return imported_count


def iter_manifest_records(manifest_path: str) -> Iterator[Dict[str, Any]]:
    """Iterate over successfully parsed manifest records.
    
    Args:
        manifest_path: Path to manifest.jsonl
        
    Yields:
        Parsed record dicts
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        logger.warning(f"Manifest file not found: {manifest_path}")
        return
    
    try:
        with open(manifest_file, "r") as f:
            for line in f:
                record = parse_manifest_line(line)
                if record:
                    yield record
    except Exception as e:
        logger.error(f"Error reading manifest: {e}")
