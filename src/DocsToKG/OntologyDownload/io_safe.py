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
#     }
#   ]
# }
# === /NAVMAP ===

"""Filesystem and payload safety utilities for the ontology downloader."""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Dict


__all__ = [
    "sanitize_filename",
    "generate_correlation_id",
    "mask_sensitive_data",
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
    masked: Dict[str, object] = {}
    for key, value in payload.items():
        lower = key.lower()
        if lower in sensitive_keys:
            masked[key] = "***masked***"
        elif isinstance(value, str) and "apikey" in value.lower():
            masked[key] = "***masked***"
        else:
            masked[key] = value
    return masked

