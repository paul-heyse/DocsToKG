"""Schema migration helpers for ontology download metadata.

Manifests evolve as new provenance fields are introduced.  This module keeps
backward-compatibility by upgrading older payloads to the latest schema version
so that downstream tooling can rely on a consistent structure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

__all__ = ["migrate_manifest"]

LOGGER = logging.getLogger(__name__)


def migrate_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a migrated manifest payload compatible with the latest schema.

    Args:
        payload: Arbitrary mapping loaded from an existing manifest file.

    Returns:
        Dictionary upgraded to the current schema version.
    """

    upgraded: Dict[str, Any] = dict(payload)
    version = str(upgraded.get("schema_version", "") or "")

    if version in {"", "1.0"}:
        upgraded.setdefault("schema_version", "1.0")
        return upgraded

    if version == "0.9":
        upgraded["schema_version"] = "1.0"
        upgraded.setdefault("resolver_attempts", [])
        return upgraded

    LOGGER.warning(
        "unknown manifest schema version",
        extra={"stage": "manifest", "schema_version": version},
    )
    return upgraded
