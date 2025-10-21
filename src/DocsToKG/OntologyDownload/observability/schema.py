"""JSON Schema generation and validation for observability events.

Provides:
- Canonical JSON Schema definition matching Event dataclass
- Dynamic schema generation from Event model
- Schema validation for incoming events
- Schema persistence to disk for CI drift detection
- Schema summary for documentation
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Event JSON Schema (Canonical)
# ============================================================================

EVENT_JSON_SCHEMA_VERSION = "1.0"

# Canonical schema definition (manually maintained to stay in sync with Event)
EVENT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://docstokg.example.org/observability/event-schema-v1.0.json",
    "title": "OntologyDownload Event Schema",
    "description": "Canonical schema for structured observability events",
    "version": EVENT_JSON_SCHEMA_VERSION,
    "type": "object",
    "required": [
        "ts",
        "type",
        "level",
        "run_id",
        "config_hash",
        "service",
        "context",
        "ids",
        "payload",
    ],
    "properties": {
        "ts": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 UTC timestamp",
        },
        "type": {
            "type": "string",
            "pattern": "^[a-z_]+(\\.[a-z_]+)*$",
            "description": "Namespaced event type (e.g., 'net.request', 'extract.done')",
        },
        "level": {
            "type": "string",
            "enum": ["INFO", "WARN", "ERROR"],
            "description": "Event level/severity",
        },
        "run_id": {
            "type": "string",
            "format": "uuid",
            "description": "UUID correlating entire CLI command run",
        },
        "config_hash": {
            "type": "string",
            "description": "Hash of normalized configuration",
        },
        "service": {
            "type": "string",
            "description": "Service name (e.g., 'ols', 'bioportal')",
        },
        "context": {
            "type": "object",
            "required": ["app_version", "os_name", "python_version"],
            "properties": {
                "app_version": {"type": "string"},
                "os_name": {"type": "string"},
                "python_version": {"type": "string"},
                "libarchive_version": {"type": ["string", "null"]},
                "hostname": {"type": ["string", "null"]},
                "pid": {"type": ["integer", "null"]},
            },
            "additionalProperties": False,
            "description": "Runtime context captured at emission",
        },
        "ids": {
            "type": "object",
            "properties": {
                "version_id": {"type": ["string", "null"]},
                "artifact_id": {"type": ["string", "null"]},
                "file_id": {"type": ["string", "null"]},
                "request_id": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
            "description": "IDs for correlating events across subsystems",
        },
        "payload": {
            "type": "object",
            "description": "Event-specific payload (varies by event type)",
            "additionalProperties": True,
        },
    },
    "additionalProperties": False,
}


# ============================================================================
# Schema Generation
# ============================================================================


def generate_settings_schema() -> Dict[str, Any]:
    """Generate canonical JSON Schema for events.

    Returns:
        Dict containing the JSON Schema
    """
    return EVENT_JSON_SCHEMA.copy()


def generate_submodel_schemas() -> Dict[str, Dict[str, Any]]:
    """Generate schemas for individual event sub-models.

    Returns:
        Dict mapping model names to their schemas
    """
    return {
        "EventContext": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "EventContext",
            "type": "object",
            "required": ["app_version", "os_name", "python_version"],
            "properties": EVENT_JSON_SCHEMA["properties"]["context"]["properties"],
            "additionalProperties": False,
        },
        "EventIds": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "EventIds",
            "type": "object",
            "properties": EVENT_JSON_SCHEMA["properties"]["ids"]["properties"],
            "additionalProperties": False,
        },
        "Event": EVENT_JSON_SCHEMA,
    }


# ============================================================================
# Schema Validation
# ============================================================================


def validate_event(event_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate an event against the canonical schema.

    Args:
        event_dict: Event as dict (e.g., from to_dict())

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        import jsonschema
    except ImportError:
        logger.warning(
            "jsonschema not installed; skipping validation. Install with: pip install jsonschema"
        )
        return True, []

    try:
        jsonschema.validate(instance=event_dict, schema=EVENT_JSON_SCHEMA)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Unexpected validation error: {e}"]


# ============================================================================
# Schema Persistence
# ============================================================================


def write_schemas_to_disk(output_dir: Path) -> Dict[str, Path]:
    """Write schemas to disk for CI drift detection.

    Args:
        output_dir: Directory to write schemas to

    Returns:
        Dict mapping schema names to their file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = {
        "event-schema.json": generate_settings_schema(),
        **{f"{name}-schema.json": schema for name, schema in generate_submodel_schemas().items()},
    }

    written_files = {}

    for filename, schema in schemas.items():
        filepath = output_dir / filename

        # Make schema reproducible (sorted keys, compact JSON)
        schema_str = json.dumps(schema, sort_keys=True, separators=(",", ":"))

        try:
            with open(filepath, "w") as f:
                f.write(schema_str)
                f.write("\n")  # Trailing newline for consistency

            written_files[filename] = filepath
            logger.info(f"Wrote schema to {filepath}")

        except Exception as e:
            logger.error(f"Error writing schema {filename}: {e}")

    return written_files


# ============================================================================
# Schema Comparison (for CI Drift Detection)
# ============================================================================


def load_schema_from_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a schema from disk.

    Args:
        filepath: Path to schema file (must be valid JSON)

    Returns:
        Parsed schema or None if load failed
    """
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading schema {filepath}: {e}")
        return None


def compare_schemas(schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two schemas.

    Args:
        schema1: First schema
        schema2: Second schema

    Returns:
        Dict with comparison results:
        {
            "identical": bool,
            "differences": List[str],  # Human-readable diff lines
        }
    """
    # Normalize both schemas (sort keys)
    s1_str = json.dumps(schema1, sort_keys=True, separators=(",", ":"))
    s2_str = json.dumps(schema2, sort_keys=True, separators=(",", ":"))

    if s1_str == s2_str:
        return {"identical": True, "differences": []}

    # Compute detailed differences
    differences = []

    # Check required fields (make copies to avoid modifying originals)
    s1_required = set(schema1.get("required", []))
    s2_required = set(schema2.get("required", []))
    if s1_required != s2_required:
        added = s2_required - s1_required
        removed = s1_required - s2_required
        if added:
            differences.append(f"Added required fields: {', '.join(sorted(added))}")
        if removed:
            differences.append(f"Removed required fields: {', '.join(sorted(removed))}")

    # Check properties (make copies to avoid modifying originals)
    s1_props = set(schema1.get("properties", {}).keys())
    s2_props = set(schema2.get("properties", {}).keys())
    if s1_props != s2_props:
        added = s2_props - s1_props
        removed = s1_props - s2_props
        if added:
            differences.append(f"Added properties: {', '.join(sorted(added))}")
        if removed:
            differences.append(f"Removed properties: {', '.join(sorted(removed))}")

    # Check version
    v1 = schema1.get("version")
    v2 = schema2.get("version")
    if v1 != v2:
        differences.append(f"Version changed: {v1} → {v2}")

    return {
        "identical": False,
        "differences": differences or ["Schemas differ in content"],
    }


# ============================================================================
# Schema Summary
# ============================================================================


def get_schema_summary() -> Dict[str, Any]:
    """Get a human-readable summary of the event schema.

    Returns:
        Dict with schema summary information
    """
    schema = generate_settings_schema()

    return {
        "schema_version": schema.get("version"),
        "total_properties": len(schema.get("properties", {})),
        "required_properties": len(schema.get("required", [])),
        "required_fields": schema.get("required", []),
        "event_levels": schema.get("properties", {}).get("level", {}).get("enum", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "properties_overview": {
            name: {
                "type": prop.get("type"),
                "required": name in schema.get("required", []),
                "description": prop.get("description", ""),
            }
            for name, prop in schema.get("properties", {}).items()
        },
    }


__all__ = [
    "EVENT_JSON_SCHEMA",
    "EVENT_JSON_SCHEMA_VERSION",
    "generate_settings_schema",
    "generate_submodel_schemas",
    "validate_event",
    "write_schemas_to_disk",
    "load_schema_from_file",
    "compare_schemas",
    "get_schema_summary",
]
