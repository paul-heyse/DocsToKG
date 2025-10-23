# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.settings_schema",
#   "purpose": "JSON Schema generation and validation for OntologyDownload settings.",
#   "sections": [
#     {
#       "id": "canonicaljsonschema",
#       "name": "CanonicalJsonSchema",
#       "anchor": "class-canonicaljsonschema",
#       "kind": "class"
#     },
#     {
#       "id": "generate-settings-schema",
#       "name": "generate_settings_schema",
#       "anchor": "function-generate-settings-schema",
#       "kind": "function"
#     },
#     {
#       "id": "generate-submodel-schemas",
#       "name": "generate_submodel_schemas",
#       "anchor": "function-generate-submodel-schemas",
#       "kind": "function"
#     },
#     {
#       "id": "write-schemas-to-disk",
#       "name": "write_schemas_to_disk",
#       "anchor": "function-write-schemas-to-disk",
#       "kind": "function"
#     },
#     {
#       "id": "validate-config-file",
#       "name": "validate_config_file",
#       "anchor": "function-validate-config-file",
#       "kind": "function"
#     },
#     {
#       "id": "get-schema-summary",
#       "name": "get_schema_summary",
#       "anchor": "function-get-schema-summary",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""JSON Schema generation and validation for OntologyDownload settings.

This module provides utilities to generate stable, canonical JSON schemas
for settings models using Pydantic v2's schema generation with custom
processing for reproducibility and drift detection.

Features:
- ✅ Deterministic schema generation (sorted keys, consistent formatting)
- ✅ Top-level and per-submodel schemas
- ✅ Automatic schema file writing to docs/schemas/
- ✅ Configuration file validation against generated schemas
- ✅ CI-ready drift detection support
"""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode


class CanonicalJsonSchema(GenerateJsonSchema):
    """Custom schema generator for reproducible JSON Schema output.

    This generator ensures that JSON schemas are generated with:
    - Sorted keys (for deterministic output)
    - Consistent formatting (2-space indentation)
    - Stable ordering of definitions and properties

    This makes schemas suitable for:
    - Version control (can detect meaningful changes)
    - CI drift detection (byte-for-byte comparison)
    - Documentation generation (consistent output)
    """

    def generate(self, schema: Any, mode: JsonSchemaMode = "validation") -> dict[str, Any]:
        """Generate schema with reproducible key ordering.

        Args:
            schema: Pydantic schema to convert to JSON Schema
            mode: JSON Schema mode (validation or serialization)

        Returns:
            Dictionary with sorted keys for deterministic output
        """
        result = super().generate(schema, mode)
        return self._sort_dict(result)

    @staticmethod
    def _sort_dict(obj: Any) -> Any:
        """Recursively sort dictionary keys for deterministic output.

        Args:
            obj: Object to sort (dict, list, or scalar)

        Returns:
            Sorted object with same type as input
        """
        if isinstance(obj, dict):
            return {k: CanonicalJsonSchema._sort_dict(obj[k]) for k in sorted(obj.keys())}
        elif isinstance(obj, list):
            return [CanonicalJsonSchema._sort_dict(item) for item in obj]
        return obj


def generate_settings_schema() -> dict[str, Any]:
    """Generate top-level OntologyDownloadSettings JSON Schema.

    Returns:
        Canonical JSON Schema for OntologyDownloadSettings with sorted keys

    Example:
        >>> schema = generate_settings_schema()
        >>> print(schema['title'])
        'OntologyDownloadSettings'
        >>> print(len(schema['properties']))
        10  # 10 domain models
    """
    from .settings import OntologyDownloadSettings

    adapter = TypeAdapter(OntologyDownloadSettings)
    schema = adapter.json_schema(schema_generator=CanonicalJsonSchema)

    return schema


def generate_submodel_schemas() -> dict[str, dict[str, Any]]:
    """Generate JSON schemas for each domain model.

    Each domain model gets its own schema file for:
    - Clear documentation of each settings domain
    - Granular validation of configuration sections
    - Easier troubleshooting of validation errors

    Returns:
        Dictionary mapping domain names to their JSON schemas

    Example:
        >>> schemas = generate_submodel_schemas()
        >>> schemas['http']['properties'].keys()
        dict_keys(['timeout_connect', 'timeout_read', ...])
    """
    from .settings import (
        CacheSettings,
        DuckDBSettings,
        ExtractionSettings,
        HttpSettings,
        LoggingSettings,
        RateLimitSettings,
        RetrySettings,
        SecuritySettings,
        StorageSettings,
        TelemetrySettings,
    )

    submodels = {
        "http": HttpSettings,
        "cache": CacheSettings,
        "retry": RetrySettings,
        "logging": LoggingSettings,
        "telemetry": TelemetrySettings,
        "security": SecuritySettings,
        "ratelimit": RateLimitSettings,
        "extraction": ExtractionSettings,
        "storage": StorageSettings,
        "duckdb": DuckDBSettings,
    }

    result: dict[str, dict[str, Any]] = {}
    for name, model_class in submodels.items():
        adapter = TypeAdapter(model_class)
        result[name] = adapter.json_schema(schema_generator=CanonicalJsonSchema)

    return result


def write_schemas_to_disk(output_dir: Path | None = None) -> tuple[Path, int]:
    """Write all schemas to docs/schemas/ directory.

    Generates and writes:
    - settings.schema.json (top-level)
    - settings.{domain}.subschema.json (one per domain model)

    Args:
        output_dir: Directory to write schemas (default: docs/schemas/)

    Returns:
        Tuple of (output_directory_path, number_of_files_written)

    Example:
        >>> path, count = write_schemas_to_disk()
        >>> print(f"Wrote {count} schema files to {path}")
        Wrote 11 schema files to /home/user/DocsToKG/docs/schemas

    Raises:
        OSError: If unable to create output directory or write files
    """
    if output_dir is None:
        # Resolve from this module's location
        this_module_dir = Path(__file__).parent
        output_dir = this_module_dir.parent.parent.parent / "docs" / "schemas"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0

    # Write top-level schema
    top_schema = generate_settings_schema()
    top_schema_path = output_dir / "settings.schema.json"
    top_schema_path.write_text(json.dumps(top_schema, indent=2, sort_keys=True) + "\n")
    files_written += 1

    # Write submodel schemas
    sub_schemas = generate_submodel_schemas()
    for name, schema in sub_schemas.items():
        schema_path = output_dir / f"settings.{name}.subschema.json"
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        files_written += 1

    return output_dir, files_written


def validate_config_file(config_path: Path) -> tuple[bool, list[str]]:
    """Validate external configuration file against generated schema.

    Loads a config file (YAML/JSON) and validates it against the
    OntologyDownloadSettings schema. Returns detailed validation errors
    with JSON pointer paths for easy troubleshooting.

    Args:
        config_path: Path to config file (YAML, JSON, or TOML)

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if validation passed
        - error_messages: List of validation error messages (empty if valid)

    Example:
        >>> valid, errors = validate_config_file(Path("config.yaml"))
        >>> if not valid:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """
    try:
        import yaml
    except ImportError:
        return False, ["PyYAML required for config file validation"]

    try:
        import jsonschema
    except ImportError:
        return False, ["jsonschema library required for validation"]

    # Load config file
    try:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            config_data = yaml.safe_load(config_path.read_text())
        elif config_path.suffix.lower() == ".json":
            config_data = json.loads(config_path.read_text())
        else:
            return False, [f"Unsupported file type: {config_path.suffix}"]
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        return False, [f"Failed to parse config file: {e}"]
    except OSError as e:
        return False, [f"Failed to read config file: {e}"]

    # Validate against schema
    try:
        schema = generate_settings_schema()
        jsonschema.validate(config_data, schema)
        return True, []
    except jsonschema.ValidationError as e:
        # Format validation errors for user readability
        errors = [f"Validation failed at {e.json_path}: {e.message}"]
        return False, errors
    except Exception as e:
        return False, [f"Validation error: {e}"]


def get_schema_summary() -> dict[str, Any]:
    """Get summary statistics about the settings schema.

    Returns information like:
    - Number of top-level fields (domain models)
    - Total number of properties across all models
    - Enum values and their counts
    - Required vs optional fields

    Returns:
        Dictionary with schema statistics

    Example:
        >>> summary = get_schema_summary()
        >>> print(f"Total models: {summary['total_models']}")
        Total models: 10
    """
    schema = generate_settings_schema()
    submodels = generate_submodel_schemas()

    # Count properties across all models
    total_properties = sum(len(m.get("properties", {})) for m in submodels.values())

    # Count required fields
    required_count = sum(len(m.get("required", [])) for m in submodels.values())

    return {
        "total_models": len(submodels),
        "total_properties": total_properties,
        "required_properties": required_count,
        "optional_properties": total_properties - required_count,
        "models": list(submodels.keys()),
    }


__all__ = [
    "CanonicalJsonSchema",
    "generate_settings_schema",
    "generate_submodel_schemas",
    "write_schemas_to_disk",
    "validate_config_file",
    "get_schema_summary",
]
