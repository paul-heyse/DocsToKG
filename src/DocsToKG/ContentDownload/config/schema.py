"""
JSON Schema Export for ContentDownload Configuration

Provides JSON Schema generation for configuration validation and tooling.
Schemas can be exported for IDE/editor support, documentation, and CI/CD validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import ContentDownloadConfig


def get_config_schema() -> Dict[str, Any]:
    """
    Generate JSON Schema for ContentDownloadConfig.

    Returns the Pydantic v2 JSON Schema which includes:
    - All fields with descriptions
    - Default values
    - Type constraints and validators
    - Required fields

    Returns:
        Dictionary representing the JSON Schema (can be serialized to JSON)

    Example:
        >>> schema = get_config_schema()
        >>> print(json.dumps(schema, indent=2))
    """
    return ContentDownloadConfig.model_json_schema()


def export_config_schema(output_path: str | Path) -> None:
    """
    Export configuration schema to a JSON file.

    Useful for:
    - IDE/editor autocomplete setup
    - JSON Schema validation tools
    - Documentation generation
    - CI/CD configuration linting

    Args:
        output_path: File path to write schema JSON to

    Example:
        >>> export_config_schema("config-schema.json")
    """
    schema = get_config_schema()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def export_resolver_defaults() -> Dict[str, Any]:
    """
    Export default resolver configuration structure.

    Useful for generating example configs for users.

    Returns:
        Dictionary with all resolver configurations
    """
    config = ContentDownloadConfig()
    return config.model_dump(mode="json")["resolvers"]


if __name__ == "__main__":
    # Export schema to file when run as script
    export_config_schema("config-schema.json")
    print("✅ Schema exported to config-schema.json")
