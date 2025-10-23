# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.cli_settings_commands",
#   "purpose": "CLI commands for settings management: show, schema, validate.",
#   "sections": [
#     {
#       "id": "show",
#       "name": "show",
#       "anchor": "function-show",
#       "kind": "function"
#     },
#     {
#       "id": "schema",
#       "name": "schema",
#       "anchor": "function-schema",
#       "kind": "function"
#     },
#     {
#       "id": "validate",
#       "name": "validate",
#       "anchor": "function-validate",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""CLI commands for settings management: show, schema, validate.

Provides three commands for managing OntologyDownloadSettings:
- `settings show`: Display effective configuration with sources
- `settings schema`: Generate and save JSON schemas
- `settings validate`: Validate config files against schema

Example:
    >>> from DocsToKG.OntologyDownload.cli_settings_commands import (
    ...     settings_show, settings_schema, settings_validate
    ... )
"""

import json
from pathlib import Path
from typing import Optional

import typer

from DocsToKG.OntologyDownload.settings import get_default_config
from DocsToKG.OntologyDownload.settings_schema import (
    get_schema_summary,
    validate_config_file,
    write_schemas_to_disk,
)
from DocsToKG.OntologyDownload.settings_sources import get_source_fingerprint

# Typer app for settings subcommands
settings_app = typer.Typer(
    name="settings",
    help="Manage OntologyDownloadSettings configuration",
    short_help="Settings management (show, schema, validate)",
)


@settings_app.command()
def show(
    no_redact: bool = typer.Option(
        False,
        "--no-redact-secrets",
        help="Include all values (including secrets)",
    ),
    format_output: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, or yaml",
    ),
) -> None:
    """Display effective configuration with source attribution.

    Shows all settings with their current values and sources (cli, env, config, default).
    By default, redacts sensitive values for security.

    Example:
        $ ontofetch settings show
        $ ontofetch settings show --format json
        $ ontofetch settings show --no-redact-secrets --format json
    """
    try:
        settings = get_default_config()
        fingerprint = get_source_fingerprint()

        # Build output data
        output_data = []

        for domain_name in [
            "http",
            "cache",
            "retry",
            "logging",
            "telemetry",
            "security",
            "ratelimit",
            "extraction",
            "storage",
            "db",
        ]:
            domain_model = getattr(settings, domain_name)
            domain_dict = domain_model.model_dump()

            for field_name, value in domain_dict.items():
                full_field_name = f"{domain_name}__{field_name}"
                source = fingerprint.get(full_field_name, "default")

                # Redact sensitive fields
                display_value = value
                if not no_redact and any(
                    sensitive in field_name.lower()
                    for sensitive in ["password", "token", "key", "secret", "auth"]
                ):
                    display_value = "***REDACTED***"

                output_data.append(
                    {
                        "field": full_field_name,
                        "value": display_value,
                        "source": source,
                        "type": type(value).__name__,
                    }
                )

        # Format output
        if format_output == "json":
            typer.echo(json.dumps(output_data, indent=2, default=str))
        elif format_output == "yaml":
            try:
                import yaml

                typer.echo(yaml.dump(output_data, default_flow_style=False))
            except ImportError:
                typer.echo(
                    "YAML format requires PyYAML. Use --format json instead.",
                    err=True,
                )
                raise typer.Exit(2)
        else:  # table
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title="OntologyDownloadSettings - Effective Configuration")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")
                table.add_column("Source", style="yellow")
                table.add_column("Type", style="magenta")

                for item in output_data:
                    table.add_row(
                        item["field"],
                        str(item["value"]),
                        item["source"],
                        item["type"],
                    )

                console.print(table)
            except ImportError:
                typer.echo(
                    "Table format requires Rich. Use --format json instead.",
                    err=True,
                )
                raise typer.Exit(2)

    except Exception as e:
        typer.echo(f"Error displaying settings: {e}", err=True)
        raise typer.Exit(1)


@settings_app.command()
def schema(
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output directory for schema files (default: docs/schemas/)",
    ),
    format_output: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json (only supported)",
    ),
) -> None:
    """Generate JSON schemas for settings.

    Creates:
    - settings.schema.json (top-level OntologyDownloadSettings)
    - settings.{domain}.subschema.json (for each of 10 domain models)

    Schemas are deterministic (sorted keys) and suitable for CI drift detection.

    Example:
        $ ontofetch settings schema
        $ ontofetch settings schema --out /tmp/schemas/
        $ ontofetch settings schema --out . --format json
    """
    try:
        if out is None:
            # Use default docs/schemas/

            cwd = Path.cwd()
            out = cwd / "docs" / "schemas"
        else:
            out = out.resolve()

        # Write schemas
        schema_dir, count = write_schemas_to_disk(out)

        typer.echo(
            f"✅ Generated {count} schema files in {schema_dir}",
            err=False,
        )

        # Show summary
        summary = get_schema_summary()
        typer.echo(
            "   • Top-level schema: settings.schema.json",
            err=False,
        )
        typer.echo(
            f"   • Submodels: {summary['total_models']}",
            err=False,
        )
        typer.echo(
            f"   • Total properties: {summary['total_properties']}",
            err=False,
        )

    except Exception as e:
        typer.echo(f"Error generating schemas: {e}", err=True)
        raise typer.Exit(1)


@settings_app.command()
def validate(
    file: Path = typer.Argument(
        ...,
        help="Configuration file to validate (YAML or JSON)",
    ),
    format_output: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json",
    ),
) -> None:
    """Validate configuration file against schema.

    Loads YAML or JSON config file and validates it against
    the OntologyDownloadSettings JSON schema. Provides detailed
    error messages if validation fails.

    Example:
        $ ontofetch settings validate config.yaml
        $ ontofetch settings validate /etc/ontofetch/settings.json --format json
    """
    try:
        config_path = file.resolve()

        if not config_path.exists():
            typer.echo(f"Error: File not found: {config_path}", err=True)
            raise typer.Exit(3)

        # Validate
        is_valid, errors = validate_config_file(config_path)

        if is_valid:
            if format_output == "json":
                typer.echo(
                    json.dumps(
                        {"valid": True, "file": str(config_path), "errors": []},
                        indent=2,
                    )
                )
            else:  # text
                typer.echo(f"✅ Valid: {config_path}")
            raise typer.Exit(0)
        else:
            if format_output == "json":
                typer.echo(
                    json.dumps(
                        {
                            "valid": False,
                            "file": str(config_path),
                            "errors": errors,
                        },
                        indent=2,
                    )
                )
            else:  # text
                typer.echo(f"❌ Validation failed: {config_path}", err=True)
                for error in errors:
                    typer.echo(f"   • {error}", err=True)
            raise typer.Exit(2)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error validating config: {e}", err=True)
        raise typer.Exit(1)


__all__ = [
    "settings_app",
    "show",
    "schema",
    "validate",
]
