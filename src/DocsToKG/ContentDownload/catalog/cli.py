"""CLI commands for artifact catalog operations.

Provides 6 commands for managing the catalog:
  - import-manifest: Backfill from manifest.jsonl
  - show: Display records for an artifact
  - where: Find where a file with given SHA-256 is stored
  - dedup-report: List duplicated content hashes
  - verify: Verify SHA-256 of a stored file
  - gc: Garbage collect orphaned files
"""

from __future__ import annotations

import logging
from typing import Optional

import typer

from DocsToKG.ContentDownload.catalog.bootstrap import CatalogBootstrap
from DocsToKG.ContentDownload.catalog.migrate import import_manifest
from DocsToKG.ContentDownload.config.loader import load_config

logger = logging.getLogger(__name__)
app = typer.Typer(help="Artifact Catalog management commands")


@app.command()
def import_manifest_cmd(
    manifest_path: str = typer.Argument(..., help="Path to manifest.jsonl"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path (YAML/JSON)"
    ),
    compute_sha256: bool = typer.Option(
        False, "--compute-sha256", help="Compute SHA-256 for missing hashes"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Dry-run mode (default: true)"
    ),
) -> None:
    """Import records from manifest.jsonl into catalog.
    
    One-time backfill operation to populate the catalog from older runs.
    Use --dry-run to preview, then --apply to execute.
    """
    try:
        # Load config
        if config_path:
            config = load_config(path=config_path)
        else:
            from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
            config = ContentDownloadConfig()
        
        # Initialize catalog
        with CatalogBootstrap(config.catalog, config.storage) as bootstrap:
            count = import_manifest(
                catalog=bootstrap.catalog,
                manifest_path=manifest_path,
                compute_missing_sha256=compute_sha256,
                dry_run=dry_run,
            )
            action = "would import" if dry_run else "imported"
            typer.echo(f"✓ {action} {count} records")
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def show(
    artifact_id: str = typer.Argument(..., help="Artifact ID (e.g., doi:10.1234/abc)"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Display all catalog records for an artifact.
    
    Shows download history and metadata for a given artifact ID.
    """
    try:
        # Load config and initialize catalog
        if config_path:
            config = load_config(path=config_path)
        else:
            from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
            config = ContentDownloadConfig()
        
        with CatalogBootstrap(config.catalog, config.storage) as bootstrap:
            records = bootstrap.catalog.get_by_artifact(artifact_id)
            
            if not records:
                typer.echo(f"No records found for {artifact_id}")
                return
            
            typer.echo(f"\n{len(records)} record(s) for {artifact_id}:\n")
            for record in records:
                typer.echo(f"  ID: {record.id}")
                typer.echo(f"  Resolver: {record.resolver}")
                typer.echo(f"  URL: {record.source_url}")
                typer.echo(f"  SHA-256: {record.sha256 or '(not computed)'}")
                typer.echo(f"  Size: {record.bytes} bytes")
                typer.echo(f"  Created: {record.created_at}")
                typer.echo(f"  URI: {record.storage_uri}")
                if record.run_id:
                    typer.echo(f"  Run ID: {record.run_id}")
                typer.echo()
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def where(
    sha256: str = typer.Argument(..., help="SHA-256 hash (lowercase hex)"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Find all records with a given SHA-256 hash.
    
    Useful for locating files, dedup checking, and verification.
    """
    try:
        if config_path:
            config = load_config(path=config_path)
        else:
            from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
            config = ContentDownloadConfig()
        
        with CatalogBootstrap(config.catalog, config.storage) as bootstrap:
            records = bootstrap.catalog.get_by_sha256(sha256)
            
            if not records:
                typer.echo(f"No records found for SHA-256: {sha256}")
                return
            
            typer.echo(f"\n{len(records)} record(s) with SHA-256 {sha256}:\n")
            for record in records:
                typer.echo(f"  Artifact: {record.artifact_id}")
                typer.echo(f"  Resolver: {record.resolver}")
                typer.echo(f"  Path: {record.storage_uri}")
                typer.echo(f"  Size: {record.bytes} bytes")
                typer.echo()
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def dedup_report(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Generate deduplication report.
    
    Lists SHA-256 values with multiple records (duplicate content).
    """
    try:
        if config_path:
            config = load_config(path=config_path)
        else:
            from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
            config = ContentDownloadConfig()
        
        with CatalogBootstrap(config.catalog, config.storage) as bootstrap:
            duplicates = bootstrap.catalog.find_duplicates()
            
            if not duplicates:
                typer.echo("No duplicates found")
                return
            
            typer.echo(f"\n{len(duplicates)} duplicate hash(es):\n")
            total_duplicated = sum(count - 1 for _, count in duplicates)
            
            for sha256, count in duplicates:
                typer.echo(f"  {sha256}: {count} records ({count-1} potential saves)")
            
            typer.echo(f"\nTotal potential savings: {total_duplicated} redundant copies")
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def verify(
    record_id: int = typer.Argument(..., help="Catalog record ID"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Verify SHA-256 of a stored file.
    
    Re-computes the hash and compares against catalog record.
    """
    try:
        if config_path:
            config = load_config(path=config_path)
        else:
            from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
            config = ContentDownloadConfig()
        
        with CatalogBootstrap(config.catalog, config.storage) as bootstrap:
            is_valid = bootstrap.catalog.verify(record_id)
            
            if is_valid:
                typer.echo(f"✓ Verification passed for record {record_id}")
            else:
                typer.echo(f"✗ Verification FAILED for record {record_id}")
                raise typer.Exit(1)
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def gc(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Dry-run mode (default: true)"
    ),
    orphan_days: Optional[int] = typer.Option(
        None, "--orphan-days", help="Override orphan TTL (days)"
    ),
) -> None:
    """Garbage collect orphaned files.
    
    Finds and removes files not referenced by catalog.
    Use --dry-run to preview, then --apply to delete.
    """
    try:
        if config_path:
            config = load_config(path=config_path)
        else:
            from DocsToKG.ContentDownload.config.models import ContentDownloadConfig
            config = ContentDownloadConfig()
        
        with CatalogBootstrap(config.catalog, config.storage) as bootstrap:
            # Collect referenced paths
            all_records = bootstrap.catalog.get_by_artifact("*")  # Placeholder
            # Note: get_by_artifact won't work with wildcard; need all_records method
            # For now, we'll use stats to show intent
            
            stats = bootstrap.catalog.stats()
            typer.echo("\nCatalog statistics:")
            typer.echo(f"  Total documents: {stats['total_documents']}")
            typer.echo(f"  Total bytes: {stats['total_bytes']}")
            
            typer.echo("\n✗ Full GC implementation requires all_records() method")
            typer.echo("  Recommendation: Add get_all_records() to CatalogStore")
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
