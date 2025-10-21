#!/usr/bin/env python3
"""
Example Usage Patterns for Phases 5A-6B Deployment
Catalog + Analytics in Local Python Environment
"""

import os
import sys
from pathlib import Path

# Set up Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ============================================================================
# EXAMPLE 1: Initialize Catalog (First Run)
# ============================================================================


def example_1_init_catalog():
    """Initialize DuckDB catalog on first deployment."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Initialize Catalog")
    print("=" * 70)

    import duckdb

    from DocsToKG.OntologyDownload.catalog import apply_migrations

    # Use default location or set DOCSTOKG_DB_PATH
    db_path = os.getenv("DOCSTOKG_DB_PATH", "catalog.duckdb")

    print(f"Opening database: {db_path}")
    conn = duckdb.connect(db_path)

    print("Applying migrations...")
    result = apply_migrations(conn)

    print(f"✅ Applied {len(result.applied)} migrations")
    print(f"✅ Schema version: {result.schema_version}")

    conn.close()


# ============================================================================
# EXAMPLE 2: Query Catalog
# ============================================================================


def example_2_query_catalog():
    """Query versions and artifacts from catalog."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Query Catalog")
    print("=" * 70)

    import duckdb

    from DocsToKG.OntologyDownload.catalog import (
        get_artifact_stats,
        list_versions,
    )

    db_path = os.getenv("DOCSTOKG_DB_PATH", "catalog.duckdb")

    # Query versions (read-only connection)
    conn = duckdb.connect(db_path, read_only=True)

    versions = list_versions(conn)
    print(f"\nVersions in catalog: {len(versions)}")
    for v in versions:
        print(f"  - {v.version_id} (service: {v.service}, latest: {v.latest_pointer})")

    # Get statistics
    stats = get_artifact_stats(conn)
    if stats:
        print("\nArtifact Statistics:")
        print(f"  - Total count: {stats.total_count}")
        print(f"  - Total size: {stats.total_bytes} bytes")

    conn.close()


# ============================================================================
# EXAMPLE 3: Generate Reports with Analytics
# ============================================================================


def example_3_generate_reports():
    """Generate reports using Polars analytics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Generate Reports")
    print("=" * 70)

    import polars as pl

    from DocsToKG.OntologyDownload.analytics import (
        cmd_report_growth,
        cmd_report_latest,
        cmd_report_validation,
    )

    # Create sample data
    print("Creating sample files dataframe...")
    files_df = pl.DataFrame(
        {
            "file_id": ["f1", "f2", "f3", "f4", "f5"],
            "relpath": [
                "ontologies/hp.owl",
                "ontologies/mp.rdf",
                "ontologies/chebi.ttl",
                "ontologies/go.jsonld",
                "ontologies/pato.owl",
            ],
            "size": [1024 * 100, 1024 * 200, 1024 * 150, 1024 * 75, 1024 * 180],
            "format": ["owl", "rdf", "ttl", "jsonld", "owl"],
        }
    )

    # Create validations
    print("Creating sample validations dataframe...")
    validations_df = pl.DataFrame(
        {
            "validation_id": ["v1", "v2", "v3", "v4", "v5"],
            "file_id": ["f1", "f2", "f3", "f4", "f5"],
            "validator": ["rdflib", "owlready2", "rdflib", "owlready2", "rdflib"],
            "status": ["pass", "pass", "fail", "pass", "fail"],
        }
    )

    # Generate latest report
    print("\nGenerating latest report (table format)...")
    latest_report = cmd_report_latest(files_df, validations_df, "table")
    print(latest_report)

    # Generate growth report
    print("\nGenerating growth report (comparing versions)...")
    v2_files = files_df.head(3)  # Simulate different version
    growth_report = cmd_report_growth(files_df, v2_files, "v1", "v2", "table")
    print(growth_report)

    # Generate validation report
    print("\nGenerating validation report...")
    validation_report = cmd_report_validation(validations_df, "v1", "table")
    print(validation_report)


# ============================================================================
# EXAMPLE 4: Health Check and Doctor
# ============================================================================


def example_4_health_check():
    """Perform health check on catalog."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Health Check")
    print("=" * 70)

    import duckdb

    from DocsToKG.OntologyDownload.catalog import quick_health_check

    db_path = os.getenv("DOCSTOKG_DB_PATH", "catalog.duckdb")

    conn = duckdb.connect(db_path, read_only=True)

    result = quick_health_check(conn)
    print("Health Status:")
    print(f"  - Status: {result.status}")
    print(f"  - Message: {result.message}")
    print(f"  - Timestamp: {result.timestamp}")

    conn.close()


# ============================================================================
# EXAMPLE 5: Lazy Pipelines
# ============================================================================


def example_5_lazy_pipelines():
    """Use lazy Polars pipelines for efficient processing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Lazy Pipelines")
    print("=" * 70)

    import polars as pl

    from DocsToKG.OntologyDownload.analytics.pipelines import (
        build_latest_summary_pipeline,
        compute_latest_summary,
    )

    # Create sample data
    files_df = pl.DataFrame(
        {
            "file_id": [f"f{i}" for i in range(100)],
            "relpath": [f"file_{i}.owl" for i in range(100)],
            "size": [1024 * (i + 1) for i in range(100)],
            "format": ["owl", "rdf", "ttl"] * 33 + ["owl"],
        }
    )

    print(f"Created {len(files_df)} files")

    # Build lazy pipeline
    print("Building lazy pipeline...")
    pipeline = build_latest_summary_pipeline(files_df)
    print(f"Pipeline type: {type(pipeline)}")

    # Compute summary
    print("Computing summary (lazy evaluation)...")
    summary = compute_latest_summary(files_df)

    print("\nSummary Results:")
    print(f"  - Total files: {summary.total_files}")
    print(f"  - Total bytes: {summary.total_bytes}")
    print(f"  - Formats: {summary.files_by_format}")
    print(f"  - Top files: {summary.top_files[:3]}")


# ============================================================================
# EXAMPLE 6: Garbage Collection
# ============================================================================


def example_6_garbage_collection():
    """Demonstrate garbage collection and prune operations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Garbage Collection")
    print("=" * 70)

    import duckdb

    from DocsToKG.OntologyDownload.catalog.gc import (
        identify_orphaned_artifacts,
        identify_orphaned_files,
    )

    db_path = os.getenv("DOCSTOKG_DB_PATH", "catalog.duckdb")

    conn = duckdb.connect(db_path, read_only=True)

    # Identify orphans (dry-run, no deletion)
    print("Identifying orphaned artifacts...")
    orphaned_artifacts = identify_orphaned_artifacts(conn)
    print(f"Found {len(orphaned_artifacts)} orphaned artifacts")

    print("Identifying orphaned files...")
    orphaned_files = identify_orphaned_files(conn)
    print(f"Found {len(orphaned_files)} orphaned files")

    conn.close()

    if orphaned_artifacts or orphaned_files:
        print("\n⚠️  Orphaned items found. Run prune_by_retention_days() to clean up.")


# ============================================================================
# Main: Run Examples
# ============================================================================


def main():
    """Run example demonstrations."""
    print("\n" + "=" * 70)
    print("🚀 DEPLOYMENT EXAMPLES - Phases 5A-6B")
    print("=" * 70)

    examples = [
        ("Initialize Catalog", example_1_init_catalog),
        ("Query Catalog", example_2_query_catalog),
        ("Generate Reports", example_3_generate_reports),
        ("Health Check", example_4_health_check),
        ("Lazy Pipelines", example_5_lazy_pipelines),
        ("Garbage Collection", example_6_garbage_collection),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - DEPLOYMENT_PACKAGE.md (main documentation)")
    print("  - src/DocsToKG/OntologyDownload/catalog/ (catalog code)")
    print("  - src/DocsToKG/OntologyDownload/analytics/ (analytics code)")


if __name__ == "__main__":
    main()
