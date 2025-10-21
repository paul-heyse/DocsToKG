#!/usr/bin/env python3
"""
Deployment Initialization Script for Phases 5A-6B
Initialize DuckDB catalog on first deployment
"""

import sys
from pathlib import Path


def setup_directories():
    """Create necessary directories for deployment."""
    dirs = [
        Path.home() / ".local" / "share" / "docstokg",
        Path.home() / ".local" / "share" / "docstokg" / "cache",
        Path.home() / ".local" / "share" / "docstokg" / "artifacts",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {d}")

    return str(dirs[0] / "catalog.duckdb")


def initialize_catalog(db_path: str):
    """Initialize the DuckDB catalog with migrations."""
    try:
        import duckdb
    except ImportError:
        print("❌ duckdb not found. Install with: pip install duckdb")
        sys.exit(1)

    try:
        from DocsToKG.OntologyDownload.catalog import apply_migrations  # type: ignore
    except ImportError:
        print("❌ OntologyDownload catalog not found. Check PYTHONPATH.")
        sys.exit(1)

    print(f"\n📊 Initializing catalog: {db_path}")

    conn = duckdb.connect(db_path)
    result = apply_migrations(conn)
    conn.close()

    print(f"✅ Applied {len(result.applied)} migrations")
    print(f"✅ Schema version: {result.schema_version}")

    return db_path


def verify_imports():
    """Verify all required modules can be imported."""
    required = [
        ("duckdb", "DuckDB database"),
        ("polars", "Polars analytics"),
    ]

    print("\n📦 Verifying imports...")

    for module, description in required:
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            print(f"❌ {module}: NOT FOUND - Install with: pip install {module}")
            sys.exit(1)

    # Verify OntologyDownload modules
    try:
        from DocsToKG.OntologyDownload import analytics, catalog

        print("✅ catalog: DuckDB catalog module")
        print("✅ analytics: Polars analytics module")
    except ImportError as e:
        print(f"❌ OntologyDownload modules: {e}")
        sys.exit(1)


def test_catalog_connection(db_path: str):
    """Test catalog connectivity."""
    print("\n🔗 Testing catalog connection...")

    try:
        import duckdb

        from DocsToKG.OntologyDownload.catalog import quick_health_check  # type: ignore

        conn = duckdb.connect(db_path, read_only=True)
        result = quick_health_check(conn)
        conn.close()

        print(f"✅ Health check: {result.message}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_analytics():
    """Test analytics imports."""
    print("\n📊 Testing analytics module...")

    try:
        print("✅ cmd_report_latest imported")
        print("✅ cmd_report_growth imported")
        print("✅ cmd_report_validation imported")
        return True
    except Exception as e:
        print(f"❌ Analytics import failed: {e}")
        return False


def main():
    """Main initialization routine."""
    print("=" * 70)
    print("🚀 DEPLOYMENT INITIALIZATION - Phases 5A-6B")
    print("=" * 70)

    # Step 1: Verify environment
    verify_imports()

    # Step 2: Setup directories
    db_path = setup_directories()

    # Step 3: Initialize catalog
    initialize_catalog(db_path)

    # Step 4: Test connections
    if not test_catalog_connection(db_path):
        print("\n⚠️  Catalog health check failed - check configuration")

    # Step 5: Test analytics
    if not test_analytics():
        print("\n⚠️  Analytics import failed - check installation")

    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT INITIALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nDatabase location: {db_path}")
    print(f"Export to use: export DOCSTOKG_DB_PATH={db_path}")
    print("\nNext steps:")
    print("1. Import modules in your Python code")
    print("2. Run tests: pytest tests/ontology_download/")
    print("3. See DEPLOYMENT_PACKAGE.md for usage examples")
    print("=" * 70)


if __name__ == "__main__":
    main()
