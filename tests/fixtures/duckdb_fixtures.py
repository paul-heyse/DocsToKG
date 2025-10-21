# === NAVMAP v1 ===
# {
#   "module": "tests.fixtures.duckdb_fixtures",
#   "purpose": "DuckDB catalog fixtures for database testing",
#   "sections": [
#     {"id": "ephemeral-db-fixture", "name": "ephemeral_duckdb", "anchor": "fixture-ephemeral-duckdb", "kind": "fixture"},
#     {"id": "test-data-fixture", "name": "duckdb_with_test_data", "anchor": "fixture-duckdb-with-test-data", "kind": "fixture"},
#     {"id": "migration-fixture", "name": "duckdb_migrations", "anchor": "fixture-duckdb-migrations", "kind": "fixture"}
#   ]
# }
# === /NAVMAP ===

"""
DuckDB catalog fixtures for testing database operations.

Provides ephemeral in-memory databases with automatic migration and test data loading.
All databases are isolated per test and automatically cleaned up.
"""

from __future__ import annotations

from typing import Any, Generator

import pytest


@pytest.fixture
def ephemeral_duckdb() -> Generator[dict[str, Any], None, None]:
    """
    Provide an ephemeral in-memory DuckDB connection.

    Yields a dict with:
    - conn: DuckDB connection object
    - cursor: Connection cursor
    - query: helper to execute and fetch
    - close: cleanup function

    Example:
        def test_database(ephemeral_duckdb):
            db = ephemeral_duckdb
            db['cursor'].execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
            db['cursor'].execute("INSERT INTO test VALUES (1, 'Alice')")
            result = db['query']("SELECT * FROM test WHERE id = 1")
            assert result[0] == (1, 'Alice')
    """
    try:
        import duckdb
    except ImportError:
        pytest.skip("DuckDB not installed")

    # Create in-memory connection
    conn = duckdb.connect(":memory:")
    cursor = conn.cursor()

    def query(sql: str) -> list[tuple[Any, ...]]:
        """Execute query and fetch all results."""
        cursor.execute(sql)
        return cursor.fetchall()

    def close() -> None:
        """Close connection."""
        try:
            conn.close()
        except Exception:
            pass

    yield {
        "conn": conn,
        "cursor": cursor,
        "query": query,
        "close": close,
    }

    # Cleanup
    close()


@pytest.fixture
def duckdb_with_test_data(ephemeral_duckdb) -> Generator[dict[str, Any], None, None]:
    """
    Provide a DuckDB connection with standard test schema and sample data.

    Yields:
        dict: Same as ephemeral_duckdb but with:
            - ontologies table (id, name, version, status)
            - versions table (version, timestamp, count)
            - artifacts table (artifact_id, size, path)

    Example:
        def test_with_data(duckdb_with_test_data):
            db = duckdb_with_test_data
            results = db['query']("SELECT COUNT(*) FROM ontologies")
            assert results[0][0] > 0
    """
    db = ephemeral_duckdb
    cursor = db["cursor"]

    # Create standard schema
    cursor.execute(
        """
        CREATE TABLE ontologies (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            status VARCHAR DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE versions (
            version VARCHAR PRIMARY KEY,
            ontology_id INTEGER,
            file_count INTEGER DEFAULT 0,
            total_size_bytes INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE artifacts (
            artifact_id VARCHAR PRIMARY KEY,
            version VARCHAR,
            file_path VARCHAR NOT NULL,
            file_size_bytes INTEGER DEFAULT 0,
            checksum_sha256 VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Insert sample test data
    cursor.execute(
        """
        INSERT INTO ontologies (id, name, version, status)
        VALUES
            (1, 'hp', '2024-01-01', 'active'),
            (2, 'chebi', '2024-01-01', 'active'),
            (3, 'test-ont', '2024-01-01', 'testing')
    """
    )

    cursor.execute(
        """
        INSERT INTO versions (version, ontology_id, file_count, total_size_bytes)
        VALUES
            ('hp-2024-01-01', 1, 5, 1048576),
            ('chebi-2024-01-01', 2, 3, 524288),
            ('test-2024-01-01', 3, 1, 102400)
    """
    )

    cursor.execute(
        """
        INSERT INTO artifacts (artifact_id, version, file_path, file_size_bytes, checksum_sha256)
        VALUES
            ('hp-1', 'hp-2024-01-01', '/data/hp/hp.owl', 524288, 'abc123'),
            ('hp-2', 'hp-2024-01-01', '/data/hp/hp.json', 524288, 'def456'),
            ('chebi-1', 'chebi-2024-01-01', '/data/chebi/chebi.owl', 524288, 'ghi789'),
            ('test-1', 'test-2024-01-01', '/data/test/test.ttl', 102400, 'jkl012')
    """
    )

    yield db


@pytest.fixture
def duckdb_migrations() -> Generator[dict[str, Any], None, None]:
    """
    Provide DuckDB schema migration helpers.

    Yields a dict with:
    - schemas: dict of schema versions
    - apply: function to apply schema by version
    - current_version: current schema version

    Example:
        def test_migrations(duckdb_migrations, ephemeral_duckdb):
            db = ephemeral_duckdb
            mig = duckdb_migrations
            mig['apply'](db['cursor'], 'v1')
            result = db['query']("SELECT table_name FROM information_schema.tables")
            assert len(result) > 0
    """
    schemas = {
        "v1": """
            CREATE TABLE IF NOT EXISTS ontologies (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS versions (
                version VARCHAR PRIMARY KEY,
                ontology_id INTEGER,
                file_count INTEGER
            );
        """,
        "v2": """
            CREATE TABLE IF NOT EXISTS ontologies (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS versions (
                version VARCHAR PRIMARY KEY,
                ontology_id INTEGER,
                file_count INTEGER,
                total_size_bytes INTEGER DEFAULT 0
            );
        """,
        "v3": """
            CREATE TABLE IF NOT EXISTS ontologies (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS versions (
                version VARCHAR PRIMARY KEY,
                ontology_id INTEGER,
                file_count INTEGER,
                total_size_bytes INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id VARCHAR PRIMARY KEY,
                version VARCHAR,
                file_path VARCHAR,
                file_size_bytes INTEGER,
                checksum_sha256 VARCHAR
            );
        """,
    }

    def apply(cursor: Any, version: str) -> None:
        """Apply schema for given version."""
        if version not in schemas:
            raise ValueError(f"Unknown schema version: {version}")
        sql = schemas[version]
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)

    yield {
        "schemas": schemas,
        "apply": apply,
        "current_version": "v3",
    }
