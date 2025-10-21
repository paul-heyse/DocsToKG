"""Database schema introspection API.

Provides methods to inspect database schema, tables, columns, and indexes
using DuckDB's information schema.

NAVMAP:
  - CatalogSchema: Main schema inspector class
  - Query Methods:
    * get_schema() - Complete schema definition
    * list_tables() - Table enumeration
    * get_table_info() - Table metadata
    * get_indexes() - Index information
"""

from __future__ import annotations

from .schema_dto import ColumnInfo, IndexInfo, SchemaInfo, TableSchema


class CatalogSchema:
    """Database schema inspector.

    Provides methods to inspect and retrieve database schema information
    including tables, columns, and indexes.

    Attributes:
        repo: Underlying Repo instance for database access
    """

    def __init__(self, repo):
        """Initialize schema inspector.

        Args:
            repo: Repo instance for database access
        """
        self.repo = repo

    def get_schema(self) -> SchemaInfo:
        """Get complete database schema.

        Returns:
            SchemaInfo with all tables, columns, and indexes

        Performance:
            Executes in < 200ms
        """
        tables = self.list_tables()
        table_schemas = []
        total_rows = 0
        total_size_bytes = 0

        for table_name in tables:
            table_info = self.get_table_info(table_name)
            table_schemas.append(table_info)
            total_rows += table_info.row_count
            total_size_bytes += table_info.size_bytes

        return SchemaInfo(
            tables=table_schemas,
            total_tables=len(table_schemas),
            total_rows=total_rows,
            total_size_bytes=total_size_bytes,
        )

    def list_tables(self) -> list[str]:
        """List all tables in database.

        Returns:
            List of table names

        Performance:
            Executes in < 50ms
        """
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        rows = self.repo.query_all(query, [])
        return [row[0] for row in rows if row]

    def get_table_info(self, table_name: str) -> TableSchema:
        """Get metadata for a specific table.

        Args:
            table_name: Name of table to inspect

        Returns:
            TableSchema with columns and indexes

        Performance:
            Executes in < 100ms per table
        """
        # Get columns
        columns = []
        col_query = (
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'main'"
        )
        col_rows = self.repo.query_all(col_query, [table_name])
        for row in col_rows:
            if row:
                col = ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    nullable=row[2] in ("YES", True),
                )
                columns.append(col)

        # Get indexes
        indexes = []
        idx_query = (
            "SELECT index_name, column_names, unique, primary "
            "FROM duckdb_indexes() "
            "WHERE table_name = ?"
        )
        idx_rows = self.repo.query_all(idx_query, [table_name])
        for row in idx_rows:
            if row:
                idx = IndexInfo(
                    name=row[0],
                    table_name=table_name,
                    columns=[row[1]] if row[1] else [],
                    is_unique=row[2] in (True, 1),
                    is_primary=row[3] in (True, 1),
                )
                indexes.append(idx)

        # Get table stats
        stats_query = (
            f"SELECT count(*), sum(total_bytes) FROM (SELECT total_bytes FROM {table_name})"
        )
        stats_rows = self.repo.query_one(stats_query, [])
        row_count = stats_rows[0] if stats_rows and stats_rows[0] else 0
        size_bytes = stats_rows[1] if stats_rows and stats_rows[1] else 0

        return TableSchema(
            name=table_name,
            columns=columns,
            indexes=indexes,
            row_count=row_count,
            size_bytes=size_bytes,
        )

    def get_indexes(self, table_name: str | None = None) -> list[IndexInfo]:
        """Get indexes for a table or all tables.

        Args:
            table_name: Specific table name, or None for all

        Returns:
            List of IndexInfo objects

        Performance:
            Executes in < 100ms
        """
        if table_name:
            query = "SELECT index_name, column_names, unique, primary FROM duckdb_indexes() WHERE table_name = ?"
            rows = self.repo.query_all(query, [table_name])
        else:
            query = (
                "SELECT table_name, index_name, column_names, unique, primary FROM duckdb_indexes()"
            )
            rows = self.repo.query_all(query, [])

        indexes = []
        for row in rows:
            if row:
                if table_name:
                    idx = IndexInfo(
                        name=row[0],
                        table_name=table_name,
                        columns=[row[1]] if row[1] else [],
                        is_unique=row[2] in (True, 1),
                        is_primary=row[3] in (True, 1),
                    )
                else:
                    idx = IndexInfo(
                        name=row[1],
                        table_name=row[0],
                        columns=[row[2]] if row[2] else [],
                        is_unique=row[3] in (True, 1),
                        is_primary=row[4] in (True, 1),
                    )
                indexes.append(idx)

        return indexes
