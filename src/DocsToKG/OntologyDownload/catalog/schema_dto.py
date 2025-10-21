"""Data Transfer Objects for schema introspection.

Represents database schema information, tables, columns, and indexes
retrieved from DuckDB information schema.

NAVMAP:
  - SchemaInfo: Complete database schema
  - TableSchema: Table structure and metadata
  - ColumnInfo: Column definition and constraints
  - IndexInfo: Index information
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnInfo:
    """Column metadata.

    Attributes:
        name: Column name
        data_type: SQL data type
        nullable: Whether column allows NULL
        default_value: Default value if any
        constraints: List of constraints (PK, FK, etc.)
    """

    name: str
    data_type: str
    nullable: bool = True
    default_value: Optional[str] = None
    constraints: list[str] = None  # type: ignore

    def __post_init__(self):
        """Validate and normalize constraints."""
        if self.constraints is None:
            object.__setattr__(self, "constraints", [])


@dataclass(frozen=True)
class IndexInfo:
    """Index metadata.

    Attributes:
        name: Index name
        table_name: Table being indexed
        columns: List of column names
        is_unique: Whether index enforces uniqueness
        is_primary: Whether this is primary key
    """

    name: str
    table_name: str
    columns: list[str]
    is_unique: bool = False
    is_primary: bool = False


@dataclass(frozen=True)
class TableSchema:
    """Table structure and metadata.

    Attributes:
        name: Table name
        columns: List of ColumnInfo objects
        indexes: List of IndexInfo objects
        row_count: Approximate number of rows
        size_bytes: Table size in bytes
    """

    name: str
    columns: list[ColumnInfo]
    indexes: list[IndexInfo]
    row_count: int = 0
    size_bytes: int = 0

    @property
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]

    @property
    def primary_key_columns(self) -> list[str]:
        """Get list of primary key column names."""
        pk_indexes = [idx for idx in self.indexes if idx.is_primary]
        if pk_indexes:
            return pk_indexes[0].columns
        return []


@dataclass(frozen=True)
class SchemaInfo:
    """Complete database schema.

    Attributes:
        tables: List of TableSchema objects
        total_tables: Number of tables
        total_rows: Total rows across all tables
        total_size_bytes: Total size of all tables
    """

    tables: list[TableSchema]
    total_tables: int = 0
    total_rows: int = 0
    total_size_bytes: int = 0

    @property
    def table_names(self) -> list[str]:
        """Get list of table names."""
        return [table.name for table in self.tables]

    @property
    def total_size_mb(self) -> float:
        """Get total size in MB."""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def average_rows_per_table(self) -> float:
        """Calculate average rows per table."""
        if self.total_tables == 0:
            return 0.0
        return self.total_rows / self.total_tables
