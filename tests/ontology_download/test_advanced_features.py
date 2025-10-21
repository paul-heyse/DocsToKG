"""Tests for advanced features - profiling and schema introspection.

Tests cover profiling DTOs, schema DTOs, and their properties.
"""

from __future__ import annotations

import pytest

from DocsToKG.OntologyDownload.catalog.profiling_dto import PlanStep, QueryProfile
from DocsToKG.OntologyDownload.catalog.schema_dto import (
    ColumnInfo,
    IndexInfo,
    SchemaInfo,
    TableSchema,
)


class TestPlanStep:
    """Test PlanStep DTO."""

    def test_plan_step_creation(self):
        """Test creating a plan step."""
        step = PlanStep(
            step_name="Seq Scan",
            rows_estimated=1000,
            rows_actual=1000,
            startup_cost=0.0,
            total_cost=100.0,
            duration_ms=5.0,
        )

        assert step.step_name == "Seq Scan"
        assert step.rows_estimated == 1000
        assert step.duration_ms == 5.0

    def test_plan_step_frozen(self):
        """Test that plan steps are immutable."""
        step = PlanStep(
            step_name="Seq Scan",
            rows_estimated=1000,
            rows_actual=1000,
            startup_cost=0.0,
            total_cost=100.0,
            duration_ms=5.0,
        )

        with pytest.raises(AttributeError):
            step.step_name = "Hash Join"


class TestQueryProfile:
    """Test QueryProfile DTO and properties."""

    def test_query_profile_creation(self):
        """Test creating a query profile."""
        steps = [
            PlanStep(
                step_name="Seq Scan",
                rows_estimated=1000,
                rows_actual=950,
                startup_cost=0.0,
                total_cost=100.0,
                duration_ms=5.0,
            )
        ]

        profile = QueryProfile(
            query="SELECT * FROM users",
            plan_steps=steps,
            total_rows=950,
            total_duration_ms=10.0,
            total_cost=100.0,
            planning_time_ms=1.0,
            effective_cache_size_bytes=1000000,
            suggestions=["Add index on user_id"],
        )

        assert profile.query == "SELECT * FROM users"
        assert len(profile.plan_steps) == 1

    def test_is_expensive(self):
        """Test expensive query detection."""
        steps = [
            PlanStep(
                step_name="Join",
                rows_estimated=10000,
                rows_actual=10000,
                startup_cost=100.0,
                total_cost=1500.0,
                duration_ms=50.0,
            )
        ]

        profile = QueryProfile(
            query="SELECT * FROM large_join",
            plan_steps=steps,
            total_rows=10000,
            total_duration_ms=50.0,
            total_cost=1500.0,
            planning_time_ms=2.0,
            effective_cache_size_bytes=1000000,
            suggestions=[],
        )

        assert profile.is_expensive

    def test_is_slow(self):
        """Test slow query detection."""
        steps = [
            PlanStep(
                step_name="Seq Scan",
                rows_estimated=1000,
                rows_actual=1000,
                startup_cost=0.0,
                total_cost=100.0,
                duration_ms=150.0,
            )
        ]

        profile = QueryProfile(
            query="SELECT * FROM users",
            plan_steps=steps,
            total_rows=1000,
            total_duration_ms=150.0,
            total_cost=100.0,
            planning_time_ms=1.0,
            effective_cache_size_bytes=1000000,
            suggestions=["Consider caching"],
        )

        assert profile.is_slow

    def test_critical_steps(self):
        """Test identifying critical plan steps."""
        steps = [
            PlanStep(
                step_name="Seq Scan",
                rows_estimated=1000,
                rows_actual=1000,
                startup_cost=0.0,
                total_cost=100.0,
                duration_ms=5.0,
            ),
            PlanStep(
                step_name="Hash Join",
                rows_estimated=5000,
                rows_actual=4900,
                startup_cost=500.0,
                total_cost=600.0,
                duration_ms=60.0,
            ),
        ]

        profile = QueryProfile(
            query="SELECT * FROM users JOIN orders",
            plan_steps=steps,
            total_rows=4900,
            total_duration_ms=65.0,
            total_cost=700.0,
            planning_time_ms=1.0,
            effective_cache_size_bytes=1000000,
            suggestions=[],
        )

        critical = profile.critical_steps
        assert len(critical) == 1
        assert critical[0].step_name == "Hash Join"


class TestColumnInfo:
    """Test ColumnInfo DTO."""

    def test_column_info_creation(self):
        """Test creating column info."""
        col = ColumnInfo(
            name="user_id",
            data_type="INTEGER",
            nullable=False,
            default_value=None,
            constraints=["PRIMARY KEY"],
        )

        assert col.name == "user_id"
        assert col.data_type == "INTEGER"
        assert "PRIMARY KEY" in col.constraints


class TestIndexInfo:
    """Test IndexInfo DTO."""

    def test_index_info_creation(self):
        """Test creating index info."""
        idx = IndexInfo(
            name="idx_user_email",
            table_name="users",
            columns=["email"],
            is_unique=True,
            is_primary=False,
        )

        assert idx.name == "idx_user_email"
        assert idx.is_unique


class TestTableSchema:
    """Test TableSchema DTO and properties."""

    def test_table_schema_creation(self):
        """Test creating table schema."""
        cols = [
            ColumnInfo("user_id", "INTEGER", False),
            ColumnInfo("email", "VARCHAR", False),
        ]
        indexes = [IndexInfo("idx_user_email", "users", ["email"], True, False)]

        schema = TableSchema(
            name="users",
            columns=cols,
            indexes=indexes,
            row_count=1000,
            size_bytes=50000,
        )

        assert schema.name == "users"
        assert len(schema.columns) == 2

    def test_column_names_property(self):
        """Test getting column names."""
        cols = [
            ColumnInfo("id", "INTEGER", False),
            ColumnInfo("name", "VARCHAR", False),
        ]

        schema = TableSchema(name="users", columns=cols, indexes=[], row_count=0, size_bytes=0)

        assert schema.column_names == ["id", "name"]

    def test_primary_key_columns(self):
        """Test getting primary key columns."""
        cols = [ColumnInfo("id", "INTEGER", False)]
        indexes = [IndexInfo("pk_users", "users", ["id"], True, True)]

        schema = TableSchema(name="users", columns=cols, indexes=indexes, row_count=0, size_bytes=0)

        assert schema.primary_key_columns == ["id"]


class TestSchemaInfo:
    """Test SchemaInfo DTO and properties."""

    def test_schema_info_creation(self):
        """Test creating schema info."""
        tables = [
            TableSchema(
                name="users",
                columns=[ColumnInfo("id", "INTEGER", False)],
                indexes=[],
                row_count=1000,
                size_bytes=50000,
            )
        ]

        schema = SchemaInfo(
            tables=tables,
            total_tables=1,
            total_rows=1000,
            total_size_bytes=50000,
        )

        assert schema.total_tables == 1

    def test_table_names_property(self):
        """Test getting table names."""
        tables = [
            TableSchema("users", [], [], 0, 0),
            TableSchema("orders", [], [], 0, 0),
        ]

        schema = SchemaInfo(tables=tables, total_tables=2, total_rows=0, total_size_bytes=0)

        assert schema.table_names == ["users", "orders"]

    def test_total_size_mb_property(self):
        """Test size calculation in MB."""
        tables = [TableSchema("users", [], [], 0, 0)]

        # 1 MB = 1024 * 1024 bytes
        schema = SchemaInfo(
            tables=tables,
            total_tables=1,
            total_rows=0,
            total_size_bytes=1024 * 1024,
        )

        assert schema.total_size_mb == 1.0

    def test_average_rows_per_table(self):
        """Test average rows calculation."""
        tables = [
            TableSchema("users", [], [], 0, 0),
            TableSchema("orders", [], [], 0, 0),
        ]

        schema = SchemaInfo(tables=tables, total_tables=2, total_rows=1000, total_size_bytes=0)

        assert schema.average_rows_per_table == 500.0
