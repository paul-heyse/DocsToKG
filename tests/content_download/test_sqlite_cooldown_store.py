"""Tests for SQLiteCooldownStore cross-process cooldown persistence."""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore


class TestSQLiteCooldownStoreBasics:
    """Basic operations: set, get, clear."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> SQLiteCooldownStore:
        """Create a temporary cooldown store."""
        return SQLiteCooldownStore(tmp_path / "test.sqlite")

    def test_set_and_get_until(self, store: SQLiteCooldownStore) -> None:
        """Test setting and retrieving a monotonic deadline."""
        host = "api.example.com"
        deadline = time.monotonic() + 60.0

        store.set_until(host, deadline, reason="test")
        retrieved = store.get_until(host)

        assert retrieved is not None
        assert retrieved > time.monotonic()  # Still in the future
        assert abs(retrieved - deadline) < 1.0  # Within 1 second

    def test_get_until_returns_none_when_not_set(self, store: SQLiteCooldownStore) -> None:
        """Test that get_until returns None for unknown hosts."""
        assert store.get_until("unknown.host") is None

    def test_clear_removes_cooldown(self, store: SQLiteCooldownStore) -> None:
        """Test that clear() removes a cooldown entry."""
        host = "api.example.com"
        deadline = time.monotonic() + 60.0

        store.set_until(host, deadline, reason="test")
        assert store.get_until(host) is not None

        store.clear(host)
        assert store.get_until(host) is None

    def test_clear_nonexistent_host_is_safe(self, store: SQLiteCooldownStore) -> None:
        """Test that clearing a nonexistent host doesn't raise."""
        store.clear("unknown.host")  # Should not raise


class TestWallMonotonicConversion:
    """Wall-clock vs monotonic time conversions."""

    @pytest.fixture
    def store_with_mocks(self, tmp_path: Path) -> tuple[SQLiteCooldownStore, MagicMock, MagicMock]:
        """Create a store with mocked time functions."""
        now_wall = MagicMock(return_value=1000.0)  # Fixed wall time
        now_mono = MagicMock(return_value=100.0)  # Fixed monotonic time

        store = SQLiteCooldownStore(
            tmp_path / "test.sqlite",
            now_wall=now_wall,
            now_mono=now_mono,
        )
        return store, now_wall, now_mono

    def test_monotonic_to_wall_conversion(
        self, store_with_mocks: tuple[SQLiteCooldownStore, MagicMock, MagicMock]
    ) -> None:
        """Test conversion from monotonic to wall time at write."""
        store, now_wall, now_mono = store_with_mocks

        # Set: monotonic 150 (= 100 + 50s delta)
        # Expected wall: 1050 (= 1000 + 50s delta)
        store.set_until("host1", 150.0, reason="test")

        # Verify by reading from DB directly
        cursor = store._conn.cursor()
        row = cursor.execute(
            "SELECT until_wall FROM breaker_cooldowns WHERE host=?", ("host1",)
        ).fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(1050.0)

    def test_wall_to_monotonic_conversion(
        self, store_with_mocks: tuple[SQLiteCooldownStore, MagicMock, MagicMock]
    ) -> None:
        """Test conversion from wall time to monotonic at read."""
        store, now_wall, now_mono = store_with_mocks

        # Set monotonic 150, which becomes wall 1050
        store.set_until("host1", 150.0, reason="test")

        # Advance time (both clocks) by 10s
        now_wall.return_value = 1010.0
        now_mono.return_value = 110.0

        # Get: should convert wall 1050 → monotonic 150 (110 + (1050 - 1010))
        # The deadline doesn't change, just the reference point shifts
        result = store.get_until("host1")
        assert result is not None
        assert result == pytest.approx(150.0)


class TestExpiry:
    """Expiry detection and pruning."""

    @pytest.fixture
    def store_with_time_control(
        self, tmp_path: Path
    ) -> tuple[SQLiteCooldownStore, MagicMock, MagicMock]:
        """Create a store with controllable time."""
        now_wall = MagicMock(return_value=1000.0)
        now_mono = MagicMock(return_value=100.0)

        store = SQLiteCooldownStore(
            tmp_path / "test.sqlite",
            now_wall=now_wall,
            now_mono=now_mono,
        )
        return store, now_wall, now_mono

    def test_expired_entry_is_cleaned_on_read(
        self, store_with_time_control: tuple[SQLiteCooldownStore, MagicMock, MagicMock]
    ) -> None:
        """Test that expired entries are deleted when read."""
        store, now_wall, now_mono = store_with_time_control

        # Set deadline 60s in the future (monotonic 160)
        store.set_until("host1", 160.0, reason="test")
        assert store.get_until("host1") is not None

        # Advance time past expiry
        now_wall.return_value = 1100.0  # 100s later
        now_mono.return_value = 200.0  # 100s later

        # Get should return None and clean up
        result = store.get_until("host1")
        assert result is None

        # Verify entry is deleted
        cursor = store._conn.cursor()
        count = cursor.execute(
            "SELECT COUNT(*) FROM breaker_cooldowns WHERE host=?", ("host1",)
        ).fetchone()[0]
        assert count == 0

    def test_prune_expired_removes_old_entries(
        self, store_with_time_control: tuple[SQLiteCooldownStore, MagicMock, MagicMock]
    ) -> None:
        """Test prune_expired() removes entries older than 1 second."""
        store, now_wall, now_mono = store_with_time_control

        # Set some cooldowns
        store.set_until("host1", 160.0, reason="test")  # Wall: 1060
        store.set_until("host2", 165.0, reason="test")  # Wall: 1065

        # Advance time
        now_wall.return_value = 1100.0

        # Prune should remove both (both expired > 1s ago)
        pruned = store.prune_expired()
        assert pruned == 2

        # Verify empty
        assert store.get_until("host1") is None
        assert store.get_until("host2") is None


class TestMultipleHosts:
    """Multiple hosts with independent cooldowns."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> SQLiteCooldownStore:
        return SQLiteCooldownStore(tmp_path / "test.sqlite")

    def test_multiple_hosts_independent(self, store: SQLiteCooldownStore) -> None:
        """Test that cooldowns for different hosts are independent."""
        now = time.monotonic()

        store.set_until("host1", now + 60.0, reason="test1")
        store.set_until("host2", now + 120.0, reason="test2")

        d1 = store.get_until("host1")
        d2 = store.get_until("host2")

        assert d1 is not None
        assert d2 is not None
        assert d2 > d1  # host2 has a later deadline

    def test_update_host_overwrites_previous(self, store: SQLiteCooldownStore) -> None:
        """Test that setting a cooldown twice overwrites the first."""
        now = time.monotonic()
        host = "host1"

        store.set_until(host, now + 60.0, reason="reason1")
        d1 = store.get_until(host)

        store.set_until(host, now + 30.0, reason="reason2")
        d2 = store.get_until(host)

        assert d2 is not None
        assert d1 is not None
        assert d2 < d1  # Shorter deadline


class TestReason:
    """Reason field storage and retrieval."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> SQLiteCooldownStore:
        return SQLiteCooldownStore(tmp_path / "test.sqlite")

    def test_reason_stored_and_retrievable(self, store: SQLiteCooldownStore) -> None:
        """Test that reason is stored in the database."""
        host = "api.example.com"
        now = time.monotonic()
        reason = "retry-after:420"

        store.set_until(host, now + 60.0, reason=reason)

        all_cooldowns = store.get_all_until()
        assert host in all_cooldowns
        _, stored_reason = all_cooldowns[host]
        assert stored_reason == reason

    def test_reason_truncated_to_128_chars(self, store: SQLiteCooldownStore) -> None:
        """Test that overly long reasons are truncated."""
        host = "api.example.com"
        now = time.monotonic()
        long_reason = "x" * 200

        store.set_until(host, now + 60.0, reason=long_reason)

        all_cooldowns = store.get_all_until()
        _, stored_reason = all_cooldowns[host]
        assert len(stored_reason) == 128


class TestGetAllUntil:
    """Diagnostics: get_all_until()."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> SQLiteCooldownStore:
        return SQLiteCooldownStore(tmp_path / "test.sqlite")

    def test_get_all_until_returns_all_cooldowns(self, store: SQLiteCooldownStore) -> None:
        """Test that get_all_until returns all entries."""
        now = time.monotonic()

        store.set_until("host1", now + 60.0, reason="r1")
        store.set_until("host2", now + 120.0, reason="r2")

        all_cd = store.get_all_until()
        assert len(all_cd) == 2
        assert "host1" in all_cd
        assert "host2" in all_cd

    def test_get_all_until_empty_when_no_entries(self, store: SQLiteCooldownStore) -> None:
        """Test that get_all_until returns empty dict when no entries."""
        all_cd = store.get_all_until()
        assert len(all_cd) == 0


class TestDatabaseInitialization:
    """Database creation and schema."""

    def test_database_file_created(self, tmp_path: Path) -> None:
        """Test that database file is created."""
        db_path = tmp_path / "subdir" / "breakers.sqlite"
        assert not db_path.exists()

        store = SQLiteCooldownStore(db_path)
        assert db_path.exists()

    def test_schema_created_on_init(self, tmp_path: Path) -> None:
        """Test that schema is created on initialization."""
        db_path = tmp_path / "test.sqlite"
        store = SQLiteCooldownStore(db_path)

        cursor = store._conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='breaker_cooldowns'"
        ).fetchall()

        assert len(tables) == 1
        assert tables[0][0] == "breaker_cooldowns"

    def test_pragmas_set_correctly(self, tmp_path: Path) -> None:
        """Test that WAL and synchronous pragmas are set."""
        db_path = tmp_path / "test.sqlite"
        store = SQLiteCooldownStore(db_path)

        cursor = store._conn.cursor()
        journal_mode = cursor.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode.upper() == "WAL"
