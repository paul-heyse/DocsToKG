# === NAVMAP v1 ===
# {
#   "module": "tests.fixtures.snapshot_fixtures",
#   "purpose": "Snapshot capture and comparison fixtures for golden testing",
#   "sections": [
#     {"id": "snapshot-manager", "name": "SnapshotManager", "anchor": "class-snapshot-manager", "kind": "class"},
#     {"id": "snapshot-fixture", "name": "snapshot_manager", "anchor": "fixture-snapshot-manager", "kind": "fixture"}
#   ]
# }
# === /NAVMAP ===

"""
Snapshot fixtures for golden and regression testing.

Provides snapshot capture, comparison, and update utilities for deterministic
output validation without external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pytest


class SnapshotManager:
    """Manager for snapshot capture and comparison."""

    def __init__(self, snapshots_dir: Path, test_name: str):
        """
        Initialize snapshot manager.

        Args:
            snapshots_dir: Directory to store snapshots
            test_name: Name of current test
        """
        self.snapshots_dir = snapshots_dir
        self.test_name = test_name
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, name: str) -> Path:
        """Get path for snapshot file."""
        return self.snapshots_dir / f"{self.test_name}_{name}.json"

    def capture(self, data: Any, name: str = "output") -> str:
        """
        Capture and canonicalize data as snapshot.

        Args:
            data: Data to capture (dict, list, str)
            name: Snapshot name

        Returns:
            Canonical JSON representation
        """
        if isinstance(data, dict):
            canonical = json.dumps(data, sort_keys=True, indent=2)
        elif isinstance(data, list):
            canonical = json.dumps(data, sort_keys=True, indent=2)
        elif isinstance(data, str):
            canonical = data
        else:
            canonical = json.dumps(str(data), indent=2)

        path = self._snapshot_path(name)
        path.write_text(canonical)
        return canonical

    def compare(self, data: Any, name: str = "output") -> tuple[bool, str, str]:
        """
        Compare data against stored snapshot.

        Args:
            data: Data to compare
            name: Snapshot name

        Returns:
            (matches, expected, actual): Comparison result and values
        """
        path = self._snapshot_path(name)

        if isinstance(data, dict):
            actual = json.dumps(data, sort_keys=True, indent=2)
        elif isinstance(data, list):
            actual = json.dumps(data, sort_keys=True, indent=2)
        elif isinstance(data, str):
            actual = data
        else:
            actual = json.dumps(str(data), indent=2)

        if not path.exists():
            # First run - capture as golden
            path.write_text(actual)
            return (True, actual, actual)

        expected = path.read_text()
        matches = expected == actual
        return (matches, expected, actual)

    def update(self, data: Any, name: str = "output") -> None:
        """Update snapshot with new data."""
        self.capture(data, name)

    def load(self, name: str = "output") -> dict[str, Any] | list[Any] | str:
        """Load snapshot from file."""
        path = self._snapshot_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

        content = path.read_text()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content


@pytest.fixture
def snapshot_manager(tmp_path: Path) -> Generator[SnapshotManager, None, None]:
    """
    Provide a snapshot manager for tests.

    Yields:
        SnapshotManager: Snapshot management instance

    Example:
        def test_with_snapshots(snapshot_manager):
            output = {"result": "success"}
            canonical = snapshot_manager.capture(output)
            assert canonical is not None

            matches, expected, actual = snapshot_manager.compare(output)
            assert matches
    """
    snapshots_dir = tmp_path / "snapshots"
    # Get test name from calling test
    test_name = "test_snapshot"

    yield SnapshotManager(snapshots_dir, test_name)
