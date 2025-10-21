# === NAVMAP v1 ===
# {
#   "module": "tests.fixtures.snapshot_assertions",
#   "purpose": "Assertion helpers for snapshot and golden testing",
#   "sections": [
#     {"id": "json-canonicalization", "name": "JSON Canonicalization", "anchor": "section-canonicalization", "kind": "section"},
#     {"id": "snapshot-assertions", "name": "SnapshotAssertions", "anchor": "class-snapshot-assertions", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""
Snapshot assertion helpers for deterministic output validation.

Provides utilities for canonical JSON comparison, sorted key ordering,
and regression detection across versions.
"""

from __future__ import annotations

import json
from typing import Any


def canonicalize_json(data: Any) -> str:
    """
    Canonicalize data as JSON with deterministic ordering.

    Args:
        data: Data to canonicalize

    Returns:
        Canonical JSON string with sorted keys

    Examples:
        >>> canonicalize_json({"b": 2, "a": 1})
        '{"a": 1, "b": 2}'

        >>> canonicalize_json([{"z": 0}, {"a": 1}])
        '[{"a": 1}, {"z": 0}]'
    """
    if isinstance(data, dict):
        # Sort dict by keys
        sorted_dict = {
            k: canonicalize_json(v) if isinstance(v, (dict, list)) else v
            for k, v in sorted(data.items())
        }
        return json.dumps(sorted_dict, sort_keys=True, indent=2)
    elif isinstance(data, list):
        # Recursively canonicalize list items
        canonical_items = [canonicalize_json(item) for item in data]
        return json.dumps([json.loads(item) for item in canonical_items], sort_keys=True, indent=2)
    elif isinstance(data, str):
        return data
    else:
        return json.dumps(data, indent=2)


class SnapshotAssertions:
    """Helper class for snapshot assertions."""

    @staticmethod
    def assert_json_equal(actual: Any, expected: Any, message: str = "") -> None:
        """
        Assert that two JSON objects are equal (ignoring key order).

        Args:
            actual: Actual output
            expected: Expected output
            message: Optional assertion message
        """
        actual_canonical = canonicalize_json(actual)
        expected_canonical = canonicalize_json(expected)
        assert (
            actual_canonical == expected_canonical
        ), f"{message}\nExpected:\n{expected_canonical}\n\nActual:\n{actual_canonical}"

    @staticmethod
    def assert_json_contains(actual: Any, substring: str, message: str = "") -> None:
        """
        Assert that canonical JSON contains substring.

        Args:
            actual: Actual output
            substring: Substring to find
            message: Optional assertion message
        """
        canonical = canonicalize_json(actual)
        assert (
            substring in canonical
        ), f"{message}\nSubstring '{substring}' not found in:\n{canonical}"

    @staticmethod
    def assert_json_keys(actual: dict[str, Any], expected_keys: list[str]) -> None:
        """
        Assert that JSON dict contains expected keys.

        Args:
            actual: Actual dict
            expected_keys: Expected keys
        """
        actual_keys = set(actual.keys())
        expected_set = set(expected_keys)
        assert (
            actual_keys == expected_set
        ), f"Key mismatch.\nExpected: {expected_set}\nActual: {actual_keys}"

    @staticmethod
    def assert_json_structure(actual: Any, template: type | dict[str, type]) -> None:
        """
        Assert that JSON matches expected structure.

        Args:
            actual: Actual output
            template: Expected structure template

        Examples:
            >>> SnapshotAssertions.assert_json_structure(
            ...     {"id": 1, "name": "test"},
            ...     {"id": int, "name": str}
            ... )
        """
        if isinstance(template, type):
            assert isinstance(
                actual, template
            ), f"Type mismatch. Expected {template}, got {type(actual)}"
        elif isinstance(template, dict):
            assert isinstance(actual, dict), f"Expected dict, got {type(actual)}"
            for key, expected_type in template.items():
                assert key in actual, f"Missing key: {key}"
                assert isinstance(actual[key], expected_type), (
                    f"Type mismatch for key '{key}'. Expected {expected_type}, "
                    f"got {type(actual[key])}"
                )

    @staticmethod
    def assert_not_in_snapshot(data: Any, forbidden: str) -> None:
        """
        Assert that forbidden string is not in snapshot.

        Useful for checking sensitive data is not leaked.

        Args:
            data: Data to check
            forbidden: Forbidden string that should not appear
        """
        canonical = canonicalize_json(data)
        assert (
            forbidden not in canonical
        ), f"Forbidden string '{forbidden}' found in snapshot:\n{canonical}"

    @staticmethod
    def assert_snapshot_diff(
        previous: Any,
        current: Any,
        allowed_changes: set[str] | None = None,
    ) -> None:
        """
        Assert that snapshot changes are in allowed set.

        Args:
            previous: Previous snapshot
            current: Current snapshot
            allowed_changes: Set of allowed changed keys (None = no changes allowed)
        """
        if allowed_changes is None:
            allowed_changes = set()

        prev_canonical = canonicalize_json(previous)
        curr_canonical = canonicalize_json(current)

        if prev_canonical == curr_canonical:
            return  # No changes - pass

        if not isinstance(previous, dict) or not isinstance(current, dict):
            raise AssertionError("Snapshot differs and both must be dicts for diff check")

        changed_keys = set()
        for key in set(previous.keys()) | set(current.keys()):
            if previous.get(key) != current.get(key):
                changed_keys.add(key)

        unexpected = changed_keys - allowed_changes
        assert not unexpected, (
            f"Unexpected snapshot changes in keys: {unexpected}\n"
            f"Allowed changes: {allowed_changes}\n"
            f"Previous: {prev_canonical}\n"
            f"Current: {curr_canonical}"
        )
