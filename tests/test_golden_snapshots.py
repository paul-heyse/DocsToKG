# === NAVMAP v1 ===
# {
#   "module": "tests.test_golden_snapshots",
#   "purpose": "Golden and snapshot tests for regression detection",
#   "sections": [
#     {"id": "snapshot-capture-tests", "name": "Snapshot capture tests", "anchor": "snapshot-capture-tests", "kind": "section"},
#     {"id": "json-canonicalization-tests", "name": "JSON canonicalization tests", "anchor": "json-canonicalization-tests", "kind": "section"},
#     {"id": "regression-tests", "name": "Regression detection tests", "anchor": "regression-tests", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Golden and snapshot tests for regression detection.

Tests snapshot capture, comparison, and regression detection without
external dependencies.
"""

from __future__ import annotations

import pytest

from tests.fixtures.snapshot_assertions import canonicalize_json, SnapshotAssertions


# --- Snapshot Capture Tests ---


@pytest.mark.unit
def test_snapshot_manager_capture_dict(snapshot_manager):
    """Test capturing dict as snapshot."""
    data = {"name": "test", "value": 42}
    canonical = snapshot_manager.capture(data)
    assert "name" in canonical
    assert "test" in canonical


@pytest.mark.unit
def test_snapshot_manager_capture_list(snapshot_manager):
    """Test capturing list as snapshot."""
    data = [{"id": 1}, {"id": 2}]
    canonical = snapshot_manager.capture(data)
    assert "id" in canonical
    assert "1" in canonical


@pytest.mark.unit
def test_snapshot_manager_compare_match(snapshot_manager):
    """Test comparing matching snapshots."""
    data = {"result": "success", "code": 200}
    snapshot_manager.capture(data)
    
    matches, _, _ = snapshot_manager.compare(data)
    assert matches


@pytest.mark.unit
def test_snapshot_manager_compare_mismatch(snapshot_manager):
    """Test comparing non-matching snapshots."""
    original = {"result": "success"}
    snapshot_manager.capture(original)
    
    modified = {"result": "failed"}
    matches, expected, actual = snapshot_manager.compare(modified)
    assert not matches
    assert "success" in expected
    assert "failed" in actual


@pytest.mark.unit
def test_snapshot_manager_update(snapshot_manager):
    """Test updating snapshot."""
    original = {"version": 1}
    snapshot_manager.capture(original)
    
    updated = {"version": 2}
    snapshot_manager.update(updated)
    
    matches, _, _ = snapshot_manager.compare(updated)
    assert matches


@pytest.mark.unit
def test_snapshot_manager_load(snapshot_manager):
    """Test loading snapshot from disk."""
    data = {"key": "value"}
    snapshot_manager.capture(data)
    
    loaded = snapshot_manager.load()
    assert loaded == data


# --- JSON Canonicalization Tests ---


@pytest.mark.unit
def test_canonicalize_json_dict_ordering():
    """Test canonicalization orders dict keys."""
    data = {"z": 1, "a": 2}
    canonical = canonicalize_json(data)
    assert canonical.index("a") < canonical.index("z")


@pytest.mark.unit
def test_canonicalize_json_nested():
    """Test canonicalization of nested structures."""
    data = {"outer": {"z": 1, "a": 2}}
    canonical = canonicalize_json(data)
    # Should have deterministic ordering
    assert "outer" in canonical


@pytest.mark.unit
def test_canonicalize_json_list():
    """Test canonicalization of lists."""
    data = [{"b": 2}, {"a": 1}]
    canonical = canonicalize_json(data)
    assert isinstance(canonical, str)
    assert "a" in canonical


@pytest.mark.unit
def test_canonicalize_json_string():
    """Test canonicalization of strings."""
    data = "test string"
    canonical = canonicalize_json(data)
    assert canonical == "test string"


# --- Snapshot Assertions Tests ---


@pytest.mark.unit
def test_assert_json_equal_identical():
    """Test JSON equality assertion for identical objects."""
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 2, "a": 1}  # Different order, same content
    
    # Should not raise
    SnapshotAssertions.assert_json_equal(data1, data2)


@pytest.mark.unit
def test_assert_json_equal_different():
    """Test JSON equality assertion for different objects."""
    data1 = {"a": 1}
    data2 = {"a": 2}
    
    with pytest.raises(AssertionError):
        SnapshotAssertions.assert_json_equal(data1, data2)


@pytest.mark.unit
def test_assert_json_contains():
    """Test JSON contains assertion."""
    data = {"name": "test", "value": 42}
    
    # Should not raise
    SnapshotAssertions.assert_json_contains(data, "test")
    
    with pytest.raises(AssertionError):
        SnapshotAssertions.assert_json_contains(data, "missing")


@pytest.mark.unit
def test_assert_json_keys():
    """Test JSON keys assertion."""
    data = {"a": 1, "b": 2}
    
    # Should not raise
    SnapshotAssertions.assert_json_keys(data, ["a", "b"])
    
    with pytest.raises(AssertionError):
        SnapshotAssertions.assert_json_keys(data, ["a", "c"])


@pytest.mark.unit
def test_assert_json_structure_type():
    """Test JSON structure assertion for types."""
    data = [1, 2, 3]
    
    # Should not raise
    SnapshotAssertions.assert_json_structure(data, list)
    
    with pytest.raises(AssertionError):
        SnapshotAssertions.assert_json_structure(data, dict)


@pytest.mark.unit
def test_assert_json_structure_template():
    """Test JSON structure assertion for template."""
    data = {"id": 1, "name": "test"}
    template = {"id": int, "name": str}
    
    # Should not raise
    SnapshotAssertions.assert_json_structure(data, template)


@pytest.mark.unit
def test_assert_not_in_snapshot():
    """Test forbidden content assertion."""
    data = {"user": "alice", "role": "admin"}
    
    # Should not raise
    SnapshotAssertions.assert_not_in_snapshot(data, "secret")
    
    with pytest.raises(AssertionError):
        SnapshotAssertions.assert_not_in_snapshot(data, "alice")


@pytest.mark.unit
def test_assert_snapshot_diff_no_changes():
    """Test diff assertion with no changes."""
    previous = {"version": 1, "name": "test"}
    current = {"name": "test", "version": 1}
    
    # Should not raise (order doesn't matter)
    SnapshotAssertions.assert_snapshot_diff(previous, current)


@pytest.mark.unit
def test_assert_snapshot_diff_allowed_changes():
    """Test diff assertion with allowed changes."""
    previous = {"version": 1, "name": "test"}
    current = {"version": 2, "name": "test"}
    
    # Should not raise - version change is allowed
    SnapshotAssertions.assert_snapshot_diff(
        previous, current, allowed_changes={"version"}
    )


@pytest.mark.unit
def test_assert_snapshot_diff_unexpected_changes():
    """Test diff assertion with unexpected changes."""
    previous = {"version": 1, "name": "test"}
    current = {"version": 2, "name": "changed"}
    
    with pytest.raises(AssertionError):
        SnapshotAssertions.assert_snapshot_diff(
            previous, current, allowed_changes={"version"}
        )


# --- Integration Tests ---


@pytest.mark.component
def test_snapshot_full_workflow(snapshot_manager):
    """Test full snapshot workflow."""
    # Initial capture
    data = {"status": "ok", "count": 5}
    snapshot_manager.capture(data)
    
    # Compare (should match)
    matches, _, _ = snapshot_manager.compare(data)
    assert matches
    
    # Use assertions
    SnapshotAssertions.assert_json_equal(data, data)
    SnapshotAssertions.assert_json_contains(data, "ok")
    SnapshotAssertions.assert_json_keys(data, ["status", "count"])
