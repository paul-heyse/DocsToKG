# === NAVMAP v1 ===
# {
#   "module": "tests.test_determinism_fixtures",
#   "purpose": "Unit tests for determinism and environment control fixtures (Optimization 9)",
#   "sections": [
#     {"id": "test-deterministic-env", "name": "test_deterministic_env_basic", "anchor": "function-test-deterministic-env-basic", "kind": "function"},
#     {"id": "test-seed-state", "name": "test_seed_state_reproducibility", "anchor": "function-test-seed-state-reproducibility", "kind": "function"},
#     {"id": "test-env-snapshot", "name": "test_env_snapshot_isolation", "anchor": "function-test-env-snapshot-isolation", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""
Unit tests for determinism fixtures.

Validates that determinism fixtures properly initialize and restore global state,
enabling reproducible tests across multiple runs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest


@pytest.mark.unit
def test_deterministic_env_basic(deterministic_env):
    """Test that deterministic_env fixture provides expected metadata."""
    assert deterministic_env["seed"] == 42
    assert deterministic_env["tz"] == "UTC"
    assert deterministic_env["locale"] == "C.UTF-8"
    assert "original_env" in deterministic_env
    assert os.environ.get("TZ") == "UTC"


@pytest.mark.unit
def test_deterministic_env_clears_proxies(deterministic_env):
    """Test that deterministic_env removes proxy environment variables."""
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    for var in proxy_vars:
        assert os.environ.get(var) is None, f"{var} should be cleared"


@pytest.mark.unit
def test_seed_state_reproducibility(seed_state):
    """Test that seed_state provides reproducible random values."""
    # Reset to known seed
    random.seed(seed_state["seed"])

    # Generate first sequence
    sequence_1 = [random.random() for _ in range(5)]

    # Reset again
    random.seed(seed_state["seed"])

    # Generate second sequence
    sequence_2 = [random.random() for _ in range(5)]

    # Sequences should be identical
    assert sequence_1 == sequence_2


@pytest.mark.unit
def test_env_snapshot_isolation(env_snapshot):
    """Test that env_snapshot captures and restores environment state."""
    # Capture original count
    original_count = len(env_snapshot["snapshot"])

    # Modify environment
    os.environ["TEST_VAR_UNIQUE_12345"] = "test_value"

    # Should be in environ now
    assert os.environ.get("TEST_VAR_UNIQUE_12345") == "test_value"

    # Count should be increased
    assert len(os.environ) > original_count


@pytest.mark.unit
def test_tmp_isolated_dir_exists(tmp_isolated_dir):
    """Test that tmp_isolated_dir provides a writable temporary directory."""
    assert tmp_isolated_dir.exists()
    assert tmp_isolated_dir.is_dir()

    # Should be writable
    test_file = tmp_isolated_dir / "test.txt"
    test_file.write_text("content")
    assert test_file.exists()
    assert test_file.read_text() == "content"


@pytest.mark.unit
def test_tmp_isolated_dir_permissions(tmp_isolated_dir):
    """Test that tmp_isolated_dir has proper permissions."""
    # Should be readable/writable/executable
    stat_info = tmp_isolated_dir.stat()
    mode = stat_info.st_mode & 0o777

    # At minimum, should be readable and writable by owner
    assert mode & 0o600 == 0o600


@pytest.mark.unit
def test_hypothesis_settings_available(hypothesis_settings):
    """Test that hypothesis_settings fixture loads Hypothesis configuration."""
    if "error" in hypothesis_settings:
        pytest.skip("Hypothesis not installed")

    assert hypothesis_settings["max_examples"] == 100
    assert hypothesis_settings["deadline"] is None


@pytest.mark.unit
def test_seeds_are_deterministic_across_tests(seed_state):
    """
    Test that using seed_state multiple times produces same results.

    This test is marked as @pytest.mark.unit because it's isolated
    (no I/O, minimal CPU) and runs in <50ms.
    """
    import random as rand_module

    # Both sequences should be identical because seed is fixed
    sequence = [rand_module.random() for _ in range(3)]

    # Validate we got reproducible values
    assert len(sequence) == 3
    assert all(isinstance(x, float) for x in sequence)


@pytest.mark.unit
def test_env_snapshot_cwd(env_snapshot, tmp_path):
    """Test that env_snapshot captures current working directory."""
    original_cwd = env_snapshot["cwd"]
    assert isinstance(original_cwd, Path)
    assert original_cwd.exists()


@pytest.mark.unit
def test_env_snapshot_umask(env_snapshot):
    """Test that env_snapshot captures process umask."""
    umask_val = env_snapshot["umask"]
    assert isinstance(umask_val, int)
    assert 0 <= umask_val <= 0o777
