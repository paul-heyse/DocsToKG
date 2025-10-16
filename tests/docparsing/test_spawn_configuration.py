# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_spawn_configuration",
#   "purpose": "Pytest coverage for docparsing spawn configuration scenarios",
#   "sections": [
#     {
#       "id": "test_set_spawn_or_warn_success",
#       "name": "test_set_spawn_or_warn_success",
#       "anchor": "TSSOW",
#       "kind": "function"
#     },
#     {
#       "id": "test_set_spawn_or_warn_already_spawn",
#       "name": "test_set_spawn_or_warn_already_spawn",
#       "anchor": "SSOW1",
#       "kind": "function"
#     },
#     {
#       "id": "test_set_spawn_or_warn_incompatible_method",
#       "name": "test_set_spawn_or_warn_incompatible_method",
#       "anchor": "SSOW2",
#       "kind": "function"
#     },
#     {
#       "id": "test_set_spawn_or_warn_unset_method",
#       "name": "test_set_spawn_or_warn_unset_method",
#       "anchor": "SSOW3",
#       "kind": "function"
#     },
#     {
#       "id": "test_set_spawn_or_warn_no_logger",
#       "name": "test_set_spawn_or_warn_no_logger",
#       "anchor": "SSOW4",
#       "kind": "function"
#     },
#     {
#       "id": "test_set_spawn_or_warn_cuda_safety",
#       "name": "test_set_spawn_or_warn_cuda_safety",
#       "anchor": "SSOW5",
#       "kind": "function"
#     },
#     {
#       "id": "test_multiprocessing_import_safety",
#       "name": "test_multiprocessing_import_safety",
#       "anchor": "TMIS",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test multiprocessing spawn setup for CUDA safety."""

import logging
from unittest.mock import Mock, patch

import pytest

from DocsToKG.DocParsing._common import set_spawn_or_warn


def test_set_spawn_or_warn_success():
    """Test successful spawn method setting."""
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        logger = Mock(spec=logging.Logger)

        # Test successful setting
        set_spawn_or_warn(logger)

        mock_set_start_method.assert_called_once_with("spawn", force=True)
        logger.debug.assert_called_with("Multiprocessing start method set to 'spawn'")


def test_set_spawn_or_warn_already_spawn():
    """Test when spawn method is already set."""
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        mock_set_start_method.side_effect = RuntimeError("Already set")

        with patch("multiprocessing.get_start_method") as mock_get_start_method:
            mock_get_start_method.return_value = "spawn"

            logger = Mock(spec=logging.Logger)
            set_spawn_or_warn(logger)

            mock_set_start_method.assert_called_once_with("spawn", force=True)
            logger.debug.assert_called_with("Multiprocessing start method already 'spawn'")


def test_set_spawn_or_warn_incompatible_method():
    """Test warning when incompatible start method is set."""
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        mock_set_start_method.side_effect = RuntimeError("Already set")

        with patch("multiprocessing.get_start_method") as mock_get_start_method:
            mock_get_start_method.return_value = "fork"

            logger = Mock(spec=logging.Logger)
            set_spawn_or_warn(logger)

            mock_set_start_method.assert_called_once_with("spawn", force=True)
            logger.warning.assert_called_with(
                "Multiprocessing start method is fork; CUDA workloads require 'spawn'."
            )


def test_set_spawn_or_warn_unset_method():
    """Test warning when start method is unset."""
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        mock_set_start_method.side_effect = RuntimeError("Already set")

        with patch("multiprocessing.get_start_method") as mock_get_start_method:
            mock_get_start_method.return_value = None

            logger = Mock(spec=logging.Logger)
            set_spawn_or_warn(logger)

            mock_set_start_method.assert_called_once_with("spawn", force=True)
            logger.warning.assert_called_with(
                "Multiprocessing start method is unset; CUDA workloads require 'spawn'."
            )


def test_set_spawn_or_warn_no_logger():
    """Test function works without logger parameter."""
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        mock_set_start_method.side_effect = RuntimeError("Already set")

        with patch("multiprocessing.get_start_method") as mock_get_start_method:
            mock_get_start_method.return_value = "fork"

            # Should not raise exception
            set_spawn_or_warn()

            mock_set_start_method.assert_called_once_with("spawn", force=True)


def test_set_spawn_or_warn_cuda_safety():
    """Test that spawn method ensures CUDA safety."""
    # This test verifies that the function attempts to set spawn
    # which is required for CUDA safety in multiprocessing
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        set_spawn_or_warn()

        # Verify spawn method was requested
        mock_set_start_method.assert_called_once_with("spawn", force=True)

        # Verify force=True is used to override existing method
        args, kwargs = mock_set_start_method.call_args
        assert kwargs["force"] is True


def test_multiprocessing_import_safety():
    """Test that multiprocessing import doesn't cause issues."""
    # This test ensures the function can be imported and called
    # without causing multiprocessing-related import issues
    try:
        from DocsToKG.DocParsing._common import set_spawn_or_warn

        set_spawn_or_warn()
        assert True  # If we get here, no exception was raised
    except ImportError as e:
        pytest.fail(f"Multiprocessing import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
