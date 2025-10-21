"""Integration tests for P1 Content-Length verification in streaming.

Tests verify that stream_to_part() correctly enforces Content-Length
matching actual bytes written when verify_content_length=True.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from DocsToKG.ContentDownload.io_utils import SizeMismatchError
from DocsToKG.ContentDownload.streaming import stream_to_part, StreamMetrics


class TestStreamToPartContentLengthVerification:
    """Tests for Content-Length verification in stream_to_part."""

    def test_content_length_match_succeeds(self, tmp_path: Path) -> None:
        """When actual bytes match expected_total, stream succeeds."""
        part_path = tmp_path / "test.part"

        # Mock client that streams exactly 1000 bytes
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes.return_value = [b"x" * 1000]

        client.build_request.return_value = MagicMock()
        client.send.return_value.__enter__.return_value = mock_resp
        client.send.return_value.__exit__.return_value = None

        # Stream with expected_total=1000 and verify_content_length=True
        metrics = stream_to_part(
            client=client,
            url="https://example.com/file",
            part_path=part_path,
            range_start=None,
            chunk_bytes=65536,
            do_fsync=False,
            preallocate_min=0,
            expected_total=1000,
            artifact_lock=lambda x: _null_context_manager(),
            logger=_mock_logger(),
            verify_content_length=True,
        )

        assert metrics.bytes_written == 1000
        assert part_path.exists()
        assert part_path.stat().st_size == 1000

    def test_content_length_mismatch_raises_error(self, tmp_path: Path) -> None:
        """When actual bytes don't match expected_total, SizeMismatchError is raised."""
        part_path = tmp_path / "test.part"

        # Mock client that streams 500 bytes
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes.return_value = [b"x" * 500]

        client.build_request.return_value = MagicMock()
        client.send.return_value.__enter__.return_value = mock_resp
        client.send.return_value.__exit__.return_value = None

        # Stream with expected_total=1000 but only 500 bytes arrive
        with pytest.raises(SizeMismatchError) as exc_info:
            stream_to_part(
                client=client,
                url="https://example.com/file",
                part_path=part_path,
                range_start=None,
                chunk_bytes=65536,
                do_fsync=False,
                preallocate_min=0,
                expected_total=1000,
                artifact_lock=lambda x: _null_context_manager(),
                logger=_mock_logger(),
                verify_content_length=True,
            )

        assert exc_info.value.expected == 1000
        assert exc_info.value.actual == 500

    def test_content_length_too_many_bytes_raises_error(self, tmp_path: Path) -> None:
        """When actual bytes exceed expected_total, SizeMismatchError is raised."""
        part_path = tmp_path / "test.part"

        # Mock client that streams 2000 bytes
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes.return_value = [b"x" * 2000]

        client.build_request.return_value = MagicMock()
        client.send.return_value.__enter__.return_value = mock_resp
        client.send.return_value.__exit__.return_value = None

        # Stream with expected_total=1000 but 2000 bytes arrive
        with pytest.raises(SizeMismatchError) as exc_info:
            stream_to_part(
                client=client,
                url="https://example.com/file",
                part_path=part_path,
                range_start=None,
                chunk_bytes=65536,
                do_fsync=False,
                preallocate_min=0,
                expected_total=1000,
                artifact_lock=lambda x: _null_context_manager(),
                logger=_mock_logger(),
                verify_content_length=True,
            )

        assert exc_info.value.expected == 1000
        assert exc_info.value.actual == 2000

    def test_content_length_disabled_skips_verification(self, tmp_path: Path) -> None:
        """When verify_content_length=False, mismatches are ignored."""
        part_path = tmp_path / "test.part"

        # Mock client that streams 500 bytes
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes.return_value = [b"x" * 500]

        client.build_request.return_value = MagicMock()
        client.send.return_value.__enter__.return_value = mock_resp
        client.send.return_value.__exit__.return_value = None

        # Stream with expected_total=1000 but verify_content_length=False
        metrics = stream_to_part(
            client=client,
            url="https://example.com/file",
            part_path=part_path,
            range_start=None,
            chunk_bytes=65536,
            do_fsync=False,
            preallocate_min=0,
            expected_total=1000,
            artifact_lock=lambda x: _null_context_manager(),
            logger=_mock_logger(),
            verify_content_length=False,
        )

        # Should not raise, just return metrics
        assert metrics.bytes_written == 500

    def test_content_length_none_skips_verification(self, tmp_path: Path) -> None:
        """When expected_total is None, verification is skipped."""
        part_path = tmp_path / "test.part"

        # Mock client that streams 500 bytes
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes.return_value = [b"x" * 500]

        client.build_request.return_value = MagicMock()
        client.send.return_value.__enter__.return_value = mock_resp
        client.send.return_value.__exit__.return_value = None

        # Stream with expected_total=None
        metrics = stream_to_part(
            client=client,
            url="https://example.com/file",
            part_path=part_path,
            range_start=None,
            chunk_bytes=65536,
            do_fsync=False,
            preallocate_min=0,
            expected_total=None,  # No verification possible
            artifact_lock=lambda x: _null_context_manager(),
            logger=_mock_logger(),
            verify_content_length=True,
        )

        # Should not raise, just return metrics
        assert metrics.bytes_written == 500

    def test_content_length_with_resume(self, tmp_path: Path) -> None:
        """Content-Length verification works correctly with resume."""
        part_path = tmp_path / "test.part"

        # Simulate partial file already present (500 bytes)
        part_path.write_bytes(b"y" * 500)

        # Mock client that streams additional 500 bytes (range request)
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.headers = {"Content-Range": "bytes 500-999/1000"}
        mock_resp.iter_bytes.return_value = [b"x" * 500]

        client.build_request.return_value = MagicMock()
        client.send.return_value.__enter__.return_value = mock_resp
        client.send.return_value.__exit__.return_value = None

        # Stream with range_start=500, expected_total=1000
        # Total should be 500 (already on disk) + 500 (new) = 1000
        metrics = stream_to_part(
            client=client,
            url="https://example.com/file",
            part_path=part_path,
            range_start=500,
            chunk_bytes=65536,
            do_fsync=False,
            preallocate_min=0,
            expected_total=1000,
            artifact_lock=lambda x: _null_context_manager(),
            logger=_mock_logger(),
            verify_content_length=True,
        )

        assert metrics.bytes_written == 500  # Only new bytes
        assert part_path.stat().st_size == 1000  # Total size


# ============================================================================
# Helpers
# ============================================================================


def _mock_logger():
    """Return a mock logger."""
    return MagicMock()


def _null_context_manager():
    """Return a null context manager."""
    from contextlib import contextmanager

    @contextmanager
    def _null():
        yield

    return _null()
