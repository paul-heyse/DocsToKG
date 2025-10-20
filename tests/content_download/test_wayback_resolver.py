"""Tests for the Wayback resolver implementation."""

import json
from unittest.mock import Mock, patch
from pathlib import Path

import httpx
import pytest

from DocsToKG.ContentDownload.core import WorkArtifact
from DocsToKG.ContentDownload.pipeline import ResolverConfig
from DocsToKG.ContentDownload.resolvers.wayback import WaybackResolver
from DocsToKG.ContentDownload.resolvers.base import ResolverEvent, ResolverEventReason


class TestWaybackResolver:
    """Test cases for WaybackResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a WaybackResolver instance."""
        return WaybackResolver()

    @pytest.fixture
    def config(self):
        """Create a ResolverConfig instance."""
        config = ResolverConfig()
        config.wayback_config = {
            "year_window": 2,
            "max_snapshots": 8,
            "min_pdf_bytes": 4096,
            "html_parse": True,
        }
        return config

    @pytest.fixture
    def artifact(self):
        """Create a WorkArtifact instance."""
        return WorkArtifact(
            work_id="test-work",
            title="Test Work",
            failed_pdf_urls=["https://example.com/paper.pdf"],
            publication_year=2023,
            doi=None,
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="test-work",
            pdf_dir=Path("/tmp/pdf"),
            html_dir=Path("/tmp/html"),
            xml_dir=Path("/tmp/xml"),
        )

    def test_is_enabled_with_failed_urls(self, resolver, config, artifact):
        """Test that resolver is enabled when there are failed PDF URLs."""
        assert resolver.is_enabled(config, artifact) is True

    def test_is_enabled_without_failed_urls(self, resolver, config):
        """Test that resolver is disabled when there are no failed PDF URLs."""
        artifact = WorkArtifact(
            work_id="test-work",
            title="Test Work",
            failed_pdf_urls=[],
            publication_year=2023,
            doi=None,
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="test-work",
            pdf_dir=Path("/tmp/pdf"),
            html_dir=Path("/tmp/html"),
            xml_dir=Path("/tmp/xml"),
        )
        assert resolver.is_enabled(config, artifact) is False

    def test_iter_urls_no_failed_urls(self, resolver, config):
        """Test that resolver skips when there are no failed URLs."""
        artifact = WorkArtifact(
            work_id="test-work",
            title="Test Work",
            failed_pdf_urls=[],
            publication_year=2023,
            doi=None,
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="test-work",
            pdf_dir=Path("/tmp/pdf"),
            html_dir=Path("/tmp/html"),
            xml_dir=Path("/tmp/xml"),
        )
        client = Mock()

        results = list(resolver.iter_urls(client, config, artifact))
        assert len(results) == 1
        assert results[0].is_event is True
        assert results[0].event == ResolverEvent.SKIPPED
        assert results[0].event_reason == ResolverEventReason.NO_FAILED_URLS

    @patch("DocsToKG.ContentDownload.resolvers.wayback.request_with_retries")
    def test_iter_urls_availability_success(self, mock_request, resolver, config, artifact):
        """Test successful availability API response."""
        # Mock availability response
        mock_response = Mock()
        mock_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
                    "timestamp": "20230101000000",
                    "status": "200",
                }
            }
        }
        mock_response.headers = {"content-type": "application/pdf", "content-length": "50000"}
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        client = Mock()
        results = list(resolver.iter_urls(client, config, artifact))

        assert len(results) == 1
        assert (
            results[0].url
            == "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"
        )
        assert results[0].metadata["source"] == "wayback"
        assert results[0].metadata["discovery_method"] == "availability"

    @patch("DocsToKG.ContentDownload.resolvers.wayback.request_with_retries")
    def test_iter_urls_cdx_success(self, mock_request, resolver, config, artifact):
        """Test successful CDX API response."""
        # Mock availability response (no results)
        availability_response = Mock()
        availability_response.json.return_value = {"archived_snapshots": {}}
        availability_response.status_code = 200

        # Mock CDX response
        cdx_response = Mock()
        cdx_response.json.return_value = [
            [
                "urlkey",
                "timestamp",
                "original",
                "mimetype",
                "statuscode",
                "digest",
                "length",
                "archive_url",
            ],
            [
                "example.com/paper.pdf",
                "20230101000000",
                "https://example.com/paper.pdf",
                "application/pdf",
                "200",
                "ABC123",
                "50000",
                "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
            ],
        ]
        cdx_response.status_code = 200

        # Mock HEAD response for PDF verification
        head_response = Mock()
        head_response.headers = {"content-type": "application/pdf", "content-length": "50000"}
        head_response.status_code = 200

        mock_request.side_effect = [availability_response, cdx_response, head_response]

        client = Mock()
        results = list(resolver.iter_urls(client, config, artifact))

        assert len(results) == 1
        assert (
            results[0].url
            == "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"
        )
        assert results[0].metadata["source"] == "wayback"
        assert results[0].metadata["discovery_method"] == "cdx_pdf_direct"

    @patch("DocsToKG.ContentDownload.resolvers.wayback.request_with_retries")
    def test_iter_urls_html_parse_success(self, mock_request, resolver, config, artifact):
        """Test successful HTML parsing to find PDF."""
        # Mock availability response (no results)
        availability_response = Mock()
        availability_response.json.return_value = {"archived_snapshots": {}}
        availability_response.status_code = 200

        # Mock CDX response with HTML snapshots
        cdx_response = Mock()
        cdx_response.json.return_value = [
            [
                "urlkey",
                "timestamp",
                "original",
                "mimetype",
                "statuscode",
                "digest",
                "length",
                "archive_url",
            ],
            [
                "example.com/",
                "20230101000000",
                "https://example.com/",
                "text/html",
                "200",
                "ABC123",
                "5000",
                "https://web.archive.org/web/20230101000000/https://example.com/",
            ],
        ]
        cdx_response.status_code = 200

        # Mock HTML response
        html_response = Mock()
        html_response.text = """
        <html>
        <head>
            <meta name="citation_pdf_url" content="https://example.com/paper.pdf">
        </head>
        <body>Content</body>
        </html>
        """
        html_response.headers = {"content-type": "text/html"}
        html_response.status_code = 200

        # Mock HEAD response for PDF verification
        head_response = Mock()
        head_response.headers = {"content-type": "application/pdf", "content-length": "50000"}
        head_response.status_code = 200

        mock_request.side_effect = [
            availability_response,
            cdx_response,
            html_response,
            head_response,
        ]

        client = Mock()
        results = list(resolver.iter_urls(client, config, artifact))

        assert len(results) == 1
        assert results[0].url == "https://example.com/paper.pdf"
        assert results[0].metadata["source"] == "wayback"
        assert results[0].metadata["discovery_method"] == "cdx_html_parse"

    @patch("DocsToKG.ContentDownload.resolvers.wayback.request_with_retries")
    def test_iter_urls_no_snapshots(self, mock_request, resolver, config, artifact):
        """Test when no snapshots are found."""
        # Mock availability response (no results)
        availability_response = Mock()
        availability_response.json.return_value = {"archived_snapshots": {}}
        availability_response.status_code = 200

        # Mock CDX response (empty)
        cdx_response = Mock()
        cdx_response.json.return_value = [
            [
                "urlkey",
                "timestamp",
                "original",
                "mimetype",
                "statuscode",
                "digest",
                "length",
                "archive_url",
            ]
        ]
        cdx_response.status_code = 200

        mock_request.side_effect = [availability_response, cdx_response]

        client = Mock()
        results = list(resolver.iter_urls(client, config, artifact))

        assert len(results) == 1
        assert results[0].is_event is True
        assert results[0].event == ResolverEvent.SKIPPED
        assert results[0].event_reason == ResolverEventReason.NO_FAILED_URLS
        assert results[0].metadata["reason"] == "no_snapshot"

    @patch("DocsToKG.ContentDownload.resolvers.wayback.request_with_retries")
    def test_iter_urls_http_error(self, mock_request, resolver, config, artifact):
        """Test handling of HTTP errors."""
        mock_request.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=Mock(status_code=500)
        )

        client = Mock()
        results = list(resolver.iter_urls(client, config, artifact))

        assert len(results) == 1
        assert results[0].is_event is True
        assert results[0].event == ResolverEvent.ERROR
        assert results[0].event_reason == ResolverEventReason.UNEXPECTED_ERROR

    def test_discover_snapshots_with_publication_year(self, resolver, config):
        """Test snapshot discovery with publication year."""
        client = Mock()
        original_url = "https://example.com/paper.pdf"
        canonical_url = "https://example.com/paper.pdf"
        publication_year = 2023

        with (
            patch.object(resolver, "_check_availability") as mock_availability,
            patch.object(resolver, "_query_cdx") as mock_cdx,
            patch.object(resolver, "_verify_pdf_snapshot") as mock_verify,
        ):
            mock_availability.return_value = (None, {"availability_checked": True})
            mock_cdx.return_value = []
            mock_verify.return_value = False

            url, metadata = resolver._discover_snapshots(
                client, config, original_url, canonical_url, publication_year, 2, 8, 4096, True
            )

            assert url is None
            assert metadata["discovery_method"] == "none"

    def test_verify_pdf_snapshot_valid_pdf(self, resolver, config):
        """Test PDF verification with valid PDF."""
        client = Mock()
        url = "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"

        with patch(
            "DocsToKG.ContentDownload.resolvers.wayback.request_with_retries"
        ) as mock_request:
            mock_response = Mock()
            mock_response.headers = {"content-type": "application/pdf", "content-length": "50000"}
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            result = resolver._verify_pdf_snapshot(client, config, url, 4096)

            assert result is True

    def test_verify_pdf_snapshot_invalid_content_type(self, resolver, config):
        """Test PDF verification with invalid content type."""
        client = Mock()
        url = "https://web.archive.org/web/20230101000000/https://example.com/page.html"

        with patch(
            "DocsToKG.ContentDownload.resolvers.wayback.request_with_retries"
        ) as mock_request:
            mock_response = Mock()
            mock_response.headers = {"content-type": "text/html", "content-length": "5000"}
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            result = resolver._verify_pdf_snapshot(client, config, url, 4096)

            assert result is False

    def test_verify_pdf_snapshot_too_small(self, resolver, config):
        """Test PDF verification with file too small."""
        client = Mock()
        url = "https://web.archive.org/web/20230101000000/https://example.com/small.pdf"

        with patch(
            "DocsToKG.ContentDownload.resolvers.wayback.request_with_retries"
        ) as mock_request:
            mock_response = Mock()
            mock_response.headers = {"content-type": "application/pdf", "content-length": "1000"}
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            result = resolver._verify_pdf_snapshot(client, config, url, 4096)

            assert result is False

    @patch("bs4.BeautifulSoup")
    def test_parse_html_for_pdf_success(self, mock_beautifulsoup, resolver, config):
        """Test HTML parsing to find PDF URL."""
        client = Mock()
        html_url = "https://web.archive.org/web/20230101000000/https://example.com/"

        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_beautifulsoup.return_value = mock_soup

        # Mock find_pdf_via_meta to return a URL
        with patch(
            "DocsToKG.ContentDownload.resolvers.wayback.find_pdf_via_meta"
        ) as mock_find_meta:
            mock_find_meta.return_value = "https://example.com/paper.pdf"

            with patch(
                "DocsToKG.ContentDownload.resolvers.wayback.request_with_retries"
            ) as mock_request:
                mock_response = Mock()
                mock_response.text = "<html><head><meta name='citation_pdf_url' content='https://example.com/paper.pdf'></head></html>"
                mock_response.headers = {"content-type": "text/html"}
                mock_response.status_code = 200
                mock_request.return_value = mock_response

                result = resolver._parse_html_for_pdf(client, config, html_url)

                assert result == "https://example.com/paper.pdf"

    def test_parse_html_for_pdf_no_beautifulsoup(self, resolver, config):
        """Test HTML parsing when BeautifulSoup is not available."""
        client = Mock()
        html_url = "https://web.archive.org/web/20230101000000/https://example.com/"

        with patch(
            "DocsToKG.ContentDownload.resolvers.wayback.request_with_retries"
        ) as mock_request:
            mock_response = Mock()
            mock_response.text = "<html>Content</html>"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            with patch("bs4.BeautifulSoup", side_effect=ImportError):
                result = resolver._parse_html_for_pdf(client, config, html_url)

                assert result is None
