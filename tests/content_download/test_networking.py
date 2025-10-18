# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_networking",
#   "purpose": "Pytest coverage for content download networking scenarios",
#   "sections": [
#     {
#       "id": "helperresponse",
#       "name": "_HelperResponse",
#       "anchor": "class-helperresponse",
#       "kind": "class"
#     },
#     {
#       "id": "make-helper-response",
#       "name": "_make_helper_response",
#       "anchor": "function-make-helper-response",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-headers-empty-metadata",
#       "name": "test_build_headers_empty_metadata",
#       "anchor": "function-test-build-headers-empty-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-headers-etag-only",
#       "name": "test_build_headers_etag_only",
#       "anchor": "function-test-build-headers-etag-only",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-headers-last-modified-only",
#       "name": "test_build_headers_last_modified_only",
#       "anchor": "function-test-build-headers-last-modified-only",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-headers-with-both-headers",
#       "name": "test_build_headers_with_both_headers",
#       "anchor": "function-test-build-headers-with-both-headers",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-cached-returns-cached-result",
#       "name": "test_interpret_response_cached_returns_cached_result",
#       "anchor": "function-test-interpret-response-cached-returns-cached-result",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-cached-missing-metadata-raises",
#       "name": "test_interpret_response_cached_missing_metadata_raises",
#       "anchor": "function-test-interpret-response-cached-missing-metadata-raises",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-modified-returns-modified-result",
#       "name": "test_interpret_response_modified_returns_modified_result",
#       "anchor": "function-test-interpret-response-modified-returns-modified-result",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-modified-extracts-headers",
#       "name": "test_interpret_response_modified_extracts_headers",
#       "anchor": "function-test-interpret-response-modified-extracts-headers",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-missing-metadata-lists-fields",
#       "name": "test_interpret_response_missing_metadata_lists_fields",
#       "anchor": "function-test-interpret-response-missing-metadata-lists-fields",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-headers-property",
#       "name": "test_build_headers_property",
#       "anchor": "function-test-build-headers-property",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-cached-property",
#       "name": "test_interpret_response_cached_property",
#       "anchor": "function-test-interpret-response-cached-property",
#       "kind": "function"
#     },
#     {
#       "id": "test-conditional-helper-rejects-negative-length",
#       "name": "test_conditional_helper_rejects_negative_length",
#       "anchor": "function-test-conditional-helper-rejects-negative-length",
#       "kind": "function"
#     },
#     {
#       "id": "test-interpret-response-requires-response-shape",
#       "name": "test_interpret_response_requires_response_shape",
#       "anchor": "function-test-interpret-response-requires-response-shape",
#       "kind": "function"
#     },
#     {
#       "id": "sequencedhandler",
#       "name": "_SequencedHandler",
#       "anchor": "class-sequencedhandler",
#       "kind": "class"
#     },
#     {
#       "id": "http-server",
#       "name": "http_server",
#       "anchor": "function-http-server",
#       "kind": "function"
#     },
#     {
#       "id": "make-artifact",
#       "name": "_make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "download",
#       "name": "_download",
#       "anchor": "function-download",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-candidate-retries-on-transient-errors",
#       "name": "test_download_candidate_retries_on_transient_errors",
#       "anchor": "function-test-download-candidate-retries-on-transient-errors",
#       "kind": "function"
#     },
#     {
#       "id": "test-retry-after-header-respected",
#       "name": "test_retry_after_header_respected",
#       "anchor": "function-test-retry-after-header-respected",
#       "kind": "function"
#     },
#     {
#       "id": "test-non-retryable-errors-do-not-retry",
#       "name": "test_non_retryable_errors_do_not_retry",
#       "anchor": "function-test-non-retryable-errors-do-not-retry",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-candidate-avoids-per-request-head",
#       "name": "test_download_candidate_avoids_per_request_head",
#       "anchor": "function-test-download-candidate-avoids-per-request-head",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-allows-pdf",
#       "name": "test_head_precheck_allows_pdf",
#       "anchor": "function-test-head-precheck-allows-pdf",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-rejects-html",
#       "name": "test_head_precheck_rejects_html",
#       "anchor": "function-test-head-precheck-rejects-html",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-degrades-to-get-pdf",
#       "name": "test_head_precheck_degrades_to_get_pdf",
#       "anchor": "function-test-head-precheck-degrades-to-get-pdf",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-degrades-to-get-html",
#       "name": "test_head_precheck_degrades_to_get_html",
#       "anchor": "function-test-head-precheck-degrades-to-get-html",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-returns-true-on-exception",
#       "name": "test_head_precheck_returns_true_on_exception",
#       "anchor": "function-test-head-precheck-returns-true-on-exception",
#       "kind": "function"
#     },
#     {
#       "id": "test-conditional-request-build-headers-requires-complete-metadata",
#       "name": "test_conditional_request_build_headers_requires_complete_metadata",
#       "anchor": "function-test-conditional-request-build-headers-requires-complete-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-conditional-request-build-headers-accepts-complete-metadata",
#       "name": "test_conditional_request_build_headers_accepts_complete_metadata",
#       "anchor": "function-test-conditional-request-build-headers-accepts-complete-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-retry-determinism-matches-request-with-retries",
#       "name": "test_retry_determinism_matches_request_with_retries",
#       "anchor": "function-test-retry-determinism-matches-request-with-retries",
#       "kind": "function"
#     },
#     {
#       "id": "mock-response",
#       "name": "_mock_response",
#       "anchor": "function-mock-response",
#       "kind": "function"
#     },
#     {
#       "id": "test-successful-request-no-retries",
#       "name": "test_successful_request_no_retries",
#       "anchor": "function-test-successful-request-no-retries",
#       "kind": "function"
#     },
#     {
#       "id": "test-transient-503-with-exponential-backoff",
#       "name": "test_transient_503_with_exponential_backoff",
#       "anchor": "function-test-transient-503-with-exponential-backoff",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-integer",
#       "name": "test_parse_retry_after_header_integer",
#       "anchor": "function-test-parse-retry-after-header-integer",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-http-date",
#       "name": "test_parse_retry_after_header_http_date",
#       "anchor": "function-test-parse-retry-after-header-http-date",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-invalid-date",
#       "name": "test_parse_retry_after_header_invalid_date",
#       "anchor": "function-test-parse-retry-after-header-invalid-date",
#       "kind": "function"
#     },
#     {
#       "id": "test-retry-after-header-overrides-backoff",
#       "name": "test_retry_after_header_overrides_backoff",
#       "anchor": "function-test-retry-after-header-overrides-backoff",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-exception-raises-after-retries",
#       "name": "test_request_exception_raises_after_retries",
#       "anchor": "function-test-request-exception-raises-after-retries",
#       "kind": "function"
#     },
#     {
#       "id": "test-timeout-retry-handling",
#       "name": "test_timeout_retry_handling",
#       "anchor": "function-test-timeout-retry-handling",
#       "kind": "function"
#     },
#     {
#       "id": "test-connection-error-retry-handling",
#       "name": "test_connection_error_retry_handling",
#       "anchor": "function-test-connection-error-retry-handling",
#       "kind": "function"
#     },
#     {
#       "id": "test-timeout-raises-after-exhaustion",
#       "name": "test_timeout_raises_after_exhaustion",
#       "anchor": "function-test-timeout-raises-after-exhaustion",
#       "kind": "function"
#     },
#     {
#       "id": "test-connection-error-raises-after-exhaustion",
#       "name": "test_connection_error_raises_after_exhaustion",
#       "anchor": "function-test-connection-error-raises-after-exhaustion",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-property",
#       "name": "test_parse_retry_after_header_property",
#       "anchor": "function-test-parse-retry-after-header-property",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-custom-retry-statuses",
#       "name": "test_request_with_custom_retry_statuses",
#       "anchor": "function-test-request-with-custom-retry-statuses",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-returns-after-exhausting-single-attempt",
#       "name": "test_request_returns_after_exhausting_single_attempt",
#       "anchor": "function-test-request-returns-after-exhausting-single-attempt",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-retries-rejects-negative-retries",
#       "name": "test_request_with_retries_rejects_negative_retries",
#       "anchor": "function-test-request-with-retries-rejects-negative-retries",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-retries-rejects-negative-backoff",
#       "name": "test_request_with_retries_rejects_negative_backoff",
#       "anchor": "function-test-request-with-retries-rejects-negative-backoff",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-retries-requires-method-and-url",
#       "name": "test_request_with_retries_requires_method_and_url",
#       "anchor": "function-test-request-with-retries-requires-method-and-url",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-retries-uses-method-fallback",
#       "name": "test_request_with_retries_uses_method_fallback",
#       "anchor": "function-test-request-with-retries-uses-method-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-retries-errors-when-no-callable-available",
#       "name": "test_request_with_retries_errors_when_no_callable_available",
#       "anchor": "function-test-request-with-retries-errors-when-no-callable-available",
#       "kind": "function"
#     },
#     {
#       "id": "test-retry-after-header-prefers-longer-delay",
#       "name": "test_retry_after_header_prefers_longer_delay",
#       "anchor": "function-test-retry-after-header-prefers-longer-delay",
#       "kind": "function"
#     },
#     {
#       "id": "test-respect-retry-after-false-skips-header",
#       "name": "test_respect_retry_after_false_skips_header",
#       "anchor": "function-test-respect-retry-after-false-skips-header",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-naive-datetime",
#       "name": "test_parse_retry_after_header_naive_datetime",
#       "anchor": "function-test-parse-retry-after-header-naive-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-handles-parse-errors",
#       "name": "test_parse_retry_after_header_handles_parse_errors",
#       "anchor": "function-test-parse-retry-after-header-handles-parse-errors",
#       "kind": "function"
#     },
#     {
#       "id": "test-parse-retry-after-header-returns-none-when-parser-returns-none",
#       "name": "test_parse_retry_after_header_returns_none_when_parser_returns_none",
#       "anchor": "function-test-parse-retry-after-header-returns-none-when-parser-returns-none",
#       "kind": "function"
#     },
#     {
#       "id": "fakeresponse",
#       "name": "FakeResponse",
#       "anchor": "class-fakeresponse",
#       "kind": "class"
#     },
#     {
#       "id": "make-artifact",
#       "name": "make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "stub-requests",
#       "name": "stub_requests",
#       "anchor": "function-stub-requests",
#       "kind": "function"
#     },
#     {
#       "id": "test-successful-pdf-download-populates-metadata",
#       "name": "test_successful_pdf_download_populates_metadata",
#       "anchor": "function-test-successful-pdf-download-populates-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-cached-response-preserves-prior-metadata",
#       "name": "test_cached_response_preserves_prior_metadata",
#       "anchor": "function-test-cached-response-preserves-prior-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-http-error-sets-metadata-to-none",
#       "name": "test_http_error_sets_metadata_to_none",
#       "anchor": "function-test-http-error-sets-metadata-to-none",
#       "kind": "function"
#     },
#     {
#       "id": "test-html-download-with-text-extraction",
#       "name": "test_html_download_with_text_extraction",
#       "anchor": "function-test-html-download-with-text-extraction",
#       "kind": "function"
#     },
#     {
#       "id": "test-dry-run-preserves-metadata-without-files",
#       "name": "test_dry_run_preserves_metadata_without_files",
#       "anchor": "function-test-dry-run-preserves-metadata-without-files",
#       "kind": "function"
#     },
#     {
#       "id": "test-small-pdf-detected-as-corrupt",
#       "name": "test_small_pdf_detected_as_corrupt",
#       "anchor": "function-test-small-pdf-detected-as-corrupt",
#       "kind": "function"
#     },
#     {
#       "id": "test-html-tail-in-pdf-marks-corruption",
#       "name": "test_html_tail_in_pdf_marks_corruption",
#       "anchor": "function-test-html-tail-in-pdf-marks-corruption",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-manifest-entry-includes-download-metadata",
#       "name": "test_build_manifest_entry_includes_download_metadata",
#       "anchor": "function-test-build-manifest-entry-includes-download-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "test-rfc5987-filename-suffix",
#       "name": "test_rfc5987_filename_suffix",
#       "anchor": "function-test-rfc5987-filename-suffix",
#       "kind": "function"
#     },
#     {
#       "id": "test-html-filename-suffix-from-disposition",
#       "name": "test_html_filename_suffix_from_disposition",
#       "anchor": "function-test-html-filename-suffix-from-disposition",
#       "kind": "function"
#     },
#     {
#       "id": "test-slugify-truncates-and-normalises",
#       "name": "test_slugify_truncates_and_normalises",
#       "anchor": "function-test-slugify-truncates-and-normalises",
#       "kind": "function"
#     },
#     {
#       "id": "test-classify-payload-variants",
#       "name": "test_classify_payload_variants",
#       "anchor": "function-test-classify-payload-variants",
#       "kind": "function"
#     },
#     {
#       "id": "test-collect-location-urls-dedupes-and-tracks-sources",
#       "name": "test_collect_location_urls_dedupes_and_tracks_sources",
#       "anchor": "function-test-collect-location-urls-dedupes-and-tracks-sources",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-doi",
#       "name": "test_normalize_doi",
#       "anchor": "function-test-normalize-doi",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-pmid",
#       "name": "test_normalize_pmid",
#       "anchor": "function-test-normalize-pmid",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-pmcid",
#       "name": "test_normalize_pmcid",
#       "anchor": "function-test-normalize-pmcid",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-arxiv",
#       "name": "test_normalize_arxiv",
#       "anchor": "function-test-normalize-arxiv",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-doi-with-https-prefix",
#       "name": "test_normalize_doi_with_https_prefix",
#       "anchor": "function-test-normalize-doi-with-https-prefix",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-doi-without-prefix",
#       "name": "test_normalize_doi_without_prefix",
#       "anchor": "function-test-normalize-doi-without-prefix",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-doi-with-whitespace",
#       "name": "test_normalize_doi_with_whitespace",
#       "anchor": "function-test-normalize-doi-with-whitespace",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-doi-none",
#       "name": "test_normalize_doi_none",
#       "anchor": "function-test-normalize-doi-none",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-doi-prefix-variants",
#       "name": "test_normalize_doi_prefix_variants",
#       "anchor": "function-test-normalize-doi-prefix-variants",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-pmcid-with-pmc-prefix",
#       "name": "test_normalize_pmcid_with_pmc_prefix",
#       "anchor": "function-test-normalize-pmcid-with-pmc-prefix",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-pmcid-without-prefix-adds-prefix",
#       "name": "test_normalize_pmcid_without_prefix_adds_prefix",
#       "anchor": "function-test-normalize-pmcid-without-prefix-adds-prefix",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-pmcid-lowercase",
#       "name": "test_normalize_pmcid_lowercase",
#       "anchor": "function-test-normalize-pmcid-lowercase",
#       "kind": "function"
#     },
#     {
#       "id": "test-strip-prefix-case-insensitive",
#       "name": "test_strip_prefix_case_insensitive",
#       "anchor": "function-test-strip-prefix-case-insensitive",
#       "kind": "function"
#     },
#     {
#       "id": "test-dedupe-preserves-order",
#       "name": "test_dedupe_preserves_order",
#       "anchor": "function-test-dedupe-preserves-order",
#       "kind": "function"
#     },
#     {
#       "id": "test-dedupe-filters-falsey-values",
#       "name": "test_dedupe_filters_falsey_values",
#       "anchor": "function-test-dedupe-filters-falsey-values",
#       "kind": "function"
#     },
#     {
#       "id": "test-dedupe-property",
#       "name": "test_dedupe_property",
#       "anchor": "function-test-dedupe-property",
#       "kind": "function"
#     },
#     {
#       "id": "make-artifact",
#       "name": "_make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "test-html-classification-overrides-misleading-content-type",
#       "name": "test_html_classification_overrides_misleading_content_type",
#       "anchor": "function-test-html-classification-overrides-misleading-content-type",
#       "kind": "function"
#     },
#     {
#       "id": "test-wayback-resolver-skips-unavailable-archives",
#       "name": "test_wayback_resolver_skips_unavailable_archives",
#       "anchor": "function-test-wayback-resolver-skips-unavailable-archives",
#       "kind": "function"
#     },
#     {
#       "id": "test-manifest-and-attempts-single-success",
#       "name": "test_manifest_and_attempts_single_success",
#       "anchor": "function-test-manifest-and-attempts-single-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-openalex-attempts-use-session-headers",
#       "name": "test_openalex_attempts_use_session_headers",
#       "anchor": "function-test-openalex-attempts-use-session-headers",
#       "kind": "function"
#     },
#     {
#       "id": "test-retry-budget-honours-max-attempts",
#       "name": "test_retry_budget_honours_max_attempts",
#       "anchor": "function-test-retry-budget-honours-max-attempts",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Consolidated content download networking tests."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import Mock, call, patch

import pytest
import requests

from DocsToKG.ContentDownload import cli as downloader
from DocsToKG.ContentDownload import download as download_impl
from DocsToKG.ContentDownload.cli import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    WorkArtifact,
    build_download_outcome,
    download_candidate,
    process_one_work,
)
from DocsToKG.ContentDownload.download import DownloadConfig
from DocsToKG.ContentDownload.core import (
    Classification,
    DownloadContext,
    ReasonCode,
    classify_payload,
    dedupe,
    normalize_arxiv,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid,
    normalize_url,
    strip_prefix,
)
from DocsToKG.ContentDownload.networking import (
    CachedResult,
    ConditionalRequestHelper,
    ModifiedResult,
    create_session,
    head_precheck,
    parse_retry_after_header,
    request_with_retries,
)
from DocsToKG.ContentDownload.pipeline import (
    AttemptRecord,
    DownloadOutcome,
    OpenAlexResolver,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
    WaybackResolver,
)
from DocsToKG.ContentDownload.telemetry import (
    JsonlSink,
    ManifestEntry,
    ManifestUrlIndex,
    RunTelemetry,
    SqliteSink,
    load_manifest_url_index,
)

# --- test_conditional_requests.py ---

try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

# --- test_conditional_requests.py ---

HAS_REQUESTS = find_spec("requests") is not None

# --- test_conditional_requests.py ---

HAS_PYALEX = find_spec("pyalex") is not None

# --- test_conditional_requests.py ---

given = hypothesis.given

# --- test_conditional_requests.py ---

if HAS_REQUESTS and HAS_PYALEX:
    from DocsToKG.ContentDownload.cli import (
        WorkArtifact,
        download_candidate,
    )
    from DocsToKG.ContentDownload.pipeline import DownloadOutcome
    from DocsToKG.ContentDownload.telemetry import ManifestEntry, build_manifest_entry

    class _DummyResponse:
        def __init__(self, status_code: int, headers: Dict[str, str]) -> None:
            self.status_code = status_code
            self.headers = headers

        def __enter__(self) -> "_DummyResponse":  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

        def iter_content(self, chunk_size: int):  # pragma: no cover - not needed for 304 path
            return iter(())

    class _DummyHead:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}

        def close(self) -> None:  # pragma: no cover - nothing to release
            return

    class _DummySession:
        def __init__(self, response: _DummyResponse) -> None:
            self._response = response

        def head(self, url: str, **kwargs: Any) -> _DummyHead:  # noqa: D401
            return _DummyHead()

        def request(self, method: str, url: str, **kwargs: Any) -> _DummyResponse:
            assert method.upper() == "GET"
            return self._response

    class _PdfStreamResponse:
        def __init__(self, chunks: Iterable[bytes]) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "application/pdf"}
            self._chunks = list(chunks)

        def __enter__(self) -> "_PdfStreamResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def iter_content(self, chunk_size: int):
            yield from self._chunks

        def close(self) -> None:  # pragma: no cover - nothing persistent to release
            return

    class _SequentialSession:
        def __init__(self, responses: Iterable[Any]) -> None:
            self._responses = iter(responses)
            self.calls: List[Tuple[str, str]] = []

        def head(self, url: str, **kwargs: Any) -> _DummyHead:
            return _DummyHead()

        def request(self, method: str, url: str, **kwargs: Any):
            method_upper = method.upper()
            assert method_upper == "GET"
            self.calls.append((method_upper, url))
            try:
                return next(self._responses)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise AssertionError(f"Unexpected extra request for {method_upper} {url}") from exc

    def _make_artifact(tmp_path: Path) -> WorkArtifact:
        pdf_dir = tmp_path / "pdfs"
        html_dir = tmp_path / "html"
        xml_dir = tmp_path / "xml"
        pdf_dir.mkdir()
        html_dir.mkdir()
        xml_dir.mkdir()
        return WorkArtifact(
            work_id="W-cond",
            title="Conditional",
            publication_year=2024,
            doi="10.1234/cond",
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="conditional",
            pdf_dir=pdf_dir,
            html_dir=html_dir,
            xml_dir=xml_dir,
        )

    def test_download_candidate_returns_cached(tmp_path: Path) -> None:
        artifact = _make_artifact(tmp_path)
        cached_bytes = b"%PDF-1.4\n%%EOF\n"
        previous_path = artifact.pdf_dir / "conditional.pdf"
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        previous_path.write_bytes(cached_bytes)
        previous_path_str = str(previous_path)
        previous = {
            "etag": '"etag"',
            "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "path": previous_path_str,
            "sha256": hashlib.sha256(cached_bytes).hexdigest(),
            "content_length": len(cached_bytes),
        }
        response = _DummyResponse(
            304,
            {"Content-Type": "application/pdf"},
        )
        session = _DummySession(response)
        outcome = download_candidate(
            session,
            artifact,
            "https://example.org/test.pdf",
            referer=None,
            timeout=5.0,
            context={"previous": {"https://example.org/test.pdf": previous}},
        )

        assert outcome.classification is Classification.CACHED
        assert outcome.path == previous_path_str
        assert outcome.sha256 == previous["sha256"]
        assert outcome.last_modified == previous["last_modified"]
        assert Path(previous_path_str).exists()

    def test_download_candidate_refetches_missing_cached_artifact(tmp_path: Path, caplog) -> None:
        artifact = _make_artifact(tmp_path)
        url = "https://example.org/missing.pdf"
        previous_path = artifact.pdf_dir / "conditional.pdf"
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        previous = {
            "etag": '"etag"',
            "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "path": str(previous_path),
            "sha256": "abc",
            "content_length": 123,
        }
        session = _SequentialSession(
            [
                _DummyResponse(304, {"Content-Type": "application/pdf"}),
                _PdfStreamResponse([b"%PDF-1.4\n", b"%%EOF\n"]),
            ]
        )

        caplog.set_level(logging.WARNING)
        outcome = download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context={"previous": {url: previous}},
        )

        assert outcome.classification is Classification.PDF
        assert outcome.http_status == 200
        assert outcome.path is not None
        assert Path(outcome.path).exists()
        downgrade_logs = [
            record
            for record in caplog.records
            if getattr(record, "reason", "") == "conditional-cache-invalid"
        ]
        assert len(downgrade_logs) == 1
        assert len(session.calls) == 2

    def test_download_candidate_refetches_when_cached_digest_mismatch(
        tmp_path: Path, caplog
    ) -> None:
        artifact = _make_artifact(tmp_path)
        url = "https://example.org/mismatch.pdf"
        previous_path = artifact.pdf_dir / "conditional.pdf"
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        old_data = b"not a real pdf payload"
        previous_path.write_bytes(old_data)
        previous = {
            "etag": '"etag"',
            "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "path": str(previous_path),
            "sha256": "deadbeef",
            "content_length": len(old_data),
        }
        new_chunks = [b"%PDF-1.5\n", b"%%EOF\n"]
        session = _SequentialSession(
            [
                _DummyResponse(304, {"Content-Type": "application/pdf"}),
                _PdfStreamResponse(new_chunks),
            ]
        )

        caplog.set_level(logging.WARNING)
        outcome = download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context={"previous": {url: previous}},
        )

        assert outcome.classification is Classification.PDF
        assert outcome.http_status == 200
        assert outcome.path is not None
        assert Path(outcome.path).exists()
        assert Path(outcome.path).read_bytes() == b"".join(new_chunks)
        downgrade_logs = [
            record
            for record in caplog.records
            if getattr(record, "reason", "") == "conditional-cache-invalid"
        ]
        assert len(downgrade_logs) == 1
        assert len(session.calls) == 2

    def test_download_candidate_handles_incomplete_resume_metadata(tmp_path: Path, caplog) -> None:
        artifact = _make_artifact(tmp_path)
        url = "https://example.org/test.pdf"

        class _StreamingResponse:
            def __init__(self) -> None:
                self.status_code = 200
                self.headers = {"Content-Type": "application/pdf"}

            def __enter__(self) -> "_StreamingResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                self.close()

            def iter_content(self, chunk_size: int = 1024):
                yield b"%PDF-1.4\n"
                yield b"%%EOF"

            def close(self) -> None:
                return None

        class _Session:
            def request(self, *, method: str, url: str, **kwargs: Any) -> _StreamingResponse:
                assert method == "GET"
                assert url == "https://example.org/test.pdf"
                return _StreamingResponse()

        context = {
            "previous": {
                url: {
                    "etag": '"etag"',
                    "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                }
            },
            "dry_run": False,
        }

        with caplog.at_level("WARNING", logger="DocsToKG.ContentDownload.network"):
            outcome = download_candidate(
                _Session(),
                artifact,
                url,
                referer=None,
                timeout=5.0,
                context=context,
            )

        assert outcome.classification is Classification.PDF
        assert "resume-metadata-incomplete" in caplog.text

    def test_download_candidate_blocks_disallowed_mime_policy(tmp_path: Path) -> None:
        artifact = _make_artifact(tmp_path)

        class _Response:
            def __init__(self) -> None:
                self.status_code = 200
                self.headers = {"Content-Type": "text/html"}

            def close(self) -> None:
                return None

        class _Session:
            def request(self, *, method: str, url: str, **kwargs: Any) -> _Response:
                assert method == "GET"
                return _Response()

        context = {
            "previous": {},
            "head_precheck_passed": True,
            "domain_content_rules": {"example.org": {"allowed_types": ("application/pdf",)}},
        }

        outcome = download_candidate(
            _Session(),
            artifact,
            "https://example.org/disallowed",
            referer=None,
            timeout=5.0,
            context=context,
    )

    assert outcome.classification is Classification.SKIPPED
    assert outcome.reason is ReasonCode.DOMAIN_DISALLOWED_MIME

def test_download_candidate_cleans_partial_on_stream_failure(tmp_path: Path, monkeypatch):
    artifact = _make_artifact(tmp_path)
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)

    class _Response:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "application/pdf"}

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def iter_content(self, chunk_size: int = 1024) -> Iterable[bytes]:
            yield b"%PDF-1.4\n"
            yield b"data"

        def close(self) -> None:
            return None

    class _Session:
        def request(self, *, method: str, url: str, **kwargs: Any) -> _Response:
            assert method == "GET"
            return _Response()

    def _failing_atomic_write(dest_path: Path, chunks: Iterable[bytes], **kwargs: Any) -> None:
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(b"partial")
        raise requests.exceptions.ChunkedEncodingError("stream interrupted")

    monkeypatch.setattr(download_impl, "atomic_write", _failing_atomic_write)

    outcome = download_candidate(
        _Session(),
        artifact,
        "https://example.org/bad.pdf",
        referer=None,
        timeout=5.0,
        context={"previous": {}},
    )

    assert outcome.classification is Classification.HTTP_ERROR
    assert not any(artifact.pdf_dir.glob("*.part"))


def test_skip_large_downloads_emit_voluntary_reason(tmp_path: Path, patcher) -> None:
        artifact = _make_artifact(tmp_path)
        url = "https://example.org/oversized.pdf"

        class _Response:
            def __init__(self) -> None:
                self.status_code = 200
                self.headers = {
                    "Content-Type": "application/pdf",
                    "Content-Length": str(25 * 1024 * 1024),
                }

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                self.close()

            def iter_content(self, chunk_size: int = 1024):
                yield b"%PDF-1.4\n"
                yield b"data"
                yield b"%%EOF"

            def close(self) -> None:
                return None

        class _Session:
            def request(self, *, method: str, url: str, **kwargs: Any) -> _Response:
                assert method == "GET"
                return _Response()

        # Ensure range resume code paths use patched request helper
        patcher.setattr(
            downloader, "request_with_retries", lambda *args, **kwargs: _Response(), raising=False
        )
        patcher.setattr(download_impl, "request_with_retries", lambda *args, **kwargs: _Response())

        context = {
            "previous": {},
            "skip_head_precheck": True,
            "skip_large_downloads": True,
            "size_warning_threshold": 5 * 1024 * 1024,
        }

        outcome = download_candidate(
            _Session(),
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context=context,
        )

        assert outcome.classification is Classification.SKIPPED
        assert outcome.reason is ReasonCode.SKIP_LARGE_DOWNLOAD
        assert outcome.reason_detail is not None
        assert outcome.path is None

    def test_build_download_outcome_accepts_small_pdf_with_head_pass(tmp_path: Path) -> None:
        artifact = _make_artifact(tmp_path)
        pdf_path = artifact.pdf_dir / "tiny.pdf"
        artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
        response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/pdf"})
        outcome = build_download_outcome(
            artifact=artifact,
            classification=Classification.PDF,
            dest_path=pdf_path,
            response=response,
            elapsed_ms=5.0,
            flagged_unknown=False,
            sha256="hash",
            content_length=pdf_path.stat().st_size,
            etag=None,
            last_modified=None,
            extracted_text_path=None,
            tail_bytes=b"%%EOF",
            dry_run=False,
            head_precheck_passed=True,
            min_pdf_bytes=DEFAULT_MIN_PDF_BYTES,
            tail_check_bytes=DEFAULT_TAIL_CHECK_BYTES,
            retry_after=None,
        )
        assert outcome.classification is Classification.PDF
        assert outcome.path == str(pdf_path)

    def test_build_download_outcome_rejects_small_pdf_without_head(tmp_path: Path) -> None:
        artifact = _make_artifact(tmp_path)
        pdf_path = artifact.pdf_dir / "tiny.pdf"
        artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
        response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/pdf"})
        outcome = build_download_outcome(
            artifact=artifact,
            classification="pdf",
            dest_path=pdf_path,
            response=response,
            elapsed_ms=5.0,
            flagged_unknown=False,
            sha256="hash",
            content_length=pdf_path.stat().st_size,
            etag=None,
            last_modified=None,
            extracted_text_path=None,
            tail_bytes=b"%%EOF",
            dry_run=False,
            head_precheck_passed=False,
            min_pdf_bytes=DEFAULT_MIN_PDF_BYTES,
            tail_check_bytes=DEFAULT_TAIL_CHECK_BYTES,
            retry_after=None,
        )
        assert outcome.classification is Classification.MISS
        assert outcome.path is None
        assert outcome.reason is ReasonCode.PDF_TOO_SMALL

    def test_manifest_entry_preserves_conditional_headers() -> None:
        outcome = DownloadOutcome(
            classification="pdf",
            path="/tmp/file.pdf",
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=12.3,
            sha256="deadbeef",
            content_length=42,
            etag='"tag"',
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        )
        artifact = WorkArtifact(
            work_id="W-cond",
            title="Conditional",
            publication_year=2024,
            doi="10.1234/cond",
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="conditional",
            pdf_dir=Path("/tmp"),
            html_dir=Path("/tmp"),
            xml_dir=Path("/tmp"),
        )
        entry = build_manifest_entry(
            artifact, "resolver", "https://example.org", outcome, [], dry_run=False
        )
        assert isinstance(entry, ManifestEntry)
        assert entry.etag == '"tag"'
        assert entry.last_modified == "Mon, 01 Jan 2024 00:00:00 GMT"


# --- test_conditional_requests.py ---


@dataclass
class _HelperResponse:
    status_code: int
    headers: Dict[str, str]


# --- test_conditional_requests.py ---


def _make_helper_response(
    status_code: int, headers: Optional[Dict[str, str]] = None
) -> _HelperResponse:
    return _HelperResponse(status_code=status_code, headers=headers or {})


# --- test_conditional_requests.py ---


def test_build_headers_empty_metadata() -> None:
    helper = ConditionalRequestHelper()

    assert helper.build_headers() == {}


# --- test_conditional_requests.py ---


def test_build_headers_etag_only() -> None:
    helper = ConditionalRequestHelper(prior_etag="abc123")

    assert helper.build_headers() == {}


# --- test_conditional_requests.py ---


def test_build_headers_last_modified_only() -> None:
    helper = ConditionalRequestHelper(prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT")

    assert helper.build_headers() == {}


# --- test_conditional_requests.py ---


def test_build_headers_with_both_headers() -> None:
    helper = ConditionalRequestHelper(
        prior_etag="abc123",
        prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
    )

    assert helper.build_headers() == {}


# --- test_conditional_requests.py ---


def test_interpret_response_cached_returns_cached_result(tmp_path: Path) -> None:
    cached_file = tmp_path / "file.pdf"
    cached_file.write_text("cached", encoding="utf-8")
    helper = ConditionalRequestHelper(
        prior_etag="abc123",
        prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
        prior_sha256="deadbeef",
        prior_content_length=1024,
        prior_path=str(cached_file),
    )
    response = _make_helper_response(304)

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, CachedResult)
    assert result.path == str(cached_file)
    assert result.sha256 == "deadbeef"
    assert result.content_length == 1024
    assert result.etag == "abc123"
    assert result.last_modified == "Wed, 21 Oct 2015 07:28:00 GMT"


# --- test_conditional_requests.py ---


def test_interpret_response_cached_missing_metadata_raises() -> None:
    helper = ConditionalRequestHelper(prior_etag="abc123")
    response = _make_helper_response(304)

    with pytest.raises(ValueError):
        helper.interpret_response(response)  # type: ignore[arg-type]


# --- test_conditional_requests.py ---


def test_interpret_response_modified_returns_modified_result() -> None:
    helper = ConditionalRequestHelper()
    response = _make_helper_response(200)

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, ModifiedResult)
    assert result.etag is None
    assert result.last_modified is None


# --- test_conditional_requests.py ---


def test_interpret_response_modified_extracts_headers() -> None:
    helper = ConditionalRequestHelper()
    response = _make_helper_response(
        200,
        {
            "ETag": '"xyz"',
            "Last-Modified": "Thu, 01 Jan 1970 00:00:00 GMT",
        },
    )

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, ModifiedResult)
    assert result.etag == '"xyz"'
    assert result.last_modified == "Thu, 01 Jan 1970 00:00:00 GMT"


# --- test_conditional_requests.py ---


def test_interpret_response_missing_metadata_lists_fields() -> None:
    helper = ConditionalRequestHelper(prior_etag="tag-only")
    response = _make_helper_response(304)

    with pytest.raises(ValueError) as excinfo:
        helper.interpret_response(response)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "path" in message
    assert "sha256" in message
    assert "content_length" in message


# --- test_conditional_requests.py ---


@given(
    etag=st.one_of(st.none(), st.text(min_size=1)),
    last_modified=st.one_of(st.none(), st.text(min_size=1)),
    complete=st.booleans(),
)
def test_build_headers_property(
    etag: Optional[str], last_modified: Optional[str], complete: bool
) -> None:
    helper = ConditionalRequestHelper(
        prior_etag=etag,
        prior_last_modified=last_modified,
        prior_sha256="sha" if complete else None,
        prior_content_length=42 if complete else None,
        prior_path="/tmp/cached.pdf" if complete else None,
    )
    headers = helper.build_headers()

    if etag or last_modified:
        if complete:
            if etag:
                assert headers["If-None-Match"] == etag
            else:
                assert "If-None-Match" not in headers
            if last_modified:
                assert headers["If-Modified-Since"] == last_modified
            else:
                assert "If-Modified-Since" not in headers
        else:
            assert headers == {}
    else:
        assert headers == {}


# --- test_conditional_requests.py ---


@given(
    path=st.text(min_size=1),
    sha=st.text(min_size=1),
    size=st.integers(min_value=1, max_value=10_000),
)
def test_interpret_response_cached_property(path: str, sha: str, size: int) -> None:
    helper = ConditionalRequestHelper(
        prior_path=path,
        prior_sha256=sha,
        prior_content_length=size,
        prior_etag="etag",
        prior_last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )
    response = _make_helper_response(304)

    with patch("DocsToKG.ContentDownload.networking.Path.exists", return_value=True):
        result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, CachedResult)
    assert result.path == path
    assert result.sha256 == sha
    assert result.content_length == size


# --- test_conditional_requests.py ---


def test_conditional_helper_rejects_negative_length() -> None:
    with pytest.raises(ValueError):
        ConditionalRequestHelper(prior_content_length=-1)


# --- test_conditional_requests.py ---


def test_interpret_response_requires_response_shape() -> None:
    helper = ConditionalRequestHelper()

    with pytest.raises(TypeError):
        helper.interpret_response(object())  # type: ignore[arg-type]


# --- test_download_retries.py ---

pytest.importorskip("requests")

# --- test_download_retries.py ---

pytest.importorskip("pyalex")


# --- test_download_retries.py ---


class _SequencedHandler(BaseHTTPRequestHandler):
    statuses: list[int] = []
    retry_after: int | None = None
    calls: list[int] = []
    head_calls: int = 0
    request_times: list[float] = []
    content: bytes = b"%PDF-1.4\n" + (b"0" * 2048) + b"\n%%EOF"

    def do_HEAD(self) -> None:  # noqa: D401 - HTTP handler signature
        self.__class__.head_calls += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: D401 - HTTP handler signature
        if not self.__class__.statuses:
            self.send_response(500)
            self.end_headers()
            return
        status = self.__class__.statuses.pop(0)
        self.__class__.request_times.append(time.monotonic())
        self.__class__.calls.append(status)
        self.send_response(status)
        if status == 429 and self.__class__.retry_after is not None:
            self.send_header("Retry-After", str(self.__class__.retry_after))
        self.send_header("Content-Type", "application/pdf")
        self.end_headers()
        if status == 200:
            self.wfile.write(self.__class__.content)

    def log_message(self, format: str, *args: object) -> None:  # noqa: D401
        return


# --- test_download_retries.py ---


@pytest.fixture
def http_server():
    handler = _SequencedHandler
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        handler.calls = []
        handler.statuses = []
        handler.retry_after = None
        handler.head_calls = 0
        handler.request_times = []
        yield handler, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


# --- test_download_retries.py ---


def _make_artifact(base_dir: Path) -> WorkArtifact:
    pdf_dir = base_dir / "pdfs"
    html_dir = base_dir / "html"
    xml_dir = base_dir / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()
    return WorkArtifact(
        work_id="W1",
        title="Test",
        publication_year=2024,
        doi="10.1234/test",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="test",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        xml_dir=xml_dir,
    )


# --- test_download_retries.py ---


def _download(
    url: str, tmp_path: Path
) -> Tuple[WorkArtifact, requests.Session, Dict[str, Any], DownloadOutcome]:
    artifact = _make_artifact(tmp_path)
    context = {
        "dry_run": False,
        "extract_html_text": False,
        "previous": {},
        "skip_head_precheck": True,
    }
    session = create_session({})
    return (
        artifact,
        session,
        context,
        download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context=context,
        ),
    )


# --- test_download_retries.py ---


@pytest.mark.parametrize("statuses", [[503, 503, 200]])
def test_download_candidate_retries_on_transient_errors(http_server, tmp_path, statuses):
    handler, server = http_server
    handler.statuses = list(statuses)
    handler.calls = []
    handler.retry_after = None
    url = f"http://127.0.0.1:{server.server_address[1]}/test.pdf"

    artifact, session, context, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification is Classification.PDF
        assert outcome.path is not None
        assert handler.calls == [503, 503, 200]
        assert Path(outcome.path).exists()
    finally:
        session.close()


# --- test_download_retries.py ---


def test_retry_after_header_respected(patcher, http_server, tmp_path):
    handler, server = http_server
    handler.statuses = [429, 200]
    handler.retry_after = 2
    handler.calls = []
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    patcher.setattr(time, "sleep", fake_sleep)
    url = f"http://127.0.0.1:{server.server_address[1]}/test.pdf"

    artifact, session, context, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification is Classification.PDF
        assert handler.calls == [429, 200]
        assert sleep_calls and sleep_calls[0] >= handler.retry_after
    finally:
        session.close()


# --- test_download_retries.py ---


def test_non_retryable_errors_do_not_retry(http_server, tmp_path):
    handler, server = http_server
    handler.statuses = [404]
    handler.retry_after = None
    url = f"http://127.0.0.1:{server.server_address[1]}/test.pdf"

    artifact = _make_artifact(tmp_path)
    context = {"dry_run": False, "extract_html_text": False, "previous": {}}
    session = create_session({})
    try:
        outcome = download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context=context,
        )
    finally:
        session.close()
    assert outcome.classification is Classification.HTTP_ERROR
    assert handler.calls == [404]


# --- test_download_retries.py ---


def test_download_candidate_avoids_per_request_head(http_server, tmp_path):
    """Ensure download path relies solely on GET without redundant HEAD calls."""

    handler, server = http_server
    handler.statuses = [200]
    handler.content = b"%PDF-1.4\n" + (b"1" * 4096) + b"\n%%EOF"
    url = f"http://127.0.0.1:{server.server_address[1]}/asset.pdf"

    _, session, _, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification is Classification.PDF
        assert handler.head_calls == 0
        assert handler.calls == [200]
    finally:
        session.close()


# --- test_head_precheck.py ---


def test_head_precheck_allows_pdf(patcher):
    head_response = Mock(status_code=200, headers={"Content-Type": "application/pdf"})
    head_response.close = Mock()

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.request_with_retries",
        lambda *args, **kwargs: head_response,
    )

    assert head_precheck(Mock(), "https://example.org/file.pdf", timeout=10.0)
    head_response.close.assert_called_once()


def test_head_precheck_rejects_html(patcher):
    head_response = Mock(status_code=200, headers={"Content-Type": "text/html"})
    head_response.close = Mock()

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.request_with_retries",
        lambda *args, **kwargs: head_response,
    )

    assert not head_precheck(Mock(), "https://example.org/page", timeout=10.0)
    head_response.close.assert_called_once()


@pytest.mark.parametrize("status", [405, 501])
def test_head_precheck_degrades_to_get_pdf(patcher, status):
    class _StreamResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "application/pdf"}
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def iter_content(self, chunk_size: int = 1024):
            yield b"%PDF"

        def close(self) -> None:
            self.closed = True

    head_response = Mock(status_code=status, headers={})
    head_response.close = Mock()
    stream_response = _StreamResponse()

    responses = [head_response, stream_response]

    def fake_request(*args, **kwargs):
        return responses.pop(0)

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.request_with_retries",
        fake_request,
    )

    assert head_precheck(Mock(), "https://example.org/pdf", timeout=10.0)
    assert stream_response.closed is True


@pytest.mark.parametrize("status", [405, 501])
def test_head_precheck_degrades_to_get_html(patcher, status):
    class _StreamResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def iter_content(self, chunk_size: int = 1024):
            yield b"<html></html>"

        def close(self) -> None:
            return None

    head_response = Mock(status_code=status, headers={})
    head_response.close = Mock()
    stream_response = _StreamResponse()

    responses = [head_response, stream_response]

    def fake_request(*args, **kwargs):
        return responses.pop(0)

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.request_with_retries",
        fake_request,
    )

    assert not head_precheck(Mock(), "https://example.org/html", timeout=10.0)


@pytest.mark.parametrize(
    "exc",
    [requests.Timeout("boom"), requests.ConnectionError("boom")],
)
def test_head_precheck_returns_true_on_exception(patcher, exc):
    patcher.setattr(
        "DocsToKG.ContentDownload.networking.request_with_retries",
        Mock(side_effect=exc),
    )

    assert head_precheck(Mock(), "https://example.org/err", timeout=5.0)


def test_conditional_request_build_headers_requires_complete_metadata(caplog) -> None:
    helper = ConditionalRequestHelper(prior_etag="abc", prior_last_modified=None)
    with caplog.at_level("WARNING", logger="DocsToKG.ContentDownload.network"):
        headers = helper.build_headers()
    assert headers == {}
    assert "resume-metadata-incomplete" in caplog.text


def test_conditional_request_build_headers_accepts_complete_metadata() -> None:
    helper = ConditionalRequestHelper(
        prior_etag="etag",
        prior_last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        prior_sha256="deadbeef",
        prior_content_length=1024,
        prior_path="/tmp/file.pdf",
    )
    headers = helper.build_headers()
    assert headers == {
        "If-None-Match": "etag",
        "If-Modified-Since": "Mon, 01 Jan 2024 00:00:00 GMT",
    }


# --- test_download_retries.py ---


def test_retry_determinism_matches_request_with_retries(patcher, http_server, tmp_path):
    """Verify retry budget and timing are governed exclusively by the helper."""

    handler, server = http_server
    handler.statuses = [429, 429, 200]
    url = f"http://127.0.0.1:{server.server_address[1]}/rate-limited.pdf"

    patcher.setattr("DocsToKG.ContentDownload.networking.random.random", lambda: 0.0)

    sleep_durations: list[float] = []

    def _capture_sleep(delay: float) -> None:
        sleep_durations.append(delay)

    patcher.setattr("DocsToKG.ContentDownload.networking.time.sleep", _capture_sleep)

    _, session, _, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification is Classification.PDF
        assert handler.calls == [429, 429, 200]
        assert handler.head_calls == 0
        # Ensure exactly max_retries + 1 attempts were issued (default helper budget)
        assert len(handler.request_times) == 3
        assert sleep_durations == [0.75, 1.5]
    finally:
        session.close()


# --- test_http_retry.py ---

try:  # pragma: no cover - dependency optional in CI
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - skip if requests missing
    requests = pytest.importorskip("requests")  # type: ignore

# --- test_http_retry.py ---

try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

# --- test_http_retry.py ---

given = hypothesis.given


# --- test_http_retry.py ---


def _mock_response(status: int, headers: Optional[Dict[str, str]] = None) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    return response


# --- test_http_retry.py ---


def test_successful_request_no_retries():
    """Verify successful request completes immediately without retries."""

    session = Mock(spec=requests.Session)
    response = _mock_response(200)
    session.request.return_value = response

    result = request_with_retries(session, "GET", "https://example.org/test")

    assert result is response
    session.request.assert_called_once_with(method="GET", url="https://example.org/test")


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_transient_503_with_exponential_backoff(mock_sleep: Mock, _: Mock) -> None:
    """Verify exponential backoff timing for transient 503 errors."""

    session = Mock(spec=requests.Session)
    response_503 = _mock_response(503, headers={})
    response_200 = _mock_response(200)
    session.request.side_effect = [response_503, response_503, response_200]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        max_retries=3,
        backoff_factor=0.5,
    )

    assert result is response_200
    assert session.request.call_count == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0)]


# --- test_http_retry.py ---


def test_parse_retry_after_header_integer() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "5"}

    assert parse_retry_after_header(response) == 5.0


# --- test_http_retry.py ---


def test_parse_retry_after_header_http_date() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    header_value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response = requests.Response()
    response.headers = {"Retry-After": header_value}

    wait = parse_retry_after_header(response)
    assert wait is not None
    assert 0.0 < wait <= 30.0


# --- test_http_retry.py ---


def test_parse_retry_after_header_invalid_date() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Thu, 32 Foo 2024 00:00:00 GMT"}

    assert parse_retry_after_header(response) is None


def test_parse_retry_after_header_rejects_non_positive_seconds() -> None:
    response_zero = requests.Response()
    response_zero.headers = {"Retry-After": "0"}
    response_negative = requests.Response()
    response_negative.headers = {"Retry-After": "-1"}

    assert parse_retry_after_header(response_zero) is None
    assert parse_retry_after_header(response_negative) is None


def test_parse_retry_after_header_rejects_nan() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "NaN"}

    assert parse_retry_after_header(response) is None


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_retry_after_header_overrides_backoff(mock_sleep: Mock, _: Mock) -> None:
    session = Mock(spec=requests.Session)
    retry_headers = {"Retry-After": "10"}
    response_retry = _mock_response(429, headers=retry_headers)
    response_success = _mock_response(200)
    session.request.side_effect = [response_retry, response_success]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        backoff_factor=0.1,
        max_retries=2,
    )

    assert result is response_success
    assert mock_sleep.call_args_list == [call(10.0)]


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_request_exception_raises_after_retries(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    error = requests.RequestException("boom")
    session.request.side_effect = error

    with pytest.raises(requests.RequestException):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=1)

    assert mock_sleep.call_count == 1
    assert session.request.call_count == 2


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_timeout_retry_handling(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    session.request.side_effect = [requests.Timeout("slow"), _mock_response(200)]

    result = request_with_retries(session, "GET", "https://example.org/timeout", max_retries=1)

    assert result.status_code == 200
    assert mock_sleep.call_count == 1


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_connection_error_retry_handling(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    session.request.side_effect = [requests.ConnectionError("down"), _mock_response(200)]

    result = request_with_retries(session, "GET", "https://example.org/conn", max_retries=1)

    assert result.status_code == 200
    assert mock_sleep.call_count == 1


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_timeout_raises_after_exhaustion(mock_sleep: Mock) -> None:
    """Ensure timeout retries raise after exhausting the retry budget."""

    session = Mock(spec=requests.Session)
    session.request.side_effect = requests.Timeout("slow")

    with pytest.raises(requests.Timeout):
        request_with_retries(session, "GET", "https://example.org/timeout", max_retries=1)

    # Only the non-terminal attempt sleeps before re-raising on the final attempt.
    assert mock_sleep.call_count == 1


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_connection_error_raises_after_exhaustion(mock_sleep: Mock) -> None:
    """Ensure connection errors propagate when retries are exhausted."""

    session = Mock(spec=requests.Session)
    session.request.side_effect = requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        request_with_retries(session, "GET", "https://example.org/conn", max_retries=1)

    assert mock_sleep.call_count == 1


# --- test_http_retry.py ---


@given(st.text())
def test_parse_retry_after_header_property(value: str) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": value}

    result = parse_retry_after_header(response)

    if result is not None:
        assert result > 0.0
        assert math.isfinite(result)


# --- test_http_retry.py ---


def test_request_with_custom_retry_statuses() -> None:
    session = Mock(spec=requests.Session)
    failing = _mock_response(404)
    success = _mock_response(200)
    session.request.side_effect = [failing, success]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        retry_statuses={404},
        max_retries=1,
    )

    assert result is success
    assert session.request.call_count == 2


# --- test_http_retry.py ---


def test_request_returns_after_exhausting_single_attempt() -> None:
    session = Mock(spec=requests.Session)
    retry_response = _mock_response(503)
    session.request.return_value = retry_response

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        max_retries=0,
    )

    assert result is retry_response


# --- test_http_retry.py ---


def test_request_with_retries_rejects_negative_retries() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=-1)


# --- test_http_retry.py ---


def test_request_with_retries_rejects_negative_backoff() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "https://example.org/test", backoff_factor=-0.1)


# --- test_http_retry.py ---


def test_request_with_retries_requires_method_and_url() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "", "https://example.org/test")

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "")


# --- test_http_retry.py ---


def test_request_with_retries_uses_method_fallback() -> None:
    session = Mock(spec=requests.Session)
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.headers = {}
    session.request.return_value = response

    response = request_with_retries(session, "GET", "https://example.org/fallback")

    assert response.status_code == 200
    session.request.assert_called_once_with(method="GET", url="https://example.org/fallback")


# --- test_http_retry.py ---


def test_request_with_retries_errors_when_no_callable_available() -> None:
    class _MinimalSession:
        pass

    with pytest.raises(AttributeError):
        request_with_retries(_MinimalSession(), "GET", "https://example.org/fail")


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
def test_retry_after_header_prefers_longer_delay(mock_sleep: Mock) -> None:
    """Verify Retry-After header longer than backoff takes precedence."""

    session = Mock(spec=requests.Session)

    retry_response = requests.Response()
    retry_response.status_code = 429
    retry_response.headers = {"Retry-After": "4"}

    success_response = requests.Response()
    success_response.status_code = 200
    success_response.headers = {}

    session.request.side_effect = [retry_response, success_response]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/with-retry-after",
        backoff_factor=0.1,
        max_retries=2,
    )

    assert result.status_code == 200
    mock_sleep.assert_called_once()
    sleep_arg = mock_sleep.call_args[0][0]
    assert pytest.approx(sleep_arg, rel=0.01) == 4.0


# --- test_http_retry.py ---


@patch("DocsToKG.ContentDownload.networking.time.sleep")
@patch("DocsToKG.ContentDownload.networking.parse_retry_after_header")
def test_respect_retry_after_false_skips_header(mock_parse: Mock, mock_sleep: Mock) -> None:
    """Ensure disabling respect_retry_after bypasses header parsing."""

    session = Mock(spec=requests.Session)
    retry_response = _mock_response(503)
    success_response = _mock_response(200)
    session.request.side_effect = [retry_response, success_response]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/no-retry-after",
        respect_retry_after=False,
        max_retries=1,
        backoff_factor=0.1,
    )

    assert result is success_response
    mock_parse.assert_not_called()
    mock_sleep.assert_called_once()


# --- test_http_retry.py ---


def test_parse_retry_after_header_naive_datetime() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=45)
    header_value = future.replace(tzinfo=None).strftime("%a, %d %b %Y %H:%M:%S")
    response = requests.Response()
    response.headers = {"Retry-After": header_value}

    wait = parse_retry_after_header(response)
    assert wait is not None
    assert 0.0 < wait <= 45.0


# --- test_http_retry.py ---


def test_parse_retry_after_header_handles_parse_errors(patcher) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00 GMT"}

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.parsedate_to_datetime",
        Mock(side_effect=TypeError("boom")),
    )

    assert parse_retry_after_header(response) is None


# --- test_http_retry.py ---


def test_parse_retry_after_header_returns_none_when_parser_returns_none(patcher) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00 GMT"}

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.parsedate_to_datetime",
        Mock(return_value=None),
    )

    assert parse_retry_after_header(response) is None


# --- test_download_outcomes.py ---

if "pyalex" not in sys.modules:
    pyalex_stub = types.ModuleType("pyalex")
    pyalex_stub.Topics = object
    pyalex_stub.Works = object
    config_stub = types.ModuleType("pyalex.config")
    config_stub.mailto = None
    pyalex_stub.config = config_stub
    sys.modules["pyalex"] = pyalex_stub
    sys.modules["pyalex.config"] = config_stub


# --- test_download_outcomes.py ---


class FakeResponse:
    def __init__(self, status_code: int, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def iter_content(self, chunk_size: int):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        pass


# --- test_download_outcomes.py ---


def make_artifact(tmp_path: Path) -> downloader.WorkArtifact:
    artifact = downloader.WorkArtifact(
        work_id="W-DOWNLOAD",
        title="Outcome Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="outcome-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    artifact.xml_dir.mkdir(parents=True, exist_ok=True)
    return artifact


# --- test_download_outcomes.py ---


def stub_requests(
    patcher, mapping: Dict[Tuple[str, str], Callable[[], FakeResponse] | FakeResponse]
):
    def fake_request(session, method, url, **kwargs):
        key = (method.upper(), url)
        if key not in mapping:
            raise AssertionError(f"Unexpected request {key}")
        response = mapping[key]
        return response() if callable(response) else response

    patcher.setattr(downloader, "request_with_retries", fake_request, raising=False)
    patcher.setattr(download_impl, "request_with_retries", fake_request)


# --- test_download_outcomes.py ---


def test_successful_pdf_download_populates_metadata(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    pdf_bytes = b"%PDF-1.4\n" + (b"x" * 2048) + b"\n%%EOF"
    expected_sha = hashlib.sha256(pdf_bytes).hexdigest()

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"etag-123"',
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            },
            chunks=[pdf_bytes],
        ),
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"skip_head_precheck": True},
    )

    assert outcome.classification is Classification.PDF
    assert outcome.path is not None
    stored = Path(outcome.path)
    assert stored.exists()
    assert outcome.sha256 == expected_sha
    assert outcome.content_length == stored.stat().st_size
    assert outcome.etag == '"etag-123"'
    assert outcome.last_modified == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert outcome.reason is None
    assert outcome.extracted_text_path is None
    rehashed = hashlib.sha256(stored.read_bytes()).hexdigest()
    assert rehashed == expected_sha


# --- test_download_outcomes.py ---


def test_cached_response_preserves_prior_metadata(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    cached_file = artifact.pdf_dir / "cached.pdf"
    cached_bytes = b"%PDF-1.4\n%EOF\n"
    cached_file.write_bytes(cached_bytes)
    cached_sha = hashlib.sha256(cached_bytes).hexdigest()
    cached_mtime = cached_file.stat().st_mtime_ns
    context = {
        "previous": {
            url: {
                "path": str(cached_file),
                "sha256": cached_sha,
                "content_length": len(cached_bytes),
                "etag": '"etag-cached"',
                "last_modified": "Tue, 02 Jan 2024 00:00:00 GMT",
                "path_mtime_ns": cached_mtime,
                "mtime_ns": cached_mtime,
            }
        }
    }

    mapping = {
        ("GET", url): lambda: FakeResponse(
            304,
            headers={"Content-Type": "application/pdf"},
        ),
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session, artifact, url, None, timeout=10.0, context=context
    )

    assert outcome.classification is Classification.CACHED
    assert outcome.path == str(cached_file)
    assert outcome.sha256 == cached_sha
    assert outcome.content_length == len(cached_bytes)
    assert outcome.etag == '"etag-cached"'
    assert outcome.last_modified == "Tue, 02 Jan 2024 00:00:00 GMT"
    assert outcome.reason is ReasonCode.CONDITIONAL_NOT_MODIFIED
    assert outcome.metadata.get("cache_validation_mode") == "fast_path"


# --- test_download_outcomes.py ---


def test_cache_validation_fast_path_skips_digest(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/cached-fast.pdf"
    cached_path = artifact.pdf_dir / "cached.pdf"
    cached_bytes = b"%PDF-1.4\nstream\n%%EOF\n"
    cached_path.write_bytes(cached_bytes)
    cached_sha = hashlib.sha256(cached_bytes).hexdigest()
    cached_mtime = cached_path.stat().st_mtime_ns

    mapping = {("GET", url): lambda: FakeResponse(304, headers={"Content-Type": "application/pdf"})}
    stub_requests(patcher, mapping)

    def raise_if_digest(signature):
        raise AssertionError("Digest computation should not run for fast-path validation")

    patcher.setattr(download_impl, "_cached_sha256", raise_if_digest)

    context = {
        "previous": {
            url: {
                "path": str(cached_path),
                "sha256": cached_sha,
                "content_length": len(cached_bytes),
                "etag": '"etag-fast"',
                "last_modified": "Wed, 03 Jan 2024 00:00:00 GMT",
                "path_mtime_ns": cached_mtime,
                "mtime_ns": cached_mtime,
            }
        }
    }

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context=context,
    )

    assert outcome.classification is Classification.CACHED
    assert outcome.reason is ReasonCode.CONDITIONAL_NOT_MODIFIED
    assert outcome.metadata.get("cache_validation_mode") == "fast_path"


def test_cache_validation_forced_digest(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/cached-digest.pdf"
    cached_path = artifact.pdf_dir / "cached-digest.pdf"
    cached_bytes = b"%PDF-1.4\nq\n%%EOF\n"
    cached_path.write_bytes(cached_bytes)
    cached_sha = hashlib.sha256(cached_bytes).hexdigest()
    cached_mtime = cached_path.stat().st_mtime_ns

    mapping = {("GET", url): lambda: FakeResponse(304, headers={"Content-Type": "application/pdf"})}
    stub_requests(patcher, mapping)

    digest_calls: List[Tuple[str, int, int]] = []

    def record_digest(signature: Tuple[str, int, int]) -> Optional[str]:
        digest_calls.append(signature)
        return cached_sha

    patcher.setattr(download_impl, "_cached_sha256", record_digest)

    context = {
        "previous": {
            url: {
                "path": str(cached_path),
                "sha256": cached_sha,
                "content_length": len(cached_bytes),
                "etag": '"etag-digest"',
                "last_modified": "Thu, 04 Jan 2024 00:00:00 GMT",
                "path_mtime_ns": cached_mtime,
                "mtime_ns": cached_mtime,
            }
        },
        "verify_cache_digest": True,
    }

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context=context,
    )

    assert digest_calls
    assert outcome.classification is Classification.CACHED
    assert outcome.metadata.get("cache_validation_mode") == "digest"


def test_cache_validation_digest_mismatch_triggers_refetch(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/cached-mismatch.pdf"
    cached_path = artifact.pdf_dir / "cached-mismatch.pdf"
    cached_bytes = b"%PDF-1.4\n" + (b"0" * 2048) + b"\n%%EOF\n"
    cached_path.write_bytes(cached_bytes)
    cached_sha = hashlib.sha256(cached_bytes).hexdigest()
    cached_mtime = cached_path.stat().st_mtime_ns

    new_bytes = b"%PDF-1.4\n" + (b"1" * 2048) + b"\n%%EOF\n"

    class _StreamingResponse:
        def __init__(self, status_code: int, headers: Dict[str, str], chunks: List[bytes]):
            self.status_code = status_code
            self.headers = headers
            self._chunks = list(chunks)

        def __enter__(self) -> "_StreamingResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def iter_content(self, chunk_size: int = 1024):
            for chunk in self._chunks:
                yield chunk

        def close(self) -> None:
            return None

    responses: List[_StreamingResponse] = [
        _StreamingResponse(304, {"Content-Type": "application/pdf"}, []),
        _StreamingResponse(
            200,
            {"Content-Type": "application/pdf"},
            [new_bytes],
        ),
    ]

    def sequenced_request(*args, **kwargs):
        if not responses:
            raise AssertionError("Unexpected additional request")
        return responses.pop(0)

    patcher.setattr(download_impl, "request_with_retries", sequenced_request)
    patcher.setattr(downloader, "request_with_retries", sequenced_request, raising=False)

    def mismatched_digest(signature: Tuple[str, int, int]) -> Optional[str]:
        return "deadbeef"

    patcher.setattr(download_impl, "_cached_sha256", mismatched_digest)

    recorded_mtime = max(cached_mtime - 1, 0)
    context = {
        "previous": {
            url: {
                "path": str(cached_path),
                "sha256": cached_sha,
                "content_length": len(cached_bytes),
                "etag": '"etag-mismatch"',
                "last_modified": "Fri, 05 Jan 2024 00:00:00 GMT",
                "path_mtime_ns": recorded_mtime,
                "mtime_ns": recorded_mtime,
            }
        }
    }

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context=context,
    )

    assert outcome.classification is Classification.PDF
    assert outcome.reason is None
    assert outcome.path is not None
    downloaded = Path(outcome.path).read_bytes()
    assert downloaded == new_bytes


def test_cache_validation_digest_cache_reuse(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/cached-cache.pdf"
    cached_path = artifact.pdf_dir / "cached-cache.pdf"
    cached_bytes = b"%PDF-1.4\n2\n%%EOF\n"
    cached_path.write_bytes(cached_bytes)
    cached_sha = hashlib.sha256(cached_bytes).hexdigest()
    cached_mtime = cached_path.stat().st_mtime_ns

    mapping = {("GET", url): lambda: FakeResponse(304, headers={"Content-Type": "application/pdf"})}
    stub_requests(patcher, mapping)

    download_impl._cached_sha256.cache_clear()

    context = {
        "previous": {
            url: {
                "path": str(cached_path),
                "sha256": cached_sha,
                "content_length": len(cached_bytes),
                "etag": '"etag-cache"',
                "last_modified": "Sat, 06 Jan 2024 00:00:00 GMT",
                "path_mtime_ns": cached_mtime,
                "mtime_ns": cached_mtime,
            }
        },
        "verify_cache_digest": True,
    }

    session = requests.Session()
    downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context=context,
    )
    first_info = download_impl._cached_sha256.cache_info()

    downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context=context,
    )
    second_info = download_impl._cached_sha256.cache_info()

    assert second_info.hits >= first_info.hits + 1

    download_impl._cached_sha256.cache_clear()


def test_http_error_sets_metadata_to_none(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"

    mapping = {
        ("GET", url): lambda: FakeResponse(404, headers={"Content-Type": "text/html"}),
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"min_pdf_bytes": 1, "previous": {}, "extract_html_text": True},
    )

    assert outcome.classification is Classification.HTTP_ERROR
    assert outcome.path is None
    assert outcome.sha256 is None
    assert outcome.content_length is None
    assert outcome.etag is None
    assert outcome.last_modified is None
    assert outcome.reason is None


# --- test_download_outcomes.py ---


def test_html_download_with_text_extraction(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/page.html"
    html_bytes = b"<!DOCTYPE html><html><body><p>Hello</p></body></html>"

    html_extractor = types.SimpleNamespace(extract=lambda text: "Hello")
    patcher.setitem(sys.modules, "trafilatura", html_extractor)

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "text/html",
                "ETag": '"etag-html"',
                "Last-Modified": "Wed, 03 Jan 2024 00:00:00 GMT",
            },
            chunks=[html_bytes],
        ),
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"extract_html_text": True},
    )

    assert outcome.classification is Classification.HTML
    assert outcome.path is not None and outcome.path.endswith(".html")
    assert outcome.extracted_text_path is not None
    extracted = Path(outcome.extracted_text_path)
    assert extracted.exists()
    assert extracted.read_text(encoding="utf-8") == "Hello"
    assert outcome.sha256 is not None
    assert outcome.etag == '"etag-html"'
    assert outcome.last_modified == "Wed, 03 Jan 2024 00:00:00 GMT"


# --- test_download_outcomes.py ---


def test_dry_run_preserves_metadata_without_files(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    pdf_bytes = b"%PDF-1.4\n" + (b"y" * 2048) + b"\n%%EOF"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"etag-dry"',
                "Last-Modified": "Thu, 04 Jan 2024 00:00:00 GMT",
            },
            chunks=[pdf_bytes],
        ),
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"dry_run": True},
    )

    assert outcome.classification is Classification.PDF
    assert outcome.path is None
    assert outcome.sha256 is None
    assert outcome.content_length is None
    assert outcome.extracted_text_path is None
    assert outcome.etag == '"etag-dry"'
    assert outcome.last_modified == "Thu, 04 Jan 2024 00:00:00 GMT"


# --- test_download_outcomes.py ---


def test_small_pdf_detected_as_corrupt(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/tiny.pdf"
    tiny_pdf = b"%PDF-1.4\n1 0 obj<<>>\nendobj\n%%EOF"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={"Content-Type": "application/pdf"},
            chunks=[tiny_pdf],
        )
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"skip_head_precheck": True},
    )

    assert outcome.classification is Classification.MISS
    assert outcome.path is None
    assert outcome.reason is ReasonCode.PDF_TOO_SMALL
    assert not any(artifact.pdf_dir.glob("*.pdf"))


# --- test_download_outcomes.py ---


def test_html_tail_in_pdf_marks_corruption(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/error.pdf"
    payload = b"%PDF-1.4\nstream\n<html>Error page</html>"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={"Content-Type": "application/pdf"},
            chunks=[payload],
        )
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"min_pdf_bytes": 1, "previous": {}, "extract_html_text": True},
    )

    assert outcome.classification is Classification.MISS
    assert outcome.path is None
    assert outcome.reason is ReasonCode.HTML_TAIL_DETECTED
    assert not any(artifact.pdf_dir.glob("*.pdf"))


# --- test_download_outcomes.py ---


def test_build_manifest_entry_includes_download_metadata(tmp_path):
    artifact = make_artifact(tmp_path)
    download_path = str(artifact.pdf_dir / "saved.pdf")
    outcome = DownloadOutcome(
        classification="pdf",
        path=download_path,
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=150.0,
        sha256="abc123",
        content_length=4096,
        etag='"etag-manifest"',
        last_modified="Fri, 05 Jan 2024 00:00:00 GMT",
        extracted_text_path=str(artifact.html_dir / "saved.txt"),
    )

    entry = build_manifest_entry(
        artifact,
        "figshare",
        "https://example.org/paper.pdf",
        outcome,
        html_paths=["/tmp/example.html"],
        dry_run=False,
        run_id="test-run",
    )

    assert entry.sha256 == "abc123"
    assert entry.content_length == 4096
    assert entry.etag == '"etag-manifest"'
    assert entry.last_modified == "Fri, 05 Jan 2024 00:00:00 GMT"
    assert entry.extracted_text_path == str(artifact.html_dir / "saved.txt")
    assert entry.run_id == "test-run"


def test_manifest_entry_serializes_null_reason(tmp_path):
    artifact = make_artifact(tmp_path)
    saved_path = artifact.pdf_dir / "saved.pdf"
    pdf_bytes = b"%PDF-1.4\n" + (b"y" * 64) + b"\n%%EOF"
    saved_path.write_bytes(pdf_bytes)
    outcome = DownloadOutcome(
        classification="pdf",
        path=str(saved_path),
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=42.0,
        reason=None,
        sha256="abc123",
        content_length=len(pdf_bytes),
        etag=None,
        last_modified=None,
        extracted_text_path=None,
    )

    entry = build_manifest_entry(
        artifact,
        "figshare",
        "https://example.org/paper.pdf",
        outcome,
        html_paths=[],
        dry_run=False,
        run_id="null-reason-test",
    )

    assert entry.reason is None

    db_path = tmp_path / "telemetry.sqlite3"
    with SqliteSink(db_path) as sink:
        sink.log_manifest(entry)

    conn = sqlite3.connect(db_path)
    try:
        reason_value = conn.execute("SELECT reason FROM manifests").fetchone()[0]
        assert reason_value is None
    finally:
        conn.close()


def test_resume_requested_triggers_full_download(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/resume.pdf"
    dest_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
    partial_path = dest_path.with_suffix(".pdf.part")
    full_bytes = b"%PDF-1.4\n" + (b"a" * 4096) + b"\n%%EOF"
    partial_path.write_bytes(full_bytes[:1024])
    expected_sha = hashlib.sha256(full_bytes).hexdigest()

    captured_headers: Dict[str, str] = {}

    def fake_request(session, method, req_url, **kwargs):
        captured_headers.clear()
        captured_headers.update(dict(kwargs.get("headers") or {}))
        return FakeResponse(
            200,
            headers={"Content-Type": "application/pdf"},
            chunks=[full_bytes[:2048], full_bytes[2048:]],
        )

    patcher.setattr(downloader, "request_with_retries", fake_request, raising=False)
    patcher.setattr(download_impl, "request_with_retries", fake_request)

    ctx = DownloadContext.from_mapping(
        {
            "skip_head_precheck": True,
            "enable_range_resume": True,
            "previous": {
                normalize_url(url): {
                    "path": str(dest_path),
                    "content_length": len(full_bytes),
                }
            },
        }
    )

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context=ctx,
    )

    assert "Range" not in captured_headers
    assert ctx.extra.get("resume_disabled") is True
    assert outcome.classification is Classification.PDF
    assert outcome.reason is None
    assert outcome.path is not None
    stored = Path(outcome.path)
    assert stored.read_bytes() == full_bytes
    assert outcome.sha256 == expected_sha


def test_pipeline_records_resume_disabled_metadata(tmp_path):
    class RecordingLogger:
        def __init__(self) -> None:
            self.attempts: List[AttemptRecord] = []

        def log_attempt(self, record: AttemptRecord, *, timestamp: Optional[str] = None) -> None:
            self.attempts.append(record)

        def log_manifest(self, entry: ManifestEntry) -> None:  # pragma: no cover - not used
            return None

        def log_summary(self, summary: Dict[str, Any]) -> None:  # pragma: no cover - not used
            return None

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterable[ResolverResult]:
            yield ResolverResult(url="https://example.org/stub.pdf")

    def stub_download(
        session: requests.Session,
        artifact: WorkArtifact,
        url: str,
        referer: Optional[str],
        timeout: float,
        context: DownloadContext,
        **kwargs: Any,
    ) -> DownloadOutcome:
        context.extra["resume_disabled"] = True
        fake_path = tmp_path / "stub.pdf"
        fake_path.write_bytes(b"%PDF-1.4\n%%EOF")
        outcome = DownloadOutcome(
            classification=Classification.PDF,
            path=str(fake_path),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=5.0,
        )
        outcome.metadata["resume_disabled"] = True
        return outcome

    logger = RecordingLogger()
    config = ResolverConfig()
    config.resolver_order = ["stub"]
    config.resolver_toggles["stub"] = True
    config.enable_head_precheck = False
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline(
        resolvers=[StubResolver()],
        config=config,
        download_func=stub_download,
        logger=logger,
        metrics=metrics,
    )

    artifact = make_artifact(tmp_path)
    session = requests.Session()
    pipeline.run(
        session,
        artifact,
        context=DownloadContext(skip_head_precheck=True, dry_run=False),
    )

    assert logger.attempts
    assert logger.attempts[0].metadata.get("resume_disabled") is True


# --- test_download_outcomes.py ---


def test_rfc5987_filename_suffix(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/no-extension"
    pdf_bytes = b"%PDF-1.4\n" + (b"z" * 2048) + b"\n%%EOF"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": "attachment; filename*=UTF-8''paper%E2%82%AC.PDF",
            },
            chunks=[pdf_bytes],
        )
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"min_pdf_bytes": 1, "previous": {}, "extract_html_text": True},
    )

    assert outcome.classification is Classification.PDF
    assert outcome.path is not None
    assert outcome.path.endswith(".pdf")


# --- test_download_outcomes.py ---


def test_html_filename_suffix_from_disposition(tmp_path, patcher):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/content"
    html_bytes = b"<html><body>Hi</body></html>"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/xhtml+xml",
                "Content-Disposition": "inline; filename=landing.xhtml",
            },
            chunks=[html_bytes],
        )
    }
    stub_requests(patcher, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"min_pdf_bytes": 1, "previous": {}, "extract_html_text": True},
    )

    assert outcome.classification is Classification.HTML
    assert outcome.path is not None
    assert outcome.path.endswith(".xhtml")


# --- test_download_utils.py ---

pytest.importorskip("pyalex")


# --- test_download_utils.py ---


def test_slugify_truncates_and_normalises():
    assert downloader.slugify("Hello, World!", keep=8) == "Hello_Wo"
    assert downloader.slugify("   ", keep=10) == "untitled"
    assert downloader.slugify("Study: B-cells & growth", keep=40) == "Study_Bcells_growth"


# --- test_download_utils.py ---


@pytest.mark.parametrize(
    "payload,ctype,url,expected",
    [
        (b"%PDF-sample", "application/pdf", "https://example.org/file.pdf", "pdf"),
        (b"   %PDF-1.4", "text/plain", "https://example.org/file.bin", "pdf"),
        (b"<html><head></head>", "text/html", "https://example.org", "html"),
        (b"", "application/pdf", "https://example.org/file.pdf", "pdf"),
        (b"", "text/plain", "https://example.org/foo.pdf", "pdf"),
    ],
)
def test_classify_payload_variants(payload, ctype, url, expected):
    expected_cls = Classification.from_wire(expected)
    assert classify_payload(payload, ctype, url) is expected_cls


# --- test_download_utils.py ---


def test_collect_location_urls_dedupes_and_tracks_sources(tmp_path: Path):
    work = {
        "best_oa_location": {
            "landing_page_url": "https://host.example/landing",
            "pdf_url": "https://host.example/paper.pdf",
            "source": {"display_name": "Host"},
        },
        "primary_location": {
            "landing_page_url": "https://host.example/landing",
            "pdf_url": "https://cdn.example/paper.pdf",
            "source": {"display_name": "Mirror"},
        },
        "locations": [
            {
                "landing_page_url": "https://mirror.example/landing",
                "pdf_url": "https://cdn.example/paper.pdf",
                "source": {"display_name": "Mirror"},
            }
        ],
        "open_access": {"oa_url": "https://oa.example/paper.pdf"},
    }
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()
    artifact = download_impl.create_artifact(work, pdf_dir, html_dir, xml_dir)
    assert artifact.landing_urls == [
        "https://host.example/landing",
        "https://mirror.example/landing",
    ]
    assert artifact.pdf_urls == [
        "https://host.example/paper.pdf",
        "https://cdn.example/paper.pdf",
        "https://oa.example/paper.pdf",
    ]
    assert artifact.source_display_names == ["Host", "Mirror"]


# --- test_download_utils.py ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://doi.org/10.1000/foo", "10.1000/foo"),
        (" 10.1000/bar ", "10.1000/bar"),
        (None, None),
    ],
)
def test_normalize_doi(value, expected):
    assert normalize_doi(value) == expected


# --- test_download_utils.py ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PMID:123456", "123456"),
        ("https://pubmed.ncbi.nlm.nih.gov/98765/", "98765"),
        (None, None),
    ],
)
def test_normalize_pmid(value, expected):
    assert normalize_pmid(value) == expected


# --- test_download_utils.py ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PMC12345", "PMC12345"),
        ("pmc9876", "PMC9876"),
        ("9876", "PMC9876"),
        (None, None),
    ],
)
def test_normalize_pmcid(value, expected):
    assert normalize_pmcid(value) == expected


# --- test_download_utils.py ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("arXiv:2101.12345", "2101.12345"),
        ("https://arxiv.org/abs/2010.00001", "2010.00001"),
        ("2101.99999", "2101.99999"),
    ],
)
def test_normalize_arxiv(value, expected):
    assert normalize_arxiv(value) == expected


# --- test_content_download_utils.py ---

try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

# --- test_content_download_utils.py ---

given = hypothesis.given


# --- test_content_download_utils.py ---


def test_normalize_doi_with_https_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"


# --- test_content_download_utils.py ---


def test_normalize_doi_without_prefix() -> None:
    assert normalize_doi("10.1234/abc") == "10.1234/abc"


# --- test_content_download_utils.py ---


def test_normalize_doi_with_whitespace() -> None:
    assert normalize_doi("  10.1234/abc  ") == "10.1234/abc"


# --- test_content_download_utils.py ---


def test_normalize_doi_none() -> None:
    assert normalize_doi(None) is None


# --- test_content_download_utils.py ---


@pytest.mark.parametrize(
    "prefix",
    [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
        "DOI:",
    ],
)
def test_normalize_doi_prefix_variants(prefix: str) -> None:
    canonical = "10.1234/Example"
    assert normalize_doi(f"{prefix}{canonical}") == canonical


# --- test_content_download_utils.py ---


def test_normalize_pmcid_with_pmc_prefix() -> None:
    assert normalize_pmcid("PMC123456") == "PMC123456"


# --- test_content_download_utils.py ---


def test_normalize_pmcid_without_prefix_adds_prefix() -> None:
    assert normalize_pmcid("123456") == "PMC123456"


# --- test_content_download_utils.py ---


def test_normalize_pmcid_lowercase() -> None:
    assert normalize_pmcid("pmc123456") == "PMC123456"


# --- test_content_download_utils.py ---


def test_strip_prefix_case_insensitive() -> None:
    assert strip_prefix("ARXIV:2301.12345", "arxiv:") == "2301.12345"


# --- test_content_download_utils.py ---


def test_dedupe_preserves_order() -> None:
    assert dedupe(["b", "a", "b", "c"]) == ["b", "a", "c"]


# --- test_content_download_utils.py ---


def test_dedupe_filters_falsey_values() -> None:
    assert dedupe(["a", "", None, "a"]) == ["a"]


# --- test_content_download_utils.py ---


@given(st.lists(st.text()))
def test_dedupe_property(values: List[str]) -> None:
    expected = []
    seen = set()
    for item in values:
        if item and item not in seen:
            expected.append(item)
            seen.add(item)

    assert dedupe(values) == expected


# --- test_edge_cases.py ---

pytest.importorskip("pyalex")

# --- test_edge_cases.py ---

requests = pytest.importorskip("requests")

# --- test_edge_cases.py ---

responses = pytest.importorskip("responses")


# --- test_edge_cases.py ---


def _make_artifact(tmp_path: Path, **overrides: Any) -> WorkArtifact:
    params: Dict[str, Any] = dict(
        work_id="WEDGE",
        title="Edge Case",
        publication_year=2024,
        doi="10.1/test",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/resource"],
        open_access_url=None,
        source_display_names=[],
        base_stem="edge-case",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )
    params.update(overrides)
    return WorkArtifact(**params)


# --- test_edge_cases.py ---


@responses.activate
def test_html_classification_overrides_misleading_content_type(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    url = artifact.pdf_urls[0]
    responses.add(responses.HEAD, url, status=200, headers={"Content-Type": "application/pdf"})
    responses.add(responses.GET, url, status=200, body="<html><body>Fake PDF</body></html>")

    outcome = download_candidate(
        requests.Session(),
        artifact,
        url,
        referer=None,
        timeout=10.0,
        context={"dry_run": False, "extract_html_text": True, "previous": {}},
    )

    assert outcome.classification is Classification.HTML
    assert outcome.path and outcome.path.endswith(".html")


# --- test_edge_cases.py ---


@responses.activate
def test_wayback_resolver_skips_unavailable_archives(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path, pdf_urls=[])
    artifact.failed_pdf_urls = ["https://example.org/missing.pdf"]
    session = requests.Session()
    config = ResolverConfig()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={
            "archived_snapshots": {"closest": {"available": False, "url": "https://archive.org"}}
        },
        status=200,
    )

    results = list(WaybackResolver().iter_urls(session, config, artifact))
    assert results == []


# --- test_edge_cases.py ---


def test_manifest_and_attempts_single_success(tmp_path: Path) -> None:
    work = {
        "id": "https://openalex.org/WEDGE",
        "title": "Edge Case",
        "publication_year": 2024,
        "ids": {"doi": "10.1/test"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }

    artifact = _make_artifact(tmp_path)
    logger_path = tmp_path / "attempts.jsonl"
    sink = JsonlSink(logger_path)
    logger = RunTelemetry(sink)
    metrics = ResolverMetrics()

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterable[ResolverResult]:
            yield ResolverResult(url="https://resolver.example/paper.pdf")

    def fake_download(*args: Any, **kwargs: Any) -> DownloadOutcome:
        pdf_path = artifact.pdf_dir / "resolver.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
        return DownloadOutcome(
            classification="pdf",
            path=str(pdf_path),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=5.0,
            sha256="deadbeef",
            content_length=12,
        )

    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        enable_head_precheck=False,
    )
    pipeline = ResolverPipeline(
        [StubResolver()], config, fake_download, logger, metrics, run_id="test-run"
    )

    options = DownloadConfig(
        dry_run=False,
        list_only=False,
        extract_html_text=False,
        run_id="test-run",
        previous_lookup={},
        resume_completed=set(),
        sniff_bytes=DEFAULT_SNIFF_BYTES,
        min_pdf_bytes=DEFAULT_MIN_PDF_BYTES,
        tail_check_bytes=DEFAULT_TAIL_CHECK_BYTES,
    )
    result = process_one_work(
        work,
        requests.Session(),
        artifact.pdf_dir,
        artifact.html_dir,
        artifact.xml_dir,
        pipeline,
        logger,
        metrics,
        options=options,
    )

    logger.close()

    assert result["saved"] is True
    records = [
        json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    attempts = [
        entry for entry in records if entry["record_type"] == "attempt" and entry["status"] == "pdf"
    ]
    manifests = [
        entry
        for entry in records
        if entry["record_type"] == "manifest" and entry["classification"] == "pdf"
    ]
    assert len(attempts) == 1
    assert len(manifests) == 1
    assert attempts[0]["work_id"] == manifests[0]["work_id"] == "WEDGE"
    assert attempts[0]["sha256"] == "deadbeef"
    assert attempts[0]["run_id"] == "test-run"
    assert manifests[0]["run_id"] == "test-run"
    assert manifests[0]["path"].endswith("resolver.pdf")
    assert Path(manifests[0]["path"]).exists()


def test_retry_after_updates_breakers(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "breaker.jsonl")
    logger = RunTelemetry(sink)
    metrics = ResolverMetrics()

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterable[ResolverResult]:
            yield ResolverResult(url="https://resolver.example/paper.pdf")

    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        enable_head_precheck=False,
        resolver_circuit_breakers={"stub": {"failure_threshold": 1, "cooldown_seconds": 1.0}},
        domain_token_buckets={
            "resolver.example": {
                "rate_per_second": 10.0,
                "capacity": 10.0,
                "breaker_threshold": 1,
                "breaker_cooldown": 1.0,
            }
        },
    )
    pipeline = ResolverPipeline(
        [StubResolver()], config, lambda *args, **kwargs: None, logger, metrics, run_id="test-run"
    )

    outcome = DownloadOutcome(
        classification=Classification.HTTP_ERROR,
        path=None,
        http_status=500,
        content_type="application/pdf",
        elapsed_ms=10.0,
        reason=ReasonCode.HTTP_STATUS,
        reason_detail="500",
        retry_after=7.5,
    )

    pipeline._update_breakers("stub", "resolver.example", outcome)

    resolver_breaker = pipeline._resolver_breakers["stub"]
    assert resolver_breaker.allow() is False
    assert resolver_breaker.cooldown_remaining() >= 7.0

    host_breaker = pipeline._ensure_host_breaker("resolver.example")
    assert host_breaker.allow() is False
    assert host_breaker.cooldown_remaining() >= 7.0

    logger.close()


# --- test_edge_cases.py ---


def test_openalex_attempts_use_session_headers(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    logger_path = tmp_path / "attempts.jsonl"
    sink = JsonlSink(logger_path)
    logger = RunTelemetry(sink)
    metrics = ResolverMetrics()
    session = requests.Session()
    session.headers.update({"User-Agent": "EdgeTester/1.0"})
    observed: List[str] = []

    def fake_download(session_obj, art, url, referer, timeout, context=None):
        observed.append(session_obj.headers.get("User-Agent"))
        return DownloadOutcome(
            classification=Classification.HTTP_ERROR.value,
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=1.0,
            reason=ReasonCode.REQUEST_EXCEPTION,
        )

    config = ResolverConfig(
        resolver_order=["openalex"],
        resolver_toggles={"openalex": True},
        enable_head_precheck=False,
    )
    pipeline = ResolverPipeline(
        [OpenAlexResolver()],
        config,
        fake_download,
        logger,
        metrics,
        run_id="test-run",
    )

    pipeline.run(session, artifact)

    assert observed == ["EdgeTester/1.0"]
    logger.close()


# --- test_edge_cases.py ---


def test_retry_budget_honours_max_attempts(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        max_attempts_per_work=3,
        enable_head_precheck=False,
    )

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterator[ResolverResult]:
            for i in range(10):
                yield ResolverResult(url=f"https://resolver.example/{i}.pdf")

    calls: List[str] = []

    def failing_download(*args: Any, **kwargs: Any) -> DownloadOutcome:
        url = args[2]
        calls.append(url)
        return DownloadOutcome(
            classification="http_error",
            path=None,
            http_status=503,
            content_type="text/plain",
            elapsed_ms=1.0,
            reason=ReasonCode.HTTP_STATUS,
        )

    class ListLogger:
        def __init__(self) -> None:
            self.records: List[Any] = []

        def log_attempt(self, record, *, timestamp: Optional[str] = None) -> None:
            self.records.append(record)

        def log_manifest(self, entry) -> None:  # pragma: no cover - noop
            return None

        def log_summary(self, summary) -> None:  # pragma: no cover - noop
            return None

        def close(self) -> None:  # pragma: no cover - noop
            return None

    pipeline = ResolverPipeline(
        [StubResolver()],
        config,
        failing_download,
        ListLogger(),
        ResolverMetrics(),
        run_id="test-run",
    )

    result = pipeline.run(requests.Session(), artifact)
    assert result.success is False
    assert len(calls) == config.max_attempts_per_work


def test_load_manifest_url_index_reads_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest.sqlite3"
    sink = SqliteSink(db_path)
    entry = ManifestEntry(
        schema_version=downloader.MANIFEST_SCHEMA_VERSION,
        timestamp="2025-01-01T00:00:00Z",
        run_id="run-idx",
        work_id="W-index",
        title="Index Test",
        publication_year=2024,
        resolver="resolver",
        url="https://example.org/resource",
        path=str(tmp_path / "file.pdf"),
        classification=Classification.PDF.value,
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="deadbeef",
        content_length=123,
        etag='"etag"',
        last_modified="Wed, 01 May 2024 00:00:00 GMT",
        extracted_text_path=None,
        dry_run=False,
    )
    sink.log_manifest(entry)
    sink.close()
    mapping = load_manifest_url_index(db_path)
    normalised = normalize_url(entry.url)
    assert normalised in mapping
    record = mapping[normalised]
    assert record["sha256"] == entry.sha256
    assert record["etag"] == entry.etag
    assert record["content_length"] == entry.content_length


def test_manifest_url_index_lazy_get_does_not_trigger_full_load(tmp_path: Path, patcher):
    db_path = tmp_path / "manifest.sqlite3"
    sink = SqliteSink(db_path)
    entry = ManifestEntry(
        schema_version=downloader.MANIFEST_SCHEMA_VERSION,
        timestamp="2025-01-02T00:00:00Z",
        run_id="run-lazy",
        work_id="W-lazy",
        title="Lazy Load",
        publication_year=2025,
        resolver="resolver",
        url="https://example.org/lazy.pdf",
        path=str(tmp_path / "lazy.pdf"),
        classification=Classification.PDF.value,
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="lazyhash",
        content_length=2048,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
    )
    sink.log_manifest(entry)
    sink.close()

    def _fail_loader(path):
        raise AssertionError("ManifestUrlIndex should not eager-load via load_manifest_url_index")

    patcher.setattr("DocsToKG.ContentDownload.telemetry.load_manifest_url_index", _fail_loader)

    index = ManifestUrlIndex(db_path, eager=False)
    record = index.get(entry.url)
    assert record is not None
    assert record["sha256"] == entry.sha256

    with pytest.raises(AssertionError):
        index.as_dict()


def test_manifest_url_index_iter_existing_filters_missing_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest.sqlite3"
    sink = SqliteSink(db_path)
    existing_path = tmp_path / "present.pdf"
    existing_path.write_bytes(b"%PDF-1.4\n")
    present_entry = ManifestEntry(
        schema_version=downloader.MANIFEST_SCHEMA_VERSION,
        timestamp="2025-01-03T00:00:00Z",
        run_id="run-present",
        work_id="W-present",
        title="Present",
        publication_year=2023,
        resolver="resolver",
        url="https://example.org/present.pdf",
        path=str(existing_path),
        classification=Classification.PDF.value,
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="presenthash",
        content_length=existing_path.stat().st_size,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
    )
    missing_entry = ManifestEntry(
        schema_version=downloader.MANIFEST_SCHEMA_VERSION,
        timestamp="2025-01-03T00:05:00Z",
        run_id="run-missing",
        work_id="W-missing",
        title="Missing",
        publication_year=2023,
        resolver="resolver",
        url="https://example.org/missing.pdf",
        path=str(tmp_path / "missing.pdf"),
        classification=Classification.PDF.value,
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="missinghash",
        content_length=4096,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
    )
    sink.log_manifest(present_entry)
    sink.log_manifest(missing_entry)
    sink.close()

    index = ManifestUrlIndex(db_path)
    existing_items = list(index.iter_existing())
    assert len(existing_items) == 1
    key, payload = existing_items[0]
    assert key == normalize_url(present_entry.url)
    assert payload["path"] == str(existing_path)


def test_manifest_url_index_eager_load_invokes_loader_once(tmp_path: Path, patcher) -> None:
    db_path = tmp_path / "manifest.sqlite3"
    db_path.touch()
    url = "https://example.org/eager.pdf"
    normalized = normalize_url(url)
    mapping = {
        normalized: {
            "url": url,
            "path": str(tmp_path / "eager.pdf"),
            "sha256": "eagerhash",
            "classification": Classification.PDF.value,
            "etag": None,
            "last_modified": None,
            "content_length": 1024,
            "mtime_ns": None,
        }
    }
    calls: List[Path] = []

    def _tracking_loader(path: Path):
        calls.append(path)
        return mapping

    patcher.setattr("DocsToKG.ContentDownload.telemetry.load_manifest_url_index", _tracking_loader)

    index = ManifestUrlIndex(db_path, eager=True)
    assert calls == [db_path]
    assert index.get(url) == mapping[normalized]
    assert index.as_dict() == mapping
    assert calls == [db_path]
