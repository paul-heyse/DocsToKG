# === NAVMAP v1 ===
# {
#   "module": "tests.test_property_based",
#   "purpose": "Pytest coverage for property based scenarios",
#   "sections": [
#     {
#       "id": "test-conditional-request-helper-build-headers",
#       "name": "test_conditional_request_helper_build_headers",
#       "anchor": "function-test-conditional-request-helper-build-headers",
#       "kind": "function"
#     },
#     {
#       "id": "test-request-with-retries-backoff-sequence",
#       "name": "test_request_with_retries_backoff_sequence",
#       "anchor": "function-test-request-with-retries-backoff-sequence",
#       "kind": "function"
#     },
#     {
#       "id": "test-dedupe-preserves-first-occurrence",
#       "name": "test_dedupe_preserves_first_occurrence",
#       "anchor": "function-test-dedupe-preserves-first-occurrence",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Property-based tests covering retry logic and conditional helpers."""

from __future__ import annotations

from collections import deque
from typing import Deque, List
from unittest.mock import patch

import httpx
from hypothesis import given
from hypothesis import strategies as st

from DocsToKG.ContentDownload.core import dedupe
from DocsToKG.ContentDownload.networking import (
    ConditionalRequestHelper,
    request_with_retries,
)


@given(
    etag=st.one_of(st.none(), st.text(max_size=20)),
    last_modified=st.one_of(st.none(), st.text(max_size=30)),
)
# --- Test Cases ---


def test_conditional_request_helper_build_headers(etag, last_modified):
    helper = ConditionalRequestHelper(
        prior_etag=etag,
        prior_last_modified=last_modified,
        prior_sha256="sha256" if etag or last_modified else None,
        prior_content_length=128 if etag or last_modified else None,
        prior_path="/tmp/attempt.jsonl" if etag or last_modified else None,
    )

    headers = helper.build_headers()

    if etag:
        assert headers["If-None-Match"] == etag
    else:
        assert "If-None-Match" not in headers

    if last_modified:
        assert headers["If-Modified-Since"] == last_modified
    else:
        assert "If-Modified-Since" not in headers


@given(st.lists(st.sampled_from([429, 500, 502, 503, 504]), max_size=4))
def test_request_with_retries_backoff_sequence(status_codes: List[int]) -> None:
    response_codes: Deque[int] = deque(status_codes + [200])

    def handler(request: httpx.Request) -> httpx.Response:
        if not response_codes:
            raise AssertionError("unexpected extra request")
        code = response_codes.popleft()
        return httpx.Response(code, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    class _SequenceWait:
        def __init__(self, values: List[float]) -> None:
            self._values = list(values)
            self._index = 0
            self._last = self._values[-1] if self._values else 0.0

        def __call__(self, retry_state) -> float:  # pragma: no cover - exercised via Tenacity
            if self._index < len(self._values):
                value = self._values[self._index]
                self._last = value
                self._index += 1
                return value
            return self._last

    delays = [0.2 * (idx + 1) for idx in range(len(status_codes))]

    with (
        patch(
            "DocsToKG.ContentDownload.networking.wait_random_exponential",
            side_effect=lambda *args, **kwargs: _SequenceWait(delays),
        ),
        patch("DocsToKG.ContentDownload.networking.time.sleep") as mock_sleep,
    ):
        result = request_with_retries(
            client,
            "GET",
            "https://example.org/property",
            max_retries=len(status_codes),
            backoff_factor=0.2,
            respect_retry_after=False,
        )

    assert result.status_code == 200
    observed_delays = [entry.args[0] for entry in mock_sleep.call_args_list]
    assert observed_delays == delays[: len(observed_delays)]
    client.close()


@given(st.lists(st.text(max_size=10)))
def test_dedupe_preserves_first_occurrence(values: List[str]) -> None:
    result = dedupe(values)

    # Ensure all retained elements are unique and in the same order as first seen.
    seen = set()
    indices = []
    for item in result:
        assert item not in seen
        seen.add(item)
        indices.append(values.index(item))

    assert indices == sorted(indices)

    # Ensure removing falsy entries matches reference behaviour.
    expected = []
    for item in values:
        if item and item not in expected:
            expected.append(item)
    assert result == expected
