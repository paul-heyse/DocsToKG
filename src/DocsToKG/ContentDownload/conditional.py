"""
Conditional Request Metadata

This module centralises helper types for conditional HTTP requests that rely on
ETag and Last-Modified headers. It provides strongly typed dataclasses for
capturing cached responses alongside a small orchestration helper that builds
request headers and interprets origin responses. The utilities are used by the
content download pipeline to honour polite download practices while avoiding
unnecessary network transfers.

Key Features:
- Dataclasses that model cached and modified responses with checksum metadata.
- Helper for constructing `If-None-Match`/`If-Modified-Since` headers.
- Utilities for validating 304 responses against previously cached artefacts.

Dependencies:
- `requests`: Required for the `Response` type used when interpreting outcomes.

Usage:
    from DocsToKG.ContentDownload.conditional import ConditionalRequestHelper

    helper = ConditionalRequestHelper(prior_etag=\"abcd\", prior_path=\"/tmp/file.pdf\")
    headers = helper.build_headers()
    response = session.get(url, headers=headers)
    result = helper.interpret_response(response)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import requests


@dataclass
class CachedResult:
    """Represents HTTP 304 Not Modified response with prior metadata.

    Attributes:
        path: File system path to the previously downloaded artifact.
        sha256: SHA256 checksum associated with the cached artifact.
        content_length: Size of the cached payload in bytes.
        etag: Entity tag reported by the origin server, if any.
        last_modified: Last-Modified header value supplied by the origin, if any.

    Examples:
        >>> CachedResult(
        ...     path=\"/tmp/file.pdf\",
        ...     sha256=\"abc\",
        ...     content_length=1024,
        ...     etag=\"W/\\\"etag\\\"\",
        ...     last_modified=\"Fri, 01 Jan 2021 00:00:00 GMT\"
        ... )
        CachedResult(path='/tmp/file.pdf', sha256='abc', content_length=1024, etag='W/"etag"', last_modified='Fri, 01 Jan 2021 00:00:00 GMT')
    """

    path: str
    sha256: str
    content_length: int
    etag: Optional[str]
    last_modified: Optional[str]


@dataclass
class ModifiedResult:
    """Represents HTTP 200 response requiring fresh download.

    Attributes:
        etag: Entity tag reported by the origin server.
        last_modified: Last-Modified header describing the remote resource timestamp.

    Examples:
        >>> ModifiedResult(etag=None, last_modified=None)
        ModifiedResult(etag=None, last_modified=None)
    """

    etag: Optional[str]
    last_modified: Optional[str]


class ConditionalRequestHelper:
    """Utility for constructing conditional requests and interpreting responses.

    Attributes:
        prior_etag: Previously observed entity tag for the resource.
        prior_last_modified: Previously observed Last-Modified header value.
        prior_sha256: Cached payload hash to validate 304 responses.
        prior_content_length: Size of the cached payload in bytes.
        prior_path: Local path where the cached artifact resides.

    Examples:
        >>> helper = ConditionalRequestHelper(prior_etag=\"abcd\", prior_path=\"/tmp/file.pdf\")
        >>> helper.build_headers()
        {'If-None-Match': 'abcd'}
    """

    def __init__(
        self,
        prior_etag: Optional[str] = None,
        prior_last_modified: Optional[str] = None,
        prior_sha256: Optional[str] = None,
        prior_content_length: Optional[int] = None,
        prior_path: Optional[str] = None,
    ) -> None:
        """Initialize helper state with metadata gathered from previous downloads.

        Args:
            prior_etag: ETag value observed during the last successful download.
            prior_last_modified: Last-Modified header value stored for the artefact.
            prior_sha256: SHA-256 checksum of the cached payload for integrity checks.
            prior_content_length: Size of the cached payload in bytes.
            prior_path: Filesystem path where the cached artefact resides.

        Returns:
            None

        Raises:
            ValueError: If ``prior_content_length`` is provided but negative.
        """

        if prior_content_length is not None and prior_content_length < 0:
            raise ValueError(
                f"prior_content_length must be non-negative, got {prior_content_length}"
            )
        self.prior_etag = prior_etag
        self.prior_last_modified = prior_last_modified
        self.prior_sha256 = prior_sha256
        self.prior_content_length = prior_content_length
        self.prior_path = prior_path

    def build_headers(self) -> Dict[str, str]:
        """Generate conditional request headers from cached metadata.

        Args:
            self: Helper instance containing cached HTTP metadata.

        Returns:
            Mapping of conditional header names to values ready for ``requests``.

        Examples:
            >>> helper = ConditionalRequestHelper(prior_etag=\"abcd\", prior_last_modified=\"Fri, 01 Jan 2021 00:00:00 GMT\")
            >>> helper.build_headers() == {'If-None-Match': 'abcd', 'If-Modified-Since': 'Fri, 01 Jan 2021 00:00:00 GMT'}
            True
        """

        headers: Dict[str, str] = {}
        if self.prior_etag:
            headers["If-None-Match"] = self.prior_etag
        if self.prior_last_modified:
            headers["If-Modified-Since"] = self.prior_last_modified
        return headers

    def interpret_response(
        self, response: requests.Response
    ) -> Union[CachedResult, ModifiedResult]:
        """Interpret response status and headers as cached or modified result.

        Args:
            response: HTTP response returned from the conditional request.

        Returns:
            `CachedResult` when the origin reports HTTP 304, otherwise `ModifiedResult`.

        Raises:
            ValueError: If a 304 response arrives without complete prior metadata.
            TypeError: If ``response`` lacks the minimal ``status_code``/``headers`` API.
        """

        if not hasattr(response, "status_code") or not hasattr(response, "headers"):
            raise TypeError("response must expose 'status_code' and 'headers' attributes")

        if response.status_code == 304:
            missing_fields = []
            if not self.prior_path:
                missing_fields.append("path")
            if not self.prior_sha256:
                missing_fields.append("sha256")
            if self.prior_content_length is None:
                missing_fields.append("content_length")

            if missing_fields:
                raise ValueError(
                    "HTTP 304 requires complete prior metadata. Missing: "
                    + ", ".join(missing_fields)
                    + ". This indicates a bug in manifest loading or caching logic."
                )
            assert self.prior_path is not None
            assert self.prior_sha256 is not None
            assert self.prior_content_length is not None
            return CachedResult(
                path=self.prior_path,
                sha256=self.prior_sha256,
                content_length=self.prior_content_length,
                etag=self.prior_etag,
                last_modified=self.prior_last_modified,
            )
        return ModifiedResult(
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )
