"""Conditional HTTP request helpers for ETag and Last-Modified caching."""
from dataclasses import dataclass
from typing import Dict, Optional, Union

import requests


@dataclass
class CachedResult:
    """Represents HTTP 304 Not Modified response with prior metadata."""

    path: str
    sha256: str
    content_length: int
    etag: Optional[str]
    last_modified: Optional[str]


@dataclass
class ModifiedResult:
    """Represents HTTP 200 response requiring fresh download."""

    etag: Optional[str]
    last_modified: Optional[str]


class ConditionalRequestHelper:
    """Utility for constructing conditional requests and interpreting responses."""

    def __init__(
        self,
        prior_etag: Optional[str] = None,
        prior_last_modified: Optional[str] = None,
        prior_sha256: Optional[str] = None,
        prior_content_length: Optional[int] = None,
        prior_path: Optional[str] = None,
    ) -> None:
        self.prior_etag = prior_etag
        self.prior_last_modified = prior_last_modified
        self.prior_sha256 = prior_sha256
        self.prior_content_length = prior_content_length
        self.prior_path = prior_path

    def build_headers(self) -> Dict[str, str]:
        """Generate conditional request headers from prior metadata."""

        headers: Dict[str, str] = {}
        if self.prior_etag:
            headers["If-None-Match"] = self.prior_etag
        if self.prior_last_modified:
            headers["If-Modified-Since"] = self.prior_last_modified
        return headers

    def interpret_response(
        self, response: requests.Response
    ) -> Union[CachedResult, ModifiedResult]:
        """Interpret response status and headers as cached or modified result."""

        if response.status_code == 304:
            if (
                self.prior_path is None
                or self.prior_sha256 is None
                or self.prior_content_length is None
            ):
                raise ValueError("304 response requires complete prior metadata")
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
