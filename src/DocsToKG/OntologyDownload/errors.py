"""Shared exception hierarchy for the ontology downloader."""

from __future__ import annotations

from typing import Optional

__all__ = [
    "UserConfigError",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "PolicyError",
    "DownloadFailure",
]


class UserConfigError(RuntimeError):
    """Raised when CLI arguments or YAML configuration inputs are invalid."""


class OntologyDownloadError(RuntimeError):
    """Base class for runtime failures during ontology planning or download."""


class ResolverError(OntologyDownloadError):
    """Raised when resolver planning cannot produce a usable fetch plan."""


class ValidationError(OntologyDownloadError):
    """Raised when ontology validation encounters unrecoverable problems."""


class PolicyError(OntologyDownloadError):
    """Raised when security, licensing, or rate limit policies are violated."""


class DownloadFailure(OntologyDownloadError):
    """Raised when an HTTP download attempt fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
