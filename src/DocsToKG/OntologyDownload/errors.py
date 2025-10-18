"""Shared exception hierarchy for DocsToKG ontology downloads."""

from __future__ import annotations

from typing import Optional

__all__ = [
    "OntologyDownloadError",
    "UnsupportedPythonError",
    "ConfigurationError",
    "ResolverError",
    "ValidationError",
    "PolicyError",
    "DownloadFailure",
    "UserConfigError",
    "ConfigError",
]


class OntologyDownloadError(RuntimeError):
    """Base exception for ontology planning, download, or validation failures."""


class UnsupportedPythonError(OntologyDownloadError):
    """Raised when the active interpreter is older than the supported minimum."""


class ConfigurationError(OntologyDownloadError):
    """Raised when configuration inputs or manifests are invalid."""


class ResolverError(OntologyDownloadError):
    """Raised when resolver planning cannot produce a usable fetch plan."""


class ValidationError(OntologyDownloadError):
    """Raised when ontology validation encounters unrecoverable issues."""


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


class UserConfigError(RuntimeError):
    """Raised when CLI arguments or YAML configuration inputs are invalid."""


# Backwards compatibility alias used throughout the package.
ConfigError = UserConfigError
