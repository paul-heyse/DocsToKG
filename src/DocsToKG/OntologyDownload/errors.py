"""Exception hierarchy shared across ontology planning, download, and validation.

The ontology pipeline spans configuration parsing, HTTP retrieval, archive
materialisation, and schema validation.  This module groups the failure modes
into a tidy hierarchy so caller code can react to high-level categories (for
example, policy violations vs. transient resolver errors) while still having
access to specialised subclasses when finer-grained handling is required.
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = [
    "OntologyDownloadError",
    "UnsupportedPythonError",
    "ConfigurationError",
    "ResolverError",
    "ValidationError",
    "RetryableValidationError",
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


class RetryableValidationError(ValidationError):
    """Validation error that supports retry semantics for resolver fallbacks."""

    def __init__(
        self,
        message: str,
        *,
        validators: Optional[Sequence[str]] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.validators = tuple(validators or ())
        self.retryable = retryable


class ValidationFailure(ValidationError):
    """Raised when validation results should be treated as a failed attempt."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


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
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.errors",
#   "purpose": "Define the exception hierarchy used across ontology planning, download, and validation",
#   "sections": [
#     {"id": "base", "name": "Base Exceptions", "anchor": "BAS", "kind": "api"},
#     {"id": "configuration", "name": "Configuration Errors", "anchor": "CFG", "kind": "api"},
#     {"id": "resolver", "name": "Resolver & Download Errors", "anchor": "RES", "kind": "api"},
#     {"id": "validation", "name": "Validation Errors", "anchor": "VAL", "kind": "api"},
#     {"id": "policy", "name": "Policy & Retry Failures", "anchor": "POL", "kind": "api"}
#   ]
# }
# === /NAVMAP ===
