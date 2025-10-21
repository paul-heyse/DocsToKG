"""Policy error codes and helpers for centralized error handling.

One error catalog used across all policy gates. Each gate raises through
the central helpers to ensure consistent emission, typing, and metrics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# ============================================================================
# Error Codes (Canonical Catalog)
# ============================================================================


class ErrorCode(str, Enum):
    """Canonical error codes for policy rejections.

    All gates reject by emitting one of these codes.
    Organized by domain (Network, Filesystem, Extraction, DB, Config).
    """

    # Network & TLS errors
    E_NET_CONNECT = "E_NET_CONNECT"  # Connection failed
    E_NET_READ = "E_NET_READ"  # Read from network failed
    E_NET_PROTOCOL = "E_NET_PROTOCOL"  # Protocol error
    E_TLS = "E_TLS"  # TLS/SSL error

    # URL & DNS errors
    E_SCHEME = "E_SCHEME"  # Invalid scheme (not http/https)
    E_USERINFO = "E_USERINFO"  # Userinfo in URL forbidden
    E_HOST_DENY = "E_HOST_DENY"  # Host not allowlisted
    E_PORT_DENY = "E_PORT_DENY"  # Port not allowed
    E_DNS_FAIL = "E_DNS_FAIL"  # DNS resolution failed
    E_PRIVATE_NET = "E_PRIVATE_NET"  # Private network not allowed

    # Redirect errors
    E_REDIRECT_DENY = "E_REDIRECT_DENY"  # Redirect target denied
    E_REDIRECT_LOOP = "E_REDIRECT_LOOP"  # Redirect loop detected

    # Filesystem & path errors
    E_TRAVERSAL = "E_TRAVERSAL"  # Path traversal attempt (.. or /)
    E_CASEFOLD_COLLISION = "E_CASEFOLD_COLLISION"  # Case-folded collision
    E_DEPTH = "E_DEPTH"  # Path too deep
    E_SEGMENT_LEN = "E_SEGMENT_LEN"  # Path segment too long
    E_PATH_LEN = "E_PATH_LEN"  # Full path too long
    E_PORTABILITY = "E_PORTABILITY"  # Non-portable path (Windows reserved)

    # Extraction & archive errors
    E_SPECIAL_TYPE = "E_SPECIAL_TYPE"  # Entry type not allowed
    E_BOMB_RATIO = "E_BOMB_RATIO"  # Zip bomb detected (global ratio)
    E_ENTRY_RATIO = "E_ENTRY_RATIO"  # Zip bomb detected (per-entry ratio)
    E_FILE_SIZE = "E_FILE_SIZE"  # File too large
    E_FILE_SIZE_STREAM = "E_FILE_SIZE_STREAM"  # Stream size limit exceeded
    E_ENTRY_BUDGET = "E_ENTRY_BUDGET"  # Too many entries

    # Storage & filesystem operations
    E_STORAGE_PUT = "E_STORAGE_PUT"  # Storage write failed
    E_STORAGE_MOVE = "E_STORAGE_MOVE"  # Storage move failed
    E_STORAGE_MARKER = "E_STORAGE_MARKER"  # Marker write failed

    # DB & transactional errors
    E_DB_TX = "E_DB_TX"  # Transaction failed
    E_DB_MIGRATION = "E_DB_MIGRATION"  # Migration error
    E_LATEST_MISMATCH = "E_LATEST_MISMATCH"  # Latest pointer mismatch

    # Configuration errors
    E_CONFIG_INVALID = "E_CONFIG_INVALID"  # Invalid configuration
    E_CONFIG_VALIDATION = "E_CONFIG_VALIDATION"  # Config validation failed


# ============================================================================
# Result Types (Typed Contracts)
# ============================================================================


@dataclass(frozen=True)
class PolicyOK:
    """Gate returned OK (passed policy check)."""

    gate_name: str
    elapsed_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PolicyReject:
    """Gate returned REJECT (failed policy check)."""

    gate_name: str
    error_code: ErrorCode
    elapsed_ms: float
    details: Dict[str, Any]  # Non-secret context (no URLs, paths with secrets, etc.)


# Type alias for gate results
PolicyResult = PolicyOK | PolicyReject


# ============================================================================
# Policy Exceptions
# ============================================================================


class PolicyException(Exception):
    """Base exception for policy rejections."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize policy exception.

        Args:
            error_code: Canonical error code
            message: Human-readable message
            details: Additional context (non-secret)
        """
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(f"{error_code.value}: {message}")


class URLPolicyException(PolicyException):
    """URL or network policy rejection."""

    pass


class FilesystemPolicyException(PolicyException):
    """Filesystem or path policy rejection."""

    pass


class ExtractionPolicyException(PolicyException):
    """Extraction or archive policy rejection."""

    pass


class StoragePolicyException(PolicyException):
    """Storage or DB policy rejection."""

    pass


class ConfigurationPolicyException(PolicyException):
    """Configuration policy rejection."""

    pass


# ============================================================================
# Error Emission Helpers
# ============================================================================


def raise_policy_error(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    exception_class: type = PolicyException,
) -> None:
    """Raise a policy error with central emission.

    Args:
        error_code: Canonical error code
        message: Human-readable message
        details: Additional context (automatically scrubbed of secrets)
        exception_class: Exception type to raise

    Raises:
        The specified exception class with error code and details
    """
    # Scrub details of sensitive information
    safe_details = _scrub_details(details or {})

    # Emit event (when observability is initialized)
    # This will be called by observability system when available
    _emit_policy_error_event(error_code, message, safe_details)

    # Raise the exception
    raise exception_class(error_code, message, safe_details)


def _scrub_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive information from detail dict.

    Filters out:
    - Keys containing 'password', 'token', 'secret', 'auth'
    - Full URLs (replace with redacted versions)
    - Full paths (replace with basenames)
    """
    scrubbed = {}
    for key, value in details.items():
        key_lower = key.lower()

        # Skip sensitive keys
        if any(
            sensitive in key_lower for sensitive in ("password", "token", "secret", "auth", "key")
        ):
            continue

        # Redact URLs
        if key_lower in ("url", "target_url", "redirect_to"):
            scrubbed[key] = "[REDACTED_URL]"
            continue

        # Redact full paths
        if key_lower in ("path", "file_path", "entry_path"):
            if isinstance(value, str):
                scrubbed[key] = value.split("/")[-1] or "[path]"
            continue

        scrubbed[key] = value

    return scrubbed


def _emit_policy_error_event(
    error_code: ErrorCode,
    message: str,
    details: Dict[str, Any],
) -> None:
    """Emit a policy.error event (stub for observability integration).

    Will be filled in when observability system is initialized.
    """
    # TODO: Integrate with observability.emit_event when available
    # emit_event(
    #     type="policy.error",
    #     level="ERROR",
    #     payload={
    #         "error_code": error_code.value,
    #         "message": message,
    #         "details": details,
    #     },
    # )
    pass


__all__ = [
    "ErrorCode",
    "PolicyOK",
    "PolicyReject",
    "PolicyResult",
    "PolicyException",
    "URLPolicyException",
    "FilesystemPolicyException",
    "ExtractionPolicyException",
    "StoragePolicyException",
    "ConfigurationPolicyException",
    "raise_policy_error",
]
