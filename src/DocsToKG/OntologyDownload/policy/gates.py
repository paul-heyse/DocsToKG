"""Six concrete policy gates for defense-in-depth security.

Each gate validates a specific boundary and returns PolicyOK or PolicyReject.
All gates are registered with the central registry on import.

Gates:
1. Configuration gate - validate settings on startup
2. URL & network gate - validate URLs, hosts, DNS
3. Filesystem & path gate - validate paths, prevent traversal
4. Extraction policy gate - validate archive entries, detect bombs
5. Storage gate - validate storage operations
6. DB transactional gate - validate database operations
"""

import time
from typing import Any, Dict, Optional, Union

from DocsToKG.OntologyDownload.policy.errors import (
    ErrorCode,
    ExtractionPolicyException,
    FilesystemPolicyException,
    PolicyOK,
    PolicyReject,
    StoragePolicyException,
    URLPolicyException,
    raise_policy_error,
)
from DocsToKG.OntologyDownload.policy.registry import policy_gate

# ============================================================================
# Gate 1: Configuration Gate
# ============================================================================


@policy_gate(
    name="config_gate",
    description="Validate configuration settings against policy",
    domain="config",
)
def config_gate(config: Any) -> Union[PolicyOK, PolicyReject]:
    """Validate configuration settings.

    Args:
        config: Configuration object to validate

    Returns:
        PolicyOK if config is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000

    try:
        # Basic validation - config must have required attributes
        if not hasattr(config, "http_settings"):
            raise ValueError("Missing http_settings")
        if not hasattr(config, "network_settings"):
            raise ValueError("Missing network_settings")

        # Check for reasonable bounds
        if hasattr(config, "timeout_sec") and config.timeout_sec < 0:
            raise_policy_error(
                ErrorCode.E_CONFIG_INVALID,
                "Timeout must be non-negative",
                {"timeout": config.timeout_sec},
            )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="config_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, URLPolicyException):
            raise
        raise_policy_error(
            ErrorCode.E_CONFIG_VALIDATION,
            str(e),
            {},
        )  # type: ignore[no-untyped-call]
        assert False  # All paths above raise


# ============================================================================
# Gate 2: URL & Network Gate
# ============================================================================


@policy_gate(
    name="url_gate",
    description="Validate URLs, hosts, and DNS resolution",
    domain="network",
)
def url_gate(
    url: str,
    allowed_hosts: Optional[set] = None,
    allowed_ports: Optional[set] = None,
) -> Union[PolicyOK, PolicyReject]:
    """Validate URL against network security policy.

    Args:
        url: URL to validate
        allowed_hosts: Set of allowed hosts (None = allow all)
        allowed_ports: Set of allowed ports (None = allow standard 80/443)

    Returns:
        PolicyOK if URL is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {}

    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)

        # Validate scheme
        if parsed.scheme not in ("http", "https"):
            details = {"scheme": parsed.scheme}
            raise_policy_error(
                ErrorCode.E_SCHEME,
                f"Invalid scheme: {parsed.scheme}",
                details,
                URLPolicyException,
            )

        # Reject userinfo in URL
        if parsed.username or parsed.password:
            raise_policy_error(
                ErrorCode.E_USERINFO,
                "Userinfo not allowed in URL",
                {},
                URLPolicyException,
            )

        host = parsed.hostname or ""
        if not host:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                "Empty or invalid host",
                {},
                URLPolicyException,
            )

        # Check host allowlist
        if allowed_hosts and host not in allowed_hosts:
            raise_policy_error(
                ErrorCode.E_HOST_DENY,
                f"Host not allowlisted: {host}",
                {"host": host},
                URLPolicyException,
            )

        # Validate port
        default_ports = {80, 443}
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        valid_ports = allowed_ports or default_ports

        if port not in valid_ports:
            raise_policy_error(
                ErrorCode.E_PORT_DENY,
                f"Port not allowed: {port}",
                {"port": port},
                URLPolicyException,
            )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="url_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, URLPolicyException):
            raise
        raise_policy_error(
            ErrorCode.E_HOST_DENY,
            str(e),
            details,
            URLPolicyException,
        )
        assert False  # All paths above raise


# ============================================================================
# Gate 3: Filesystem & Path Gate
# ============================================================================


@policy_gate(
    name="path_gate",
    description="Validate filesystem paths, prevent traversal",
    domain="filesystem",
)
def path_gate(
    path: str,
    root: Optional[str] = None,
    max_depth: int = 10,
) -> Union[PolicyOK, PolicyReject]:
    """Validate filesystem path against policy.

    Args:
        path: Path to validate
        root: Root directory (paths must be under this)
        max_depth: Maximum directory depth

    Returns:
        PolicyOK if path is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {"path": path.split("/")[-1] if "/" in path else path}

    try:
        import os

        # Reject absolute paths
        if os.path.isabs(path):
            raise_policy_error(
                ErrorCode.E_TRAVERSAL,
                "Absolute paths not allowed",
                details,
                FilesystemPolicyException,
            )

        # Reject .. or ./.. traversal attempts
        if ".." in path:
            raise_policy_error(
                ErrorCode.E_TRAVERSAL,
                "Path traversal attempt (..) detected",
                details,
                FilesystemPolicyException,
            )

        # Reject paths starting with /
        if path.startswith("/"):
            raise_policy_error(
                ErrorCode.E_TRAVERSAL,
                "Paths starting with / not allowed",
                details,
                FilesystemPolicyException,
            )

        # Check depth
        parts = [p for p in path.split("/") if p and p != "."]
        if len(parts) > max_depth:
            raise_policy_error(
                ErrorCode.E_DEPTH,
                f"Path too deep: {len(parts)} > {max_depth}",
                {"depth": len(parts), "max_depth": max_depth},
                FilesystemPolicyException,
            )

        # Check segment length (Windows limit is 255)
        for segment in parts:
            if len(segment) > 255:
                raise_policy_error(
                    ErrorCode.E_SEGMENT_LEN,
                    f"Path segment too long: {len(segment)} > 255",
                    {"segment_len": len(segment)},
                    FilesystemPolicyException,
                )

        # Check overall path length (Windows limit is 260)
        if len(path) > 260:
            raise_policy_error(
                ErrorCode.E_PATH_LEN,
                f"Path too long: {len(path)} > 260",
                {"path_len": len(path)},
                FilesystemPolicyException,
            )

        # Check for Windows reserved names
        reserved = {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}
        for segment in parts:
            if segment.upper() in reserved:
                raise_policy_error(
                    ErrorCode.E_PORTABILITY,
                    f"Windows reserved name: {segment}",
                    {"reserved_name": segment},
                    FilesystemPolicyException,
                )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="path_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, FilesystemPolicyException):
            raise
        raise_policy_error(
            ErrorCode.E_TRAVERSAL,
            str(e),
            details,
            FilesystemPolicyException,
        )
        assert False  # All paths above raise


# ============================================================================
# Gate 4: Extraction Policy Gate
# ============================================================================


@policy_gate(
    name="extraction_gate",
    description="Validate archive entries, detect zip bombs",
    domain="extraction",
)
def extraction_gate(
    entry: Dict[str, Any],
    max_ratio: float = 10.0,
    max_entry_size: int = 100 * 1024 * 1024,  # 100MB
) -> Union[PolicyOK, PolicyReject]:
    """Validate archive entry against extraction policy.

    Args:
        entry: Archive entry dict with 'type', 'size', 'compressed_size'
        max_ratio: Max expansion ratio (uncompressed / compressed)
        max_entry_size: Max uncompressed size in bytes

    Returns:
        PolicyOK if entry is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {}

    try:
        # Check entry type - only regular files allowed
        entry_type = entry.get("type", "file")
        if entry_type not in ("file", "regular"):
            raise_policy_error(
                ErrorCode.E_SPECIAL_TYPE,
                f"Entry type not allowed: {entry_type}",
                {"entry_type": entry_type},
                ExtractionPolicyException,
            )

        # Check file size
        size = entry.get("size", 0)
        if size > max_entry_size:
            raise_policy_error(
                ErrorCode.E_FILE_SIZE,
                f"File too large: {size} > {max_entry_size}",
                {"size": size, "max": max_entry_size},
                ExtractionPolicyException,
            )

        # Check compression ratio (zip bomb detection)
        compressed_size = entry.get("compressed_size", size)
        if compressed_size > 0:
            ratio = size / compressed_size
            if ratio > max_ratio:
                raise_policy_error(
                    ErrorCode.E_ENTRY_RATIO,
                    f"Zip bomb detected: ratio {ratio:.1f} > {max_ratio}",
                    {"ratio": ratio, "max_ratio": max_ratio},
                    ExtractionPolicyException,
                )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="extraction_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, ExtractionPolicyException):
            raise
        raise_policy_error(
            ErrorCode.E_BOMB_RATIO,
            str(e),
            details,
            ExtractionPolicyException,
        )
        assert False  # All paths above raise


# ============================================================================
# Gate 5: Storage Gate
# ============================================================================


@policy_gate(
    name="storage_gate",
    description="Validate storage operations",
    domain="storage",
)
def storage_gate(
    operation: str,
    details_dict: Optional[Dict[str, Any]] = None,
) -> Union[PolicyOK, PolicyReject]:
    """Validate storage operation against policy.

    Args:
        operation: Storage operation ('put', 'move', 'marker')
        details_dict: Operation details

    Returns:
        PolicyOK if operation is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details_dict = details_dict or {}

    try:
        # Valid operations
        valid_ops = {"put", "move", "marker", "delete"}
        if operation not in valid_ops:
            raise_policy_error(
                ErrorCode.E_STORAGE_PUT,
                f"Invalid storage operation: {operation}",
                {"operation": operation},
                StoragePolicyException,
            )

        # Basic validation - could be extended with path checks, etc.
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="storage_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, StoragePolicyException):
            raise
        raise_policy_error(
            ErrorCode.E_STORAGE_PUT,
            str(e),
            details_dict,
            StoragePolicyException,
        )
        assert False  # All paths above raise


# ============================================================================
# Gate 6: DB Transactional Gate
# ============================================================================


@policy_gate(
    name="db_gate",
    description="Validate database transactional operations",
    domain="db",
)
def db_gate(
    operation: str,
    details_dict: Optional[Dict[str, Any]] = None,
) -> Union[PolicyOK, PolicyReject]:
    """Validate database operation against transactional policy.

    Args:
        operation: DB operation ('commit', 'rollback', 'migrate')
        details_dict: Operation details

    Returns:
        PolicyOK if operation is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details_dict = details_dict or {}

    try:
        # Valid operations
        valid_ops = {"commit", "rollback", "migrate", "checkpoint"}
        if operation not in valid_ops:
            raise_policy_error(
                ErrorCode.E_DB_TX,
                f"Invalid DB operation: {operation}",
                {"operation": operation},
                StoragePolicyException,
            )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="db_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, StoragePolicyException):
            raise
        raise_policy_error(
            ErrorCode.E_DB_TX,
            str(e),
            details_dict,
            StoragePolicyException,
        )
        assert False  # All paths above raise


__all__ = [
    "config_gate",
    "url_gate",
    "path_gate",
    "extraction_gate",
    "storage_gate",
    "db_gate",
]
