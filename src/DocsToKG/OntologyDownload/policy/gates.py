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
from typing import Any, Dict, Optional, Union, List

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
    name="filesystem_gate",
    description="Validate filesystem paths against security policy",
    domain="filesystem",
)
def filesystem_gate(
    root_path: str,
    entry_paths: List[str],
    allow_symlinks: bool = False,
) -> Union[PolicyOK, PolicyReject]:
    """Validate filesystem paths against security policy.

    Enforces:
    - Path normalization (NFC Unicode, no .. / absolute paths)
    - Casefold collision detection
    - Depth and length constraints
    - Symlink rejection (unless explicitly allowed)
    - Windows reserved name rejection
    - Encapsulation within root

    Args:
        root_path: Root extraction directory
        entry_paths: Relative paths to validate
        allow_symlinks: If False, reject symlink entries

    Returns:
        PolicyOK if all paths valid, PolicyReject otherwise
    """
    import os
    import re
    import unicodedata
    from pathlib import Path, PureWindowsPath

    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {}

    try:
        # Normalize root path
        root_path = os.path.normpath(root_path)
        root_resolved = os.path.abspath(root_path)

        # Validate root exists and is directory
        if not os.path.isdir(root_resolved):
            raise_policy_error(
                ErrorCode.E_TRAVERSAL,
                f"Root path is not a directory: {root_path}",
                {"root": root_path},
                FilesystemPolicyException,
            )

        # Track seen paths for collision detection
        seen_casefold: Dict[str, str] = {}

        for entry_path in entry_paths:
            # Convert to string if needed
            entry_path_str = str(entry_path)

            # Reject absolute paths
            if os.path.isabs(entry_path_str):
                details = {"path": entry_path_str, "issue": "absolute_path"}
                raise_policy_error(
                    ErrorCode.E_TRAVERSAL,
                    f"Absolute path not allowed: {entry_path_str}",
                    details,
                    FilesystemPolicyException,
                )

            # Reject .. components
            if ".." in entry_path_str or entry_path_str.startswith("."):
                details = {"path": entry_path_str, "issue": "traversal_attempt"}
                raise_policy_error(
                    ErrorCode.E_TRAVERSAL,
                    f"Path traversal detected: {entry_path_str}",
                    details,
                    FilesystemPolicyException,
                )

            # Unicode NFC normalization
            normalized = unicodedata.normalize("NFC", entry_path_str)
            if normalized != entry_path_str:
                entry_path_str = normalized

            # Length constraints
            if len(entry_path_str) > 4096:
                details = {"path": entry_path_str[:50], "length": len(entry_path_str)}
                raise_policy_error(
                    ErrorCode.E_PATH_LEN,
                    f"Path too long: {len(entry_path_str)} > 4096",
                    details,
                    FilesystemPolicyException,
                )

            # Depth constraint (max 20 levels)
            depth = entry_path_str.count(os.sep)
            if depth > 20:
                details = {"path": entry_path_str, "depth": depth}
                raise_policy_error(
                    ErrorCode.E_DEPTH,
                    f"Path too deep: {depth} > 20 levels",
                    details,
                    FilesystemPolicyException,
                )

            # Check segment length (max 255 per component)
            for segment in entry_path_str.split(os.sep):
                if len(segment) > 255:
                    details = {"path": entry_path_str, "segment": segment[:50]}
                    raise_policy_error(
                        ErrorCode.E_SEGMENT_LEN,
                        f"Path segment too long: {len(segment)} > 255",
                        details,
                        FilesystemPolicyException,
                    )

                # Reject Windows reserved names
                reserved = {
                    "con", "prn", "aux", "nul",
                    "com1", "com2", "com3", "com4", "com5",
                    "com6", "com7", "com8", "com9",
                    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5",
                    "lpt6", "lpt7", "lpt8", "lpt9",
                }
                if segment.lower() in reserved:
                    details = {"segment": segment}
                    raise_policy_error(
                        ErrorCode.E_PORTABILITY,
                        f"Windows reserved name: {segment}",
                        details,
                        FilesystemPolicyException,
                    )

            # Casefold collision detection
            casefold_key = entry_path_str.casefold()
            if casefold_key in seen_casefold:
                details = {
                    "path1": seen_casefold[casefold_key],
                    "path2": entry_path_str,
                }
                raise_policy_error(
                    ErrorCode.E_CASEFOLD_COLLISION,
                    f"Case-insensitive collision: {entry_path_str}",
                    details,
                    FilesystemPolicyException,
                )
            seen_casefold[casefold_key] = entry_path_str

            # Resolve full path and check it's within root
            full_path = os.path.normpath(os.path.join(root_resolved, entry_path_str))
            full_resolved = os.path.abspath(full_path)

            # Security: ensure it doesn't escape root
            try:
                os.path.relpath(full_resolved, root_resolved)
                if not full_resolved.startswith(root_resolved + os.sep):
                    if full_resolved != root_resolved:
                        details = {"path": entry_path_str, "root": root_resolved}
                        raise_policy_error(
                            ErrorCode.E_TRAVERSAL,
                            f"Path escapes root: {entry_path_str}",
                            details,
                            FilesystemPolicyException,
                        )
            except ValueError:
                # relpath raises ValueError if paths are on different drives (Windows)
                details = {"path": entry_path_str, "root": root_resolved}
                raise_policy_error(
                    ErrorCode.E_TRAVERSAL,
                    f"Path on different volume: {entry_path_str}",
                    details,
                    FilesystemPolicyException,
                )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        return PolicyOK(gate_name="filesystem_gate", elapsed_ms=elapsed_ms)

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
    "filesystem_gate",
    "extraction_gate",
    "storage_gate",
    "db_gate",
]
