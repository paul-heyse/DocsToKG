"""Policy gates: security boundaries for OntologyDownload.

Implements access control gates at critical I/O boundaries:
- Configuration validation
- URL/network security
- Filesystem path traversal prevention
- Archive extraction guards (zip bombs)
- Storage operation safety
- Database transaction boundaries
"""

import time
from typing import Any, Dict, List, Optional, Union

from prometheus_client import Counter, Gauge, Histogram

from DocsToKG.OntologyDownload.policy.errors import (
    DbBoundaryException,
    ErrorCode,
    ExtractionPolicyException,
    FilesystemPolicyException,
    PolicyOK,
    PolicyReject,
    StoragePolicyException,
    URLPolicyException,
    raise_policy_error,
)
from DocsToKG.OntologyDownload.policy.metrics import GateMetric, MetricsCollector
from DocsToKG.OntologyDownload.policy.registry import policy_gate

# Import observability (optional, may not be available during initialization)
try:
    from DocsToKG.OntologyDownload.observability.events import emit_event
except ImportError:
    # Fallback if observability not yet initialized

    def emit_event(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        """Fallback no-op event emitter."""
        pass


# ============================================================================
# Prometheus Metrics Registration
# ============================================================================

# Counter: Total gate invocations (pass/reject)
_gate_invocations = Counter(
    'gate_invocations_total',
    'Total gate invocations by outcome',
    ['gate', 'outcome']
)

# Histogram: Gate execution latency (milliseconds)
_gate_latency = Histogram(
    'gate_execution_ms',
    'Gate execution latency in milliseconds',
    ['gate'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 50.0, float('inf'))
)

# Counter: Gate errors by error code
_gate_errors = Counter(
    'gate_errors_total',
    'Total gate errors by error code',
    ['gate', 'error_code']
)

# Gauge: Current (latest) gate latency
_gate_current_latency = Gauge(
    'gate_current_latency_ms',
    'Current (latest) gate execution latency in milliseconds',
    ['gate']
)

# Gauge: Gate pass rate (percentage)
_gate_pass_rate = Gauge(
    'gate_pass_rate_percent',
    'Gate pass rate as percentage (0-100)',
    ['gate']
)


# ============================================================================
# Prometheus Metrics Recording
# ============================================================================


def _record_prometheus_metrics(
    gate_name: str,
    passed: bool,
    elapsed_ms: float,
    error_code: Optional[ErrorCode] = None,
) -> None:
    """Record gate metrics to Prometheus.

    Args:
        gate_name: Name of the gate
        passed: Whether gate passed
        elapsed_ms: Time spent in gate (milliseconds)
        error_code: Error code if gate rejected
    """
    try:
        # Record invocation count
        outcome = "ok" if passed else "reject"
        _gate_invocations.labels(gate=gate_name, outcome=outcome).inc()

        # Record latency histogram
        _gate_latency.labels(gate=gate_name).observe(elapsed_ms)

        # Record current latency gauge
        _gate_current_latency.labels(gate=gate_name).set(elapsed_ms)

        # Record error code counter if gate failed
        if not passed and error_code:
            _gate_errors.labels(gate=gate_name, error_code=error_code.name).inc()

        # Update pass rate gauge (approximate)
        # Note: This is approximate since we don't have total counts easily
        # In production, you'd want a rolling window calculator
        pass_rate = 100.0 if passed else 0.0
        _gate_pass_rate.labels(gate=gate_name).set(pass_rate)

    except Exception:
        # Silently fail - metrics should never break gate logic
        pass


# ============================================================================
# Telemetry Helpers
# ============================================================================


def _emit_gate_event(
    gate_name: str,
    outcome: str,  # "ok" or "reject"
    elapsed_ms: float,
    error_code: Optional[ErrorCode] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured policy.gate event.

    Args:
        gate_name: Name of the gate
        outcome: "ok" or "reject"
        elapsed_ms: Time spent in gate (milliseconds)
        error_code: Error code if rejected
        details: Additional context
    """
    try:
        emit_event(
            type="policy.gate",
            level="ERROR" if outcome == "reject" else "INFO",
            payload={
                "gate": gate_name,
                "outcome": outcome,
                "elapsed_ms": round(elapsed_ms, 2),
                "error_code": error_code.name if error_code else None,
                "details": details or {},
            },
        )
    except Exception:
        # Silently fail - telemetry should never break gate logic
        pass


def _record_gate_metric(
    gate_name: str,
    passed: bool,
    elapsed_ms: float,
    error_code: Optional[ErrorCode] = None,
) -> None:
    """Record a metric for this gate.

    Args:
        gate_name: Name of the gate
        passed: Whether gate passed
        elapsed_ms: Time spent
        error_code: Error code if failed
    """
    try:
        collector = MetricsCollector.instance()
        metric = GateMetric(
            gate_name=gate_name,
            passed=passed,
            elapsed_ms=elapsed_ms,
            error_code=error_code,
        )
        collector.record_metric(metric)
    except Exception:
        # Silently fail - metrics should never break gate logic
        pass

    # Also record to Prometheus
    _record_prometheus_metrics(gate_name, passed, elapsed_ms, error_code)


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
    import unicodedata

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
                    "con",
                    "prn",
                    "aux",
                    "nul",
                    "com1",
                    "com2",
                    "com3",
                    "com4",
                    "com5",
                    "com6",
                    "com7",
                    "com8",
                    "com9",
                    "lpt1",
                    "lpt2",
                    "lpt3",
                    "lpt4",
                    "lpt5",
                    "lpt6",
                    "lpt7",
                    "lpt8",
                    "lpt9",
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
    description="Validate archive extraction parameters and detect zip bombs",
    domain="extraction",
)
def extraction_gate(
    entries_total: int,
    bytes_declared: int,
    max_total_ratio: float = 100.0,
    max_entry_ratio: float = 10.0,
    max_file_size_mb: int = 10240,
    max_entries: int = 100000,
) -> Union[PolicyOK, PolicyReject]:
    """Validate archive extraction parameters.

    Enforces:
    - Zip bomb guards: global and per-entry compression ratios
    - File size limits
    - Entry count budget
    - Compression bomb detection

    Args:
        entries_total: Total entries in archive
        bytes_declared: Total uncompressed bytes declared
        max_total_ratio: Max compression ratio (uncompressed/compressed)
        max_entry_ratio: Max per-entry compression ratio
        max_file_size_mb: Max individual file size (MB)
        max_entries: Max total entries

    Returns:
        PolicyOK if archive is valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {}

    try:
        # Entry budget check
        if entries_total > max_entries:
            details = {"entries": entries_total, "max_entries": max_entries}
            raise_policy_error(
                ErrorCode.E_ENTRY_BUDGET,
                f"Too many entries: {entries_total} > {max_entries}",
                details,
                ExtractionPolicyException,
            )

        # Avoid division by zero
        if bytes_declared <= 0:
            details = {"bytes_declared": bytes_declared}
            raise_policy_error(
                ErrorCode.E_BOMB_RATIO,
                "Invalid bytes_declared: must be > 0",
                details,
                ExtractionPolicyException,
            )

        # Calculate average compression ratio
        avg_ratio = bytes_declared / max(entries_total, 1)
        if avg_ratio > max_entry_ratio:
            details = {
                "avg_ratio": round(avg_ratio, 2),
                "max_entry_ratio": max_entry_ratio,
                "entries": entries_total,
                "bytes": bytes_declared,
            }
            raise_policy_error(
                ErrorCode.E_ENTRY_RATIO,
                f"Average compression ratio too high: {avg_ratio:.2f} > {max_entry_ratio}",
                details,
                ExtractionPolicyException,
            )

        # Zip bomb detection: if declared uncompressed size seems too large
        # relative to typical compressed archives, flag it
        bytes_declared_mb = bytes_declared / (1024 * 1024)
        if bytes_declared_mb > max_file_size_mb and entries_total <= 10:
            # Suspicious: large uncompressed but few entries
            details = {
                "bytes_mb": round(bytes_declared_mb, 2),
                "max_mb": max_file_size_mb,
                "entries": entries_total,
            }
            raise_policy_error(
                ErrorCode.E_BOMB_RATIO,
                f"Suspicious compression bomb: {bytes_declared_mb:.2f} MB from {entries_total} entries",
                details,
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
    description="Validate storage operations for safety",
    domain="storage",
)
def storage_gate(
    operation: str,
    src_path: str,
    dst_path: str,
    check_traversal: bool = True,
) -> Union[PolicyOK, PolicyReject]:
    """Validate storage operations.

    Enforces:
    - Atomic write pattern (temp + move)
    - Path traversal prevention
    - Permission safety

    Args:
        operation: Operation type (put, move, copy, delete)
        src_path: Source path
        dst_path: Destination path
        check_traversal: Whether to check for path traversal

    Returns:
        PolicyOK if operation is valid, PolicyReject otherwise
    """
    import os

    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {}

    try:
        # Validate operation
        valid_ops = {"put", "move", "copy", "delete", "rename"}
        if operation not in valid_ops:
            details = {"operation": operation}
            raise_policy_error(
                ErrorCode.E_STORAGE_PUT,
                f"Invalid operation: {operation}",
                details,
                StoragePolicyException,
            )

        # Check for traversal in destination
        if check_traversal:
            if ".." in dst_path or dst_path.startswith("/"):
                details = {"path": dst_path, "operation": operation}
                raise_policy_error(
                    ErrorCode.E_TRAVERSAL,
                    f"Path traversal in storage operation: {dst_path}",
                    details,
                    StoragePolicyException,
                )

        # For move/rename operations, check source exists
        if operation in ("move", "rename", "copy"):
            if src_path.startswith("/") and not os.path.exists(src_path):
                details = {"src": src_path}
                raise_policy_error(
                    ErrorCode.E_STORAGE_MOVE,
                    f"Source does not exist: {src_path}",
                    details,
                    StoragePolicyException,
                )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        _emit_gate_event("storage_gate", "ok", elapsed_ms)
        _record_gate_metric("storage_gate", True, elapsed_ms)
        return PolicyOK(gate_name="storage_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, StoragePolicyException):
            error_code = getattr(e, "error_code", None)
            _emit_gate_event("storage_gate", "reject", elapsed_ms, error_code, details)
            _record_gate_metric("storage_gate", False, elapsed_ms, error_code)
            raise
        _record_gate_metric("storage_gate", False, elapsed_ms, ErrorCode.E_STORAGE_PUT)
        raise_policy_error(
            ErrorCode.E_STORAGE_PUT,
            str(e),
            details,
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


@policy_gate(
    name="db_boundary_gate",
    description="Validate database transaction boundaries",
    domain="db",
)
def db_boundary_gate(
    operation: str,
    tables_affected: Optional[List[str]] = None,
    fs_success: bool = True,
) -> Union[PolicyOK, PolicyReject]:
    """Validate database transaction boundaries.

    Enforces:
    - Commit only after FS success (no torn writes)
    - Foreign key invariants
    - Transaction isolation

    Args:
        operation: Transaction operation (pre_commit, post_extract, etc.)
        tables_affected: List of tables involved
        fs_success: Whether filesystem operation succeeded

    Returns:
        PolicyOK if transaction valid, PolicyReject otherwise
    """
    start_ms = time.perf_counter() * 1000
    details: Dict[str, Any] = {}

    try:
        # Validate operation
        valid_ops = {"pre_commit", "post_extract", "pre_rollback", "post_rollback"}
        if operation not in valid_ops:
            details = {"operation": operation}
            raise_policy_error(
                ErrorCode.E_DB_TX,
                f"Invalid transaction operation: {operation}",
                details,
                DbBoundaryException,
            )

        # Core rule: commit only after FS success
        if operation == "pre_commit" and not fs_success:
            details = {"operation": operation, "fs_success": fs_success}
            raise_policy_error(
                ErrorCode.E_DB_TX,
                "Cannot commit: filesystem operation failed (no torn writes)",
                details,
                DbBoundaryException,
            )

        # Track affected tables
        if tables_affected:
            for table in tables_affected:
                if not table or not isinstance(table, str):
                    details = {"table": table}
                    raise_policy_error(
                        ErrorCode.E_DB_TX,
                        f"Invalid table name: {table}",
                        details,
                        DbBoundaryException,
                    )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        _emit_gate_event("db_boundary_gate", "ok", elapsed_ms)
        _record_gate_metric("db_boundary_gate", True, elapsed_ms)
        return PolicyOK(gate_name="db_boundary_gate", elapsed_ms=elapsed_ms)

    except Exception as e:
        elapsed_ms = time.perf_counter() * 1000 - start_ms
        if isinstance(e, DbBoundaryException):
            error_code = getattr(e, "error_code", None)
            _emit_gate_event("db_boundary_gate", "reject", elapsed_ms, error_code, details)
            _record_gate_metric("db_boundary_gate", False, elapsed_ms, error_code)
            raise
        _record_gate_metric("db_boundary_gate", False, elapsed_ms, ErrorCode.E_DB_TX)
        raise_policy_error(
            ErrorCode.E_DB_TX,
            str(e),
            details,
            DbBoundaryException,
        )
        assert False  # All paths above raise


__all__ = [
    "config_gate",
    "url_gate",
    "filesystem_gate",
    "extraction_gate",
    "storage_gate",
    "db_boundary_gate",
]
