"""Policy subsystem: defense-in-depth gates for safety enforcement.

One central registry of policy gates that wrap every I/O boundary.
Each gate returns OK or Reject with structured error codes.

Modules:
- errors: Error catalog and contracts (canonical ErrorCode enum)
- registry: Central gate registry and decorator
- gates: Six concrete gates (config, URL, path, extraction, storage, DB)
- metrics: Per-gate telemetry counters
"""

from DocsToKG.OntologyDownload.policy.errors import (
    ConfigurationPolicyException,
    ErrorCode,
    ExtractionPolicyException,
    FilesystemPolicyException,
    PolicyException,
    PolicyOK,
    PolicyReject,
    PolicyResult,
    StoragePolicyException,
    URLPolicyException,
)

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
]
