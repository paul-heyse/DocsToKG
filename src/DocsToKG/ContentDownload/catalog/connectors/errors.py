# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.connectors.errors",
#   "purpose": "Provider-Specific Exception Types.",
#   "sections": [
#     {
#       "id": "developmentprovidererror",
#       "name": "DevelopmentProviderError",
#       "anchor": "class-developmentprovidererror",
#       "kind": "class"
#     },
#     {
#       "id": "enterpriseprovidererror",
#       "name": "EnterpriseProviderError",
#       "anchor": "class-enterpriseprovidererror",
#       "kind": "class"
#     },
#     {
#       "id": "cloudprovidererror",
#       "name": "CloudProviderError",
#       "anchor": "class-cloudprovidererror",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Provider-Specific Exception Types

Extends the base exception types from base.py with provider-specific errors
for better error handling and diagnostics.
"""

from __future__ import annotations

from .base import (
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    ProviderOperationError,
)

__all__ = [
    "ProviderError",
    "ProviderConnectionError",
    "ProviderOperationError",
    "ProviderConfigError",
    "DevelopmentProviderError",
    "EnterpriseProviderError",
    "CloudProviderError",
]


class DevelopmentProviderError(ProviderError):
    """Error specific to Development provider (SQLite)."""

    pass


class EnterpriseProviderError(ProviderError):
    """Error specific to Enterprise provider (Postgres)."""

    pass


class CloudProviderError(ProviderError):
    """Error specific to Cloud provider (RDS + S3)."""

    pass
