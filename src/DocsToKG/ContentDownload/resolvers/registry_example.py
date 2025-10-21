"""
Example Resolver Implementation with Pydantic v2 Config

Shows best practices for implementing resolvers that integrate with
the new Pydantic v2 config system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from DocsToKG.ContentDownload.config import ContentDownloadConfig, UnpaywallConfig
from DocsToKG.ContentDownload.resolvers.registry_v2 import register


@register("unpaywall_example")
class UnpaywallExampleResolver:
    """
    Example resolver showing integration with Pydantic v2 config.

    This resolver demonstrates:
    - @register decorator for auto-registration
    - from_config classmethod for config-driven instantiation
    - Per-resolver config overrides (email, timeout)
    """

    def __init__(
        self,
        email: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Initialize resolver.

        Args:
            email: Optional email for Unpaywall API
            timeout: Optional timeout override
            **kwargs: Additional config
        """
        self.email = email
        self.timeout = timeout
        self.kwargs = kwargs

    @classmethod
    def from_config(
        cls,
        resolver_cfg: UnpaywallConfig,
        root_cfg: ContentDownloadConfig,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> UnpaywallExampleResolver:
        """
        Factory: create resolver from Pydantic config.

        Demonstrates how to extract config fields and handle overrides.

        Args:
            resolver_cfg: UnpaywallConfig instance (from ResolversConfig.unpaywall)
            root_cfg: Root ContentDownloadConfig
            overrides: Optional programmatic overrides

        Returns:
            Instantiated resolver
        """
        overrides = overrides or {}

        # Extract email from resolver config
        email = resolver_cfg.email

        # Extract timeout: prefer resolver override, fallback to root HTTP config
        timeout = resolver_cfg.timeout_read_s or root_cfg.http.timeout_read_s

        # Allow CLI/programmatic overrides
        if "email" in overrides:
            email = overrides["email"]
        if "timeout" in overrides:
            timeout = overrides["timeout"]

        return cls(email=email, timeout=timeout)

    def resolve(self, artifact: Any) -> List[Any]:
        """
        Resolve download plans for artifact.

        Args:
            artifact: Work item to resolve

        Returns:
            List of DownloadPlan objects
        """
        # TODO: Implement actual resolution logic
        # This is just a template showing the interface
        return []
