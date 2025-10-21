"""
Resolver Registry with Pydantic v2 Integration

Provides a clean registry pattern for ContentDownload resolvers:
- @register(name) decorator for resolver registration
- Dynamic resolver instantiation by name
- Config-driven ordering and enablement
- Per-resolver config overrides via from_config() classmethod

This module integrates with the new ContentDownloadConfig (Pydantic v2)
to enable flexible resolver composition.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type

from DocsToKG.ContentDownload.config import ContentDownloadConfig

_LOGGER = logging.getLogger(__name__)

# ============================================================================
# Registry
# ============================================================================

_REGISTRY: Dict[str, Type[Any]] = {}


def register(name: str):
    """
    Decorator to register a resolver in the global registry.

    Usage:
        @register("unpaywall")
        class UnpaywallResolver:
            ...

    Args:
        name: Resolver name (must match ResolversConfig field name)

    Returns:
        Decorator function
    """

    def deco(cls: Type[Any]) -> Type[Any]:
        if name in _REGISTRY:
            _LOGGER.warning(f"Overriding already-registered resolver: {name}")
        _REGISTRY[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        _LOGGER.debug(f"Registered resolver: {name} → {cls.__name__}")
        return cls

    return deco


def get_registry() -> Dict[str, Type[Any]]:
    """
    Get the global resolver registry (copy).

    Returns:
        Dict of name → resolver class
    """
    return dict(_REGISTRY)


def get_resolver_class(name: str) -> Type[Any]:
    """
    Lookup resolver class by name.

    Args:
        name: Resolver name

    Returns:
        Resolver class

    Raises:
        ValueError: If resolver not found
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown resolver: {name!r}. Available: {available}")
    return _REGISTRY[name]


# ============================================================================
# Builder
# ============================================================================


def build_resolvers(
    config: ContentDownloadConfig,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Build resolver instances from config with ordering and enablement.

    Respects:
    - resolvers.order (execution sequence)
    - Per-resolver enabled flag
    - Per-resolver config overrides
    - Optional overrides dict (CLI, programmatic)

    Args:
        config: ContentDownloadConfig (Pydantic v2 model)
        overrides: Optional overrides dict (e.g., from CLI)

    Returns:
        List of resolver instances in configured order

    Raises:
        ValueError: If unknown resolver in order or other config issues
    """
    resolvers: List[Any] = []
    overrides = overrides or {}

    for resolver_name in config.resolvers.order:
        # Get resolver config
        resolver_cfg = getattr(config.resolvers, resolver_name, None)
        if resolver_cfg is None:
            raise ValueError(
                f"Resolver {resolver_name!r} not in config. "
                f"Available: {list(config.resolvers.__dict__.keys())}"
            )

        # Check enabled
        if not resolver_cfg.enabled:
            _LOGGER.debug(f"Skipping disabled resolver: {resolver_name}")
            continue

        # Get resolver class
        try:
            resolver_cls = get_resolver_class(resolver_name)
        except ValueError as e:
            _LOGGER.warning(f"Resolver not available: {resolver_name} — {e}")
            continue

        # Instantiate resolver
        try:
            # Prefer from_config classmethod if available
            if hasattr(resolver_cls, "from_config"):
                inst = resolver_cls.from_config(resolver_cfg, config, overrides)
            else:
                # Fallback: pass config to __init__ if it accepts it
                inst = resolver_cls(config=resolver_cfg)
            resolvers.append(inst)
            _LOGGER.debug(f"Built resolver: {resolver_name} ({resolver_cls.__name__})")
        except Exception as e:
            _LOGGER.error(f"Failed to instantiate resolver {resolver_name}: {e}")
            raise

    _LOGGER.info(
        f"Built {len(resolvers)} resolvers in order: "
        f"{[r._registry_name if hasattr(r, '_registry_name') else r.__class__.__name__ for r in resolvers]}"
    )

    return resolvers


# ============================================================================
# Protocol (for type hints)
# ============================================================================


class ResolverProtocol(Protocol):
    """
    Protocol for resolver implementations.

    Resolvers should implement this interface to work with the registry.
    """

    _registry_name: ClassVar[str]
    """Registry name (set by @register decorator)."""

    @classmethod
    def from_config(
        cls,
        resolver_cfg: Any,
        root_cfg: ContentDownloadConfig,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ResolverProtocol:
        """
        Factory method to create resolver from Pydantic config.

        Args:
            resolver_cfg: Resolver-specific config (e.g., UnpaywallConfig)
            root_cfg: Root ContentDownloadConfig
            overrides: Optional overrides dict

        Returns:
            Instantiated resolver
        """
        ...

    def resolve(self, artifact: Any) -> List[Any]:
        """
        Resolve download plans for an artifact.

        Args:
            artifact: Work item/artifact to resolve

        Returns:
            List of DownloadPlan objects
        """
        ...
