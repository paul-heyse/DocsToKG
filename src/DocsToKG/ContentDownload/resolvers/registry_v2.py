"""
Unified Resolver Registry with Pydantic v2 Integration

Bridges the new registry_v2 pattern with existing ResolverRegistry.
- @register_v2(name) decorator for new resolvers
- Automatic discovery of existing ResolverRegistry instances
- Config-driven resolver instantiation
- Backward compatible with existing resolver system
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type

from DocsToKG.ContentDownload.config import ContentDownloadConfig

_LOGGER = logging.getLogger(__name__)

# ============================================================================
# Registry Bridge
# ============================================================================

_REGISTRY_V2: Dict[str, Type[Any]] = {}


def _discover_existing_resolvers() -> Dict[str, Type[Any]]:
    """Discover resolvers from existing ResolverRegistry."""
    try:
        from DocsToKG.ContentDownload.resolvers.base import ResolverRegistry

        discovered = dict(ResolverRegistry._providers or {})
        _LOGGER.debug(f"Discovered {len(discovered)} existing resolvers from ResolverRegistry")
        return discovered
    except Exception as e:
        _LOGGER.warning(f"Could not discover existing resolvers: {e}")
        return {}


def register_v2(name: str):
    """Decorator to register a resolver in registry_v2."""

    def deco(cls: Type[Any]) -> Type[Any]:
        if name in _REGISTRY_V2:
            _LOGGER.warning(f"Overriding already-registered v2 resolver: {name}")
        _REGISTRY_V2[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        _LOGGER.debug(f"Registered v2 resolver: {name} → {cls.__name__}")
        return cls

    return deco


def get_registry() -> Dict[str, Type[Any]]:
    """Get combined registry (v2 + discovered existing)."""
    existing = _discover_existing_resolvers()
    combined = {**existing, **_REGISTRY_V2}
    return combined


def get_resolver_class(name: str) -> Type[Any]:
    """Lookup resolver class by name."""
    registry = get_registry()
    if name not in registry:
        available = sorted(registry.keys())
        raise ValueError(f"Unknown resolver: {name!r}. Available: {available}")
    return registry[name]


# ============================================================================
# Builder
# ============================================================================


def build_resolvers(
    config: ContentDownloadConfig,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Build resolver instances from config with ordering and enablement."""
    resolvers: List[Any] = []
    overrides = overrides or {}

    for resolver_name in config.resolvers.order:
        resolver_cfg = getattr(config.resolvers, resolver_name, None)
        if resolver_cfg is None:
            _LOGGER.debug(f"Skipping resolver {resolver_name} (no config)")
            continue

        if not resolver_cfg.enabled:
            _LOGGER.debug(f"Skipping disabled resolver: {resolver_name}")
            continue

        try:
            resolver_cls = get_resolver_class(resolver_name)
        except ValueError as e:
            _LOGGER.warning(f"Resolver not available: {resolver_name} — {e}")
            continue

        try:
            # Try from_config first (new pattern)
            if hasattr(resolver_cls, "from_config"):
                inst = resolver_cls.from_config(resolver_cfg, config, overrides)
            # Fallback to existing pattern (expects no args or config kwarg)
            else:
                try:
                    inst = resolver_cls(config=resolver_cfg)
                except TypeError:
                    inst = resolver_cls()

            resolvers.append(inst)
            _LOGGER.debug(f"Built resolver: {resolver_name} ({resolver_cls.__name__})")
        except Exception as e:
            _LOGGER.error(f"Failed to instantiate {resolver_name}: {e}")
            raise

    _LOGGER.info(
        f"Built {len(resolvers)} resolvers in order: "
        f"{[r._registry_name if hasattr(r, '_registry_name') else getattr(r, 'name', r.__class__.__name__) for r in resolvers]}"
    )

    return resolvers


# ============================================================================
# Protocol (for type hints)
# ============================================================================


class ResolverProtocol(Protocol):
    """Protocol for resolver implementations."""

    _registry_name: ClassVar[str]

    @classmethod
    def from_config(
        cls,
        resolver_cfg: Any,
        root_cfg: ContentDownloadConfig,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ResolverProtocol:
        """Factory method to create resolver from Pydantic config."""
        ...

    def resolve(self, artifact: Any) -> List[Any]:
        """Resolve download plans for an artifact."""
        ...
