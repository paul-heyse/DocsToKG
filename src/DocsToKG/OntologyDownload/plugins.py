"""Plugin discovery helpers for ontology downloader components."""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Any, MutableMapping, Optional, Protocol

__all__ = [
    "ResolverPlugin",
    "ValidatorPlugin",
    "load_resolver_plugins",
    "ensure_resolver_plugins",
    "load_validator_plugins",
]


class ResolverPlugin(Protocol):
    """Protocol describing resolver plugins discovered via entry points."""

    def plan(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime only
        """Plan a fetch for the provided ontology specification."""


class ValidatorPlugin(Protocol):
    """Protocol describing validator plugins discovered via entry points."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime only
        """Execute the validator."""


_RESOLVER_PLUGINS_LOADED = False
_VALIDATOR_PLUGINS_LOADED = False


def load_resolver_plugins(
    registry: MutableMapping[str, ResolverPlugin],
    *,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
) -> None:
    """Discover resolver plugins registered via ``entry_points``."""

    global _RESOLVER_PLUGINS_LOADED
    if _RESOLVER_PLUGINS_LOADED and not reload:
        return

    log = logger or logging.getLogger(__name__)
    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(
            "resolver plugin discovery failed",
            extra={"stage": "init", "error": str(exc)},
        )
        _RESOLVER_PLUGINS_LOADED = True
        return

    for entry in entry_points.select(group="docstokg.ontofetch.resolver"):
        try:
            candidate = entry.load()
            resolver = candidate() if isinstance(candidate, type) else candidate
            if not hasattr(resolver, "plan"):
                raise TypeError("resolver plugin must implement a plan method")
            name = getattr(resolver, "NAME", entry.name)
            registry[name] = resolver  # type: ignore[assignment]
            log.info(
                "resolver plugin registered",
                extra={"stage": "init", "resolver": name},
            )
        except Exception as exc:  # pragma: no cover - plugin failures are unpredictable
            log.warning(
                "resolver plugin failed",
                extra={"stage": "init", "resolver": entry.name, "error": str(exc)},
            )
    _RESOLVER_PLUGINS_LOADED = True


def ensure_resolver_plugins(
    registry: MutableMapping[str, ResolverPlugin],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Load resolver plugins exactly once per interpreter."""

    load_resolver_plugins(registry, logger=logger, reload=False)


def load_validator_plugins(
    registry: MutableMapping[str, ValidatorPlugin],
    *,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
) -> None:
    """Discover validator plugins registered via ``entry_points``."""

    global _VALIDATOR_PLUGINS_LOADED
    if _VALIDATOR_PLUGINS_LOADED and not reload:
        return

    log = logger or logging.getLogger(__name__)
    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(
            "validator plugin discovery failed",
            extra={"stage": "init", "error": str(exc)},
        )
        _VALIDATOR_PLUGINS_LOADED = True
        return

    for entry in entry_points.select(group="docstokg.ontofetch.validator"):
        try:
            handler = entry.load()
            if not callable(handler):
                raise TypeError("validator plugin must be callable")
            registry[entry.name] = handler  # type: ignore[assignment]
            log.info(
                "validator plugin registered",
                extra={"stage": "init", "validator": entry.name},
            )
        except Exception as exc:  # pragma: no cover - plugin failures are unpredictable
            log.warning(
                "validator plugin failed",
                extra={"stage": "init", "validator": entry.name, "error": str(exc)},
            )
    _VALIDATOR_PLUGINS_LOADED = True
