"""Plugin discovery helpers for ontology downloader components."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from importlib import metadata
from typing import Any, Dict, MutableMapping, Optional, Protocol

__all__ = [
    "ResolverPlugin",
    "ValidatorPlugin",
    "load_resolver_plugins",
    "ensure_resolver_plugins",
    "load_validator_plugins",
    "ensure_plugins_loaded",
    "register_plugin_registry",
    "get_plugin_registry",
    "get_resolver_registry",
    "get_validator_registry",
    "list_registered_plugins",
    "get_registered_plugin_meta",
]


class ResolverPlugin(Protocol):
    """Protocol describing resolver plugins discovered via entry points."""

    def plan(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime only
        """Plan a fetch for the provided ontology specification."""


class ValidatorPlugin(Protocol):
    """Protocol describing validator plugins discovered via entry points."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime only
        """Execute the validator."""


_PLUGINS_LOCK = threading.Lock()
_PLUGINS_INITIALIZED = False
_RESOLVER_ENTRY_META: Dict[str, Dict[str, str]] = {}
_VALIDATOR_ENTRY_META: Dict[str, Dict[str, str]] = {}
_PLUGIN_REGISTRIES: Dict[str, MutableMapping[str, Any]] = {}
_RESOLVER_REGISTRY: MutableMapping[str, ResolverPlugin] = {}
_VALIDATOR_REGISTRY: MutableMapping[str, ValidatorPlugin] = {}


def _describe_plugin(obj: object) -> str:
    module = getattr(obj, "__module__", obj.__class__.__module__)
    name = getattr(obj, "__qualname__", obj.__class__.__name__)
    return f"{module}.{name}"


def _detect_entry_version(entry) -> str:
    dist = getattr(entry, "dist", None)
    if dist is not None:
        version = getattr(dist, "version", None)
        if version:
            return version
    module_name = getattr(entry, "module", "")
    root = module_name.split(".")[0] if module_name else module_name
    if root:
        try:
            return metadata.version(root)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional plugin
            return "unknown"
    return "unknown"


def _load_resolver_plugins_locked(
    registry: MutableMapping[str, ResolverPlugin],
    *,
    logger: logging.Logger,
) -> None:
    """Populate ``registry`` with resolver plugins discovered via entry points."""

    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "resolver plugin discovery failed",
            extra={"stage": "init", "error": str(exc)},
        )
        return

    for entry in entry_points.select(group="docstokg.ontofetch.resolver"):
        try:
            candidate = entry.load()
            resolver = candidate() if isinstance(candidate, type) else candidate
            if not hasattr(resolver, "plan"):
                raise TypeError("resolver plugin must implement a plan method")
            name = getattr(resolver, "NAME", entry.name)
            registry[name] = resolver  # type: ignore[assignment]
            _RESOLVER_ENTRY_META[name] = {
                "qualified": _describe_plugin(resolver),
                "version": _detect_entry_version(entry),
            }
            logger.info(
                "resolver plugin registered",
                extra={"stage": "init", "resolver": name},
            )
        except Exception as exc:  # pragma: no cover - plugin failures are unpredictable
            logger.warning(
                "resolver plugin failed",
                extra={"stage": "init", "resolver": entry.name, "error": str(exc)},
            )


def load_resolver_plugins(
    registry: MutableMapping[str, ResolverPlugin],
    *,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
) -> None:
    """Discover resolver plugins registered via ``entry_points``."""

    resolvers, _ = ensure_plugins_loaded(logger=logger, reload=reload)
    if registry is not resolvers:
        registry.update(resolvers)


def ensure_resolver_plugins(
    registry: MutableMapping[str, ResolverPlugin],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Load resolver plugins exactly once per interpreter."""

    load_resolver_plugins(registry, logger=logger, reload=False)


def _load_validator_plugins_locked(
    registry: MutableMapping[str, ValidatorPlugin],
    *,
    logger: logging.Logger,
) -> None:
    """Populate ``registry`` with validator plugins discovered via entry points."""

    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "validator plugin discovery failed",
            extra={"stage": "init", "error": str(exc)},
        )
        return

    for entry in entry_points.select(group="docstokg.ontofetch.validator"):
        try:
            handler = entry.load()
            if not callable(handler):
                raise TypeError("validator plugin must be callable")
            registry[entry.name] = handler  # type: ignore[assignment]
            _VALIDATOR_ENTRY_META[entry.name] = {
                "qualified": _describe_plugin(handler),
                "version": _detect_entry_version(entry),
            }
            logger.info(
                "validator plugin registered",
                extra={"stage": "init", "validator": entry.name},
            )
        except Exception as exc:  # pragma: no cover - plugin failures are unpredictable
            logger.warning(
                "validator plugin failed",
                extra={"stage": "init", "validator": entry.name, "error": str(exc)},
            )


def load_validator_plugins(
    registry: MutableMapping[str, ValidatorPlugin],
    *,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
) -> None:
    """Discover validator plugins registered via ``entry_points``."""

    _, validators = ensure_plugins_loaded(logger=logger, reload=reload)
    if registry is not validators:
        registry.update(validators)


def ensure_plugins_loaded(
    *,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
) -> tuple[MutableMapping[str, ResolverPlugin], MutableMapping[str, ValidatorPlugin]]:
    """Load resolver and validator plugins exactly once in a thread-safe manner."""

    global _PLUGINS_INITIALIZED

    log = logger or logging.getLogger(__name__)
    with _PLUGINS_LOCK:
        resolver_registry = _RESOLVER_REGISTRY
        validator_registry = _VALIDATOR_REGISTRY

        if reload:
            for name in list(_RESOLVER_ENTRY_META):
                resolver_registry.pop(name, None)
            for name in list(_VALIDATOR_ENTRY_META):
                validator_registry.pop(name, None)
            _RESOLVER_ENTRY_META.clear()
            _VALIDATOR_ENTRY_META.clear()
            _PLUGINS_INITIALIZED = False

        if not _PLUGINS_INITIALIZED:
            _load_resolver_plugins_locked(resolver_registry, logger=log)
            _load_validator_plugins_locked(validator_registry, logger=log)
            _PLUGINS_INITIALIZED = True

        return resolver_registry, validator_registry


def get_resolver_registry(
    *,
    logger: Optional[logging.Logger] = None,
) -> MutableMapping[str, ResolverPlugin]:
    """Return the resolver plugin registry, ensuring plugins are loaded once."""

    resolvers, _ = ensure_plugins_loaded(logger=logger, reload=False)
    return resolvers


def get_validator_registry(
    *,
    logger: Optional[logging.Logger] = None,
) -> MutableMapping[str, ValidatorPlugin]:
    """Return the validator plugin registry, ensuring plugins are loaded once."""

    _, validators = ensure_plugins_loaded(logger=logger, reload=False)
    return validators


def register_plugin_registry(kind: str, registry: MutableMapping[str, Any]) -> None:
    """Register the in-memory plugin registry for discovery helpers."""

    global _RESOLVER_REGISTRY, _VALIDATOR_REGISTRY

    with _PLUGINS_LOCK:
        if kind == "resolver":
            if _RESOLVER_REGISTRY is not registry:
                registry.update(_RESOLVER_REGISTRY)
                _RESOLVER_REGISTRY = registry  # type: ignore[assignment]
        elif kind == "validator":
            if _VALIDATOR_REGISTRY is not registry:
                registry.update(_VALIDATOR_REGISTRY)
                _VALIDATOR_REGISTRY = registry  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown plugin kind: {kind}")
        _PLUGIN_REGISTRIES[kind] = registry


def get_plugin_registry(kind: str) -> MutableMapping[str, Any]:
    """Return the registered plugin registry for ``kind``."""

    try:
        return _PLUGIN_REGISTRIES[kind]
    except KeyError as exc:
        raise ValueError(f"No plugin registry registered for kind '{kind}'") from exc


def list_registered_plugins(kind: str) -> "OrderedDict[str, str]":
    """Return mapping of plugin names to qualified identifiers for ``kind``."""

    if kind == "resolver":
        registry = get_resolver_registry()
    elif kind == "validator":
        registry = get_validator_registry()
    else:
        raise ValueError(f"Unknown plugin kind: {kind}")
    items = {name: _describe_plugin(obj) for name, obj in registry.items()}
    return OrderedDict(sorted(items.items()))


def get_registered_plugin_meta(kind: str) -> Dict[str, Dict[str, str]]:
    """Return metadata captured for entry-point registered plugins."""

    ensure_plugins_loaded()

    if kind == "resolver":
        return dict(_RESOLVER_ENTRY_META)
    if kind == "validator":
        return dict(_VALIDATOR_ENTRY_META)
    raise ValueError(f"Unknown plugin kind: {kind}")
