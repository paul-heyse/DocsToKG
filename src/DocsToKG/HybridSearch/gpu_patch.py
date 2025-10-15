"""
Compatibility shim for legacy GPU monkey patch imports.

Older services imported ``DocsToKG.HybridSearch.gpu_patch`` to force GPU-only
behaviour. The functionality now lives directly inside the core modules, but
this stub keeps the import path valid. ``apply_patch`` is a no-op retained for
backwards compatibility.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_patch() -> None:
    """Preserve legacy GPU patch API while performing no operation.

    Args:
        None

    Returns:
        None
    """
    logger.debug("HybridSearch GPU patch already integrated; nothing to apply.")


# Run immediately on import for parity with the historic behaviour.
apply_patch()
