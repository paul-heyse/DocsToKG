"""Backward-compatible re-export of feature utilities.

Historically tests imported ``DocsToKG.HybridSearch.devtools.features`` for the
tokeniser and deterministic feature generator. The canonical implementation now
lives in :mod:`DocsToKG.HybridSearch.features`. This module keeps the old import
path viable by forwarding every symbol, ensuring regression harnesses and IDE
navigation continue to work while the README and docs point at the new location.
"""

from ..features import *  # noqa: F401,F403
