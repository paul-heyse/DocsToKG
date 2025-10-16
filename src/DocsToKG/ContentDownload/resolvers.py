"""
Deprecated shim for the legacy ``DocsToKG.ContentDownload.resolvers`` module.

Resolver orchestration, registrations, and helper types are now defined in
``DocsToKG.ContentDownload.pipeline``.  This module re-exports that public API
so existing imports continue to function while callers migrate to the new
module.
"""

from .pipeline import *  # noqa: F401,F403
