"""
Resolver subsystem for ContentDownload.

Each resolver implements the Resolver protocol:
  - name: str
  - resolve(artifact, session, ctx, telemetry, run_id) -> ResolverResult

Canonical types (from api module):
  - DownloadPlan: url + resolver_name + optional hints
  - ResolverResult: plans (Sequence) + notes (Mapping)

Usage example:
    from DocsToKG.ContentDownload.api import DownloadPlan, ResolverResult

    class MyResolver:
        name = "my_resolver"
        
        def resolve(self, artifact, session, ctx, telemetry, run_id):
            # Attempt to find a PDF URL
            url = get_pdf_url(artifact, session)
            if not url:
                return ResolverResult(plans=[])  # Nothing to offer
            
            # Return one or more plans
            plan = DownloadPlan(url=url, resolver_name=self.name)
            return ResolverResult(plans=[plan])
"""

from __future__ import annotations

from .base import Resolver

__all__ = ["Resolver"]
