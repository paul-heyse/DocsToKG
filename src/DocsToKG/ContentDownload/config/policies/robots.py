# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.policies.robots",
#   "purpose": "Robots.txt policy configuration.",
#   "sections": [
#     {
#       "id": "robotspolicy",
#       "name": "RobotsPolicy",
#       "anchor": "class-robotspolicy",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Robots.txt policy configuration.

Controls respect for robots.txt and cache settings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RobotsPolicy(BaseModel):
    """robots.txt handling policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Whether to respect robots.txt")
    ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")


__all__ = ["RobotsPolicy"]
