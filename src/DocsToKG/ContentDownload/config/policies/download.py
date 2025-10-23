# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.config.policies.download",
#   "purpose": "Download policy configuration.",
#   "sections": [
#     {
#       "id": "downloadpolicy",
#       "name": "DownloadPolicy",
#       "anchor": "class-downloadpolicy",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Download policy configuration.

Controls how downloads are performed and verified.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DownloadPolicy(BaseModel):
    """Download operation policies."""

    model_config = ConfigDict(extra="forbid")

    atomic_write: bool = Field(
        default=True,
        description="Use atomic writes (tmp file + rename)",
    )
    verify_content_length: bool = Field(
        default=True,
        description="Verify Content-Length header matches actual bytes",
    )
    chunk_size_bytes: int = Field(
        default=1 << 20,
        ge=4096,
        description="Read chunk size in bytes",
    )
    max_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Maximum allowed download size (None = unlimited)",
    )


__all__ = ["DownloadPolicy"]
