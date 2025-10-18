from __future__ import annotations

from types import SimpleNamespace

__all__ = ["BaseChunk"]


class BaseChunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.meta = SimpleNamespace(doc_items=[], prov=[])
