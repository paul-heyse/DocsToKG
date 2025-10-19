from __future__ import annotations

from typing import List

from .base import BaseChunk

__all__ = ["HybridChunker"]


class HybridChunker:
    def __init__(self, tokenizer, **_kwargs) -> None:
        self.tokenizer = tokenizer

    def chunk(self, dl_doc) -> List[BaseChunk]:
        return [BaseChunk(text) for text in getattr(dl_doc, "paragraphs", [])]

    def contextualize(self, chunk: BaseChunk) -> str:
        return chunk.text
