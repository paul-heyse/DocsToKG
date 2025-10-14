"""Chunk generation for hybrid retrieval ingestion."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Tuple

from .tokenization import sliding_window, tokenize_with_spans
from .types import DocumentInput


@dataclass(slots=True)
class ChunkDefinition:
    """Chunk boundaries prior to feature enrichment."""

    chunk_id: str
    vector_id: str
    text: str
    token_offset: Tuple[int, int]
    char_offset: Tuple[int, int]


_CHUNK_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


class Chunker:
    """Split documents into deterministic overlapping token windows."""

    def __init__(self, *, max_tokens: int = 800, overlap: int = 150) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must be within (0, max_tokens)")
        self._max_tokens = max_tokens
        self._overlap = overlap

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def overlap(self) -> int:
        return self._overlap

    def chunk(self, document: DocumentInput) -> List[ChunkDefinition]:
        tokens, spans = tokenize_with_spans(document.text)
        if not tokens:
            return [self._empty_chunk(document)]

        window_iter = sliding_window(tokens, self._max_tokens, self._overlap)
        chunks: List[ChunkDefinition] = []
        start_index = 0
        for idx, window_tokens in enumerate(window_iter):
            end_index = start_index + len(window_tokens)
            char_start = spans[start_index][0]
            char_end = spans[end_index - 1][1]
            text_slice = document.text[char_start:char_end]
            chunk_uuid = uuid.uuid5(
                _CHUNK_NAMESPACE,
                f"{document.doc_id}:{document.namespace}:{start_index}:{end_index}",
            )
            chunk_id = f"{document.doc_id}::chunk::{idx}"
            chunks.append(
                ChunkDefinition(
                    chunk_id=chunk_id,
                    vector_id=str(chunk_uuid),
                    text=text_slice,
                    token_offset=(start_index, end_index),
                    char_offset=(char_start, char_end),
                )
            )
            step = len(window_tokens) - self._overlap
            if step <= 0:
                break
            start_index += step
        return chunks

    def _empty_chunk(self, document: DocumentInput) -> ChunkDefinition:
        chunk_uuid = uuid.uuid5(_CHUNK_NAMESPACE, f"{document.doc_id}:{document.namespace}:empty")
        return ChunkDefinition(
            chunk_id=f"{document.doc_id}::chunk::0",
            vector_id=str(chunk_uuid),
            text="",
            token_offset=(0, 0),
            char_offset=(0, 0),
        )

