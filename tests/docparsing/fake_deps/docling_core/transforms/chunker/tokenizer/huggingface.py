from __future__ import annotations

from tests.docparsing.fake_deps.transformers import AutoTokenizer

__all__ = ["HuggingFaceTokenizer"]


class HuggingFaceTokenizer:
    def __init__(self, tokenizer: AutoTokenizer, max_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return max(1, len(self.tokenizer(text)))
