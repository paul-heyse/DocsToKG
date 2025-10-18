from __future__ import annotations

from typing import List

__all__ = ["AutoTokenizer"]


class AutoTokenizer:
    def __init__(self, model_name: str, use_fast: bool = True) -> None:
        self.model_name = model_name
        self.use_fast = use_fast

    @classmethod
    def from_pretrained(cls, model_name: str, use_fast: bool = True) -> "AutoTokenizer":
        return cls(model_name, use_fast=use_fast)

    def __call__(self, text: str) -> List[str]:
        return text.split()
