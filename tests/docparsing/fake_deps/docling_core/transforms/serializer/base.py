from __future__ import annotations

from types import SimpleNamespace

__all__ = ["BaseDocSerializer", "SerializationResult"]


class BaseDocSerializer:
    def post_process(self, text: str) -> str:
        return text


class SerializationResult(SimpleNamespace):
    pass
