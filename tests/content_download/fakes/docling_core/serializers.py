"""Serialiser shims for the content download fake dependency tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["RichSerializerProvider"]


@dataclass
class _SerializerRegistration:
    name: str
    serializer: Any


class RichSerializerProvider:
    """Very small stand-in for the production serializer provider."""

    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, serializer: Any) -> None:
        self._registry[name] = serializer

    def get(self, name: str) -> Any:
        return self._registry[name]

    def as_list(self) -> list[_SerializerRegistration]:
        return [
            _SerializerRegistration(name, serializer) for name, serializer in self._registry.items()
        ]
