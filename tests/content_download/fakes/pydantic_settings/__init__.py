"""Minimal stub for :mod:`pydantic_settings` used in tests."""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["BaseSettings", "SettingsConfigDict"]


def SettingsConfigDict(**kwargs: Any) -> Dict[str, Any]:
    return dict(kwargs)


class BaseSettings:
    model_config: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "BaseSettings":
        return cls(**data)
