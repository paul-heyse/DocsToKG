"""Minimal stub for :mod:`pydantic_core` used by tests."""


class ValidationError(Exception):
    """Compatibility shim for the real pydantic-core ValidationError."""


__all__ = ["ValidationError"]
