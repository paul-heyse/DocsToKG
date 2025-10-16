"""CLI shim for validator worker execution."""

from __future__ import annotations

from .validation_core import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover
    main()
