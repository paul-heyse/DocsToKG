"""
Package Sanity Check

This module provides a fast smoke test that ensures the package exports
its lightweight `ping` utility, which is also used as a health check in
deployment probes.

Usage:
    pytest tests/test_basic.py
"""

from DocsToKG import ping


def test_ping():
    assert ping() == "pong"
