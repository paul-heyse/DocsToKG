# === NAVMAP v1 ===
# {
#   "module": "tests.test_basic",
#   "purpose": "Pytest coverage for basic scenarios",
#   "sections": [
#     {
#       "id": "test-ping",
#       "name": "test_ping",
#       "anchor": "function-test-ping",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Package Sanity Check

This module provides a fast smoke test that ensures the package exports
its lightweight `ping` utility, which is also used as a health check in
deployment probes.

Usage:
    pytest tests/test_basic.py
"""

from DocsToKG import ping

# --- Test Cases ---


def test_ping():
    assert ping() == "pong"
