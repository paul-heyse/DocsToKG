"""Dense embedding backend implementations.

These modules implement concrete dense embedding providers.  The package
wrapper ensures relative imports resolve cleanly for tooling such as
Sphinx AutoAPI.
"""

__all__ = [
    "qwen_vllm",
    "sentence_transformers",
    "tei",
    "fallback",
]

