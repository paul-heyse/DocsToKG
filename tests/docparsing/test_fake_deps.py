from __future__ import annotations

import sys

import pytest

from tests.docparsing.stubs import dependency_stubs


@pytest.fixture(autouse=True)
def _reset_fake_modules():
    """Ensure we start each test without cached fake modules."""

    for name in list(sys.modules):
        if name.startswith("tests.docparsing.fake_deps"):
            sys.modules.pop(name, None)
    for name in (
        "sentence_transformers",
        "vllm",
        "tqdm",
        "pydantic",
        "transformers",
        "docling_core",
        "docling_core.transforms",
        "docling_core.transforms.chunker",
        "docling_core.transforms.chunker.base",
        "docling_core.transforms.chunker.hybrid_chunker",
        "docling_core.transforms.chunker.hierarchical_chunker",
        "docling_core.transforms.chunker.tokenizer",
        "docling_core.transforms.chunker.tokenizer.huggingface",
        "docling_core.transforms.serializer",
        "docling_core.transforms.serializer.base",
        "docling_core.transforms.serializer.common",
        "docling_core.transforms.serializer.markdown",
        "docling_core.types",
        "docling_core.types.doc",
        "docling_core.types.doc.document",
    ):
        sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if name.startswith("tests.docparsing.fake_deps"):
            sys.modules.pop(name, None)


def test_dependency_stubs_registers_docling_core():
    dependency_stubs()
    import docling_core  # noqa: F401

    assert "docling_core.transforms.serializer.base" in sys.modules
    from docling_core.transforms.serializer.base import BaseDocSerializer

    serializer = BaseDocSerializer()
    assert serializer.post_process("text") == "text"


def test_dependency_stubs_respects_dense_dim_override():
    dependency_stubs(dense_dim=12)
    from vllm import LLM

    llm = LLM(dense_dim=12)
    vector = llm.embed(["alpha"])[0].outputs.embedding
    assert len(vector) == 12

    dependency_stubs()  # reset to default for subsequent tests
