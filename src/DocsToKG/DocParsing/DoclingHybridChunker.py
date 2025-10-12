# install (pick one tokenizer family)
# pip install "docling-core[chunking]" transformers     # HuggingFace tokenizer
# pip install "docling-core[chunking-openai]" tiktoken  # OpenAI/tiktoken tokenizer

from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

# ---- serializer to inject rich metadata into the TEXT you embed ----
from typing import Any
from typing_extensions import override
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureItem,
    PictureClassificationData,
    PictureDescriptionData,
    PictureMoleculeData,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
    MarkdownPictureSerializer,
    MarkdownParams,
)
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)


class RichSerializerProvider(ChunkingSerializerProvider):
    """Adds headings/captions (default), renders tables as Markdown,
    and turns picture annotations into text, so embedders see the context."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        class AnnotationPictureSerializer(MarkdownPictureSerializer):
            @override
            def serialize(
                self,
                *,
                item: PictureItem,
                doc_serializer: BaseDocSerializer,
                doc: DoclingDocument,
                **_: Any,
            ) -> SerializationResult:
                parts = []
                # include any available picture annotations in-line
                for ann in item.annotations:
                    if isinstance(ann, PictureClassificationData) and ann.predicted_classes:
                        parts.append(f"Picture type: {ann.predicted_classes[0].class_name}")
                    elif isinstance(ann, PictureMoleculeData):
                        parts.append(f"SMILES: {ann.smi}")
                    elif isinstance(ann, PictureDescriptionData):
                        parts.append(f"Picture description: {ann.text}")
                text = "\n".join(parts) or "<!-- image -->"
                text = doc_serializer.post_process(text=text)
                return create_ser_result(text=text, span_source=item)

        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # tables → Markdown (easier for embedders)
            picture_serializer=AnnotationPictureSerializer(),
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


# ---- tokenizer aligned to BERT/SPLADE (512 max) ----
hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=512)

# ---- hybrid chunker: no overlap; split/merge around 512; merge small peers ----
chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,  # fills undersized siblings under same headings/captions
    serializer_provider=RichSerializerProvider(),
)

# usage
# dl_doc = ...  # your DoclingDocument
# chunks = list(chunker.chunk(dl_doc=dl_doc))
# texts  = [chunker.contextualize(c) for c in chunks]  # feed these to BM25/SPLADE/Qwen
