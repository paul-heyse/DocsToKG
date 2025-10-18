from __future__ import annotations

from typing import Iterable, List, Sequence

__all__ = [
    "DocTagsDocument",
    "DoclingDocument",
    "PictureClassificationData",
    "PictureDescriptionData",
    "PictureItem",
    "PictureMoleculeData",
]


class DocTagsDocument:
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    @classmethod
    def from_doctags_and_image_pairs(
        cls,
        texts: Sequence[str],
        images: Iterable[object] | None = None,
        **_kwargs: object,
    ) -> "DocTagsDocument":
        return cls(texts)


class DoclingDocument:
    def __init__(self, paragraphs: Sequence[str], name: str) -> None:
        self.paragraphs = list(paragraphs)
        self.name = name

    @classmethod
    def load_from_doctags(cls, doc_tags: DocTagsDocument, document_name: str) -> "DoclingDocument":
        joined = "\n".join(doc_tags.texts)
        parts = [p.strip() for p in joined.split("\n\n") if p.strip()]
        if not parts:
            parts = [joined.strip() or "Synthetic paragraph"]
        return cls(parts, document_name)

    def __iter__(self):
        return iter(self.paragraphs)


class PictureClassificationData:
    predicted_classes: List[str] = []


class PictureDescriptionData:
    text: str = ""


class PictureMoleculeData:
    smi: str = ""


class PictureItem:
    def __init__(self) -> None:
        self.annotations: List[object] = []

    def caption_text(self, _doc: object) -> str:
        return ""
