from __future__ import annotations

__all__ = ["MarkdownParams", "MarkdownPictureSerializer", "MarkdownTableSerializer"]


class MarkdownParams:
    def __init__(self, image_placeholder: str = "") -> None:
        self.image_placeholder = image_placeholder


class MarkdownPictureSerializer:
    pass


class MarkdownTableSerializer:
    pass
