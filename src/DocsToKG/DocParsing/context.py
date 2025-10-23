# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.context",
#   "purpose": "Typed runtime context shared across DocParsing stages.",
#   "sections": [
#     {
#       "id": "parsingcontext",
#       "name": "ParsingContext",
#       "anchor": "class-parsingcontext",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Typed runtime context shared across DocParsing stages.

The :class:`ParsingContext` dataclass captures the run-scoped attributes that
the CLI and pipeline stages exchange. It replaces loosely typed dictionaries so
callers benefit from IDE completion, static analysis, and centralised default
management. The context can also serialise itself into manifest-friendly
payloads when stages record configuration snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import _manifest_value


@dataclass(slots=True)
class ParsingContext:
    """Runtime metadata describing a DocParsing invocation."""

    run_id: str
    data_root: Path
    in_dir: Path | None = None
    out_dir: Path | None = None
    doctags_dir: Path | None = None
    chunks_dir: Path | None = None
    vectors_dir: Path | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    soft_barrier_margin: int | None = None
    serializer_provider: str | None = None
    shard_count: int | None = None
    shard_index: int | None = None
    workers: int | None = None
    files_parallel: int | None = None
    validate_only: bool | None = None
    plan_only: bool | None = None
    resume: bool | None = None
    force: bool | None = None
    inject_anchors: bool | None = None
    offline: bool | None = None
    vector_format: str = "jsonl"
    profile: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise paths eagerly to simplify downstream usage."""

        self.data_root = self._resolve_path(self.data_root)
        if self.in_dir is not None:
            self.in_dir = self._resolve_path(self.in_dir)
        if self.out_dir is not None:
            self.out_dir = self._resolve_path(self.out_dir)
        if self.doctags_dir is not None:
            self.doctags_dir = self._resolve_path(self.doctags_dir)
        if self.chunks_dir is not None:
            self.chunks_dir = self._resolve_path(self.chunks_dir)
        if self.vectors_dir is not None:
            self.vectors_dir = self._resolve_path(self.vectors_dir)

    @staticmethod
    def _resolve_path(value: Path | str) -> Path:
        """Return ``value`` coerced to an absolute :class:`Path`."""

        return Path(value).expanduser().resolve()

    @classmethod
    def field_names(cls) -> frozenset[str]:
        """Expose recognised field names (excluding the ``extra`` payload)."""

        return frozenset(name for name in cls.__dataclass_fields__ if name != "extra")  # type: ignore[attr-defined]

    def apply_config(self, cfg: Any) -> None:
        """Populate context attributes from a stage configuration dataclass."""

        manifest = cfg.to_manifest() if hasattr(cfg, "to_manifest") else {}
        recognised = self.field_names()

        for name in recognised:
            if not hasattr(cfg, name):
                continue
            value = getattr(cfg, name)
            if value is None:
                continue
            if isinstance(value, Path):
                value = self._resolve_path(value)
            setattr(self, name, value)

        for key, value in manifest.items():
            if key in recognised:
                continue
            if value is None:
                continue
            self.extra[key] = value
        self.vector_format = str(self.vector_format or "jsonl").lower()

    def merge_extra(self, mapping: Mapping[str, Any]) -> None:
        """Merge arbitrary manifest-safe metadata into the context."""

        for key, value in mapping.items():
            if value is None:
                continue
            self.extra[key] = value

    def update_extra(self, **values: Any) -> None:
        """Convenience helper mirroring :meth:`dict.update` with filtering."""

        self.merge_extra(values)

    def to_manifest(self) -> dict[str, Any]:
        """Serialise the context to a manifest-friendly dictionary."""

        payload: dict[str, Any] = {}
        for name in self.field_names():
            if name == "extra":
                continue
            value = getattr(self, name)
            if value is None:
                continue
            payload[name] = _manifest_value(value)
        for key, value in self.extra.items():
            if value is None:
                continue
            payload[key] = _manifest_value(value)
        return payload

    def copy(self) -> "ParsingContext":
        """Return a shallow copy suitable for isolated mutation."""

        clone = ParsingContext(
            run_id=self.run_id,
            data_root=self.data_root,
            in_dir=self.in_dir,
            out_dir=self.out_dir,
            doctags_dir=self.doctags_dir,
            chunks_dir=self.chunks_dir,
            vectors_dir=self.vectors_dir,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            soft_barrier_margin=self.soft_barrier_margin,
            serializer_provider=self.serializer_provider,
            shard_count=self.shard_count,
            shard_index=self.shard_index,
            workers=self.workers,
            files_parallel=self.files_parallel,
            validate_only=self.validate_only,
            plan_only=self.plan_only,
            resume=self.resume,
            force=self.force,
            inject_anchors=self.inject_anchors,
            offline=self.offline,
            vector_format=self.vector_format,
            profile=self.profile,
        )
        clone.extra = dict(self.extra)
        return clone
