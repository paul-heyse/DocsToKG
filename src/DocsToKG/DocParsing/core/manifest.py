"""Manifest bookkeeping utilities shared across DocParsing stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

__all__ = [
    "ResumeController",
    "should_skip_output",
]


def should_skip_output(
    output_path: Path,
    manifest_entry: Optional[Mapping[str, object]],
    input_hash: str,
    resume: bool,
    force: bool,
) -> bool:
    """Return ``True`` when resume/skip conditions indicate work can be skipped."""

    if not resume or force:
        return False
    if not output_path.exists():
        return False
    if not manifest_entry or not isinstance(manifest_entry, Mapping):
        return False

    status = manifest_entry.get("status")
    if status not in {"success", "skip"}:
        return False

    stored_hash = manifest_entry.get("input_hash")
    return stored_hash == input_hash


@dataclass(slots=True)
class ResumeController:
    """Centralize resume/force decisions using manifest metadata."""

    resume: bool
    force: bool
    manifest_index: Optional[Mapping[str, Mapping[str, object]]] = None

    def entry(self, doc_id: str) -> Optional[Mapping[str, object]]:
        """Return the manifest entry associated with ``doc_id`` when available."""

        if not self.manifest_index:
            return None
        return self.manifest_index.get(doc_id)

    def should_skip(
        self, doc_id: str, output_path: Path, input_hash: str
    ) -> Tuple[bool, Optional[Mapping[str, object]]]:
        """Return ``True`` when work for ``doc_id`` can be safely skipped."""

        entry = self.entry(doc_id)
        skip = should_skip_output(output_path, entry, input_hash, self.resume, self.force)
        return skip, entry

    def should_process(
        self, doc_id: str, output_path: Path, input_hash: str
    ) -> Tuple[bool, Optional[Mapping[str, object]]]:
        """Return ``True`` when ``doc_id`` requires processing."""

        skip, entry = self.should_skip(doc_id, output_path, input_hash)
        return not skip, entry
