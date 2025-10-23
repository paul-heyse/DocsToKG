# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.manifest",
#   "purpose": "Resume and manifest decision helpers for DocParsing stages.",
#   "sections": [
#     {
#       "id": "should-skip-output",
#       "name": "should_skip_output",
#       "anchor": "function-should-skip-output",
#       "kind": "function"
#     },
#     {
#       "id": "resumecontroller",
#       "name": "ResumeController",
#       "anchor": "class-resumecontroller",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Resume and manifest decision helpers for DocParsing stages.

Chunking and embedding maintain JSONL manifests that record the hash of the
inputs they processed. This module encapsulates the skip/resume heuristics that
interpret those manifests, providing a ``ResumeController`` and helper
functions so stages can avoid redundant work while still supporting ``--force``
and manual overrides.
"""

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
    candidate_path = output_path
    if manifest_entry and isinstance(manifest_entry, Mapping):
        recorded_path = manifest_entry.get("output_path")
        if recorded_path:
            candidate_path = Path(str(recorded_path))
    if not Path(candidate_path).exists():
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

    def can_skip_without_hash(
        self, doc_id: str, output_path: Path
    ) -> Tuple[bool, Optional[Mapping[str, object]]]:
        """Return ``True`` when manifest metadata alone justifies skipping.

        The predicate implements a fast path for planners and other tooling that
        only need to confirm that successful outputs already exist. It honours
        the resume/force flags and checks that the manifest recorded a
        successful or skipped status without touching the input payload.
        """

        entry = self.entry(doc_id)
        if not self.resume or self.force:
            return False, entry
        if not entry or not isinstance(entry, Mapping):
            return False, entry

        status = entry.get("status")
        if status not in {"success", "skip"}:
            return False, entry

        candidate_path = output_path
        if entry and isinstance(entry, Mapping):
            recorded_path = entry.get("output_path")
            if recorded_path:
                candidate_path = Path(str(recorded_path))
        if not Path(candidate_path).exists():
            return False, entry

        return True, entry

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
