# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cancellation",
#   "purpose": "Tests for the cancellation primitives used by the ontology downloader.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Tests for the cancellation primitives used by the ontology downloader."""

from DocsToKG.OntologyDownload.cancellation import (
    CancellationToken,
    CancellationTokenGroup,
)


def test_tokens_created_after_cancel_all_are_cancelled() -> None:
    """Tokens created after ``cancel_all`` should start in a cancelled state."""

    group = CancellationTokenGroup()
    first = group.create_token()
    assert not first.is_cancelled()

    group.cancel_all()

    second = group.create_token()
    assert second.is_cancelled()

    third = CancellationToken()
    group.add_token(third)
    assert third.is_cancelled()
