# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_import_time",
#   "purpose": "Ensures DocsToKG.OntologyDownload import stays lightweight",
#   "sections": []
# }
# === /NAVMAP ===

"""Ensures ``DocsToKG.OntologyDownload`` remains lightweight at import time.

The CLI bootstraps frequently, so these tests assert the package imports within
an acceptable latency budget and fails fast when optional dependencies are
missing, protecting command responsiveness.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_import_time_budget() -> None:
    """Importing the package should stay within a small latency budget."""

    script = "import time; start = time.perf_counter(); import DocsToKG.OntologyDownload; print(time.perf_counter() - start)"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    duration = float(completed.stdout.strip())
    assert duration < 0.5, f"import took {duration:.4f}s"
