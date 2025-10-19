"""Guardrail checks ensuring ontology_download tests follow harness guidance."""

from __future__ import annotations

from pathlib import Path


def test_pytest_monkeypatch_not_used():
    """Prevent reintroduction of the pytest.monkeypatch fixture in this suite."""

    root = Path(__file__).resolve().parent
    needle = "monkey" + "patch"
    offenders: list[Path] = []
    for path in root.glob("**/*.py"):
        if path.name in {"test_guardrails.py", "conftest.py"}:
            continue
        text = path.read_text()
        if needle in text:
            offenders.append(path.relative_to(root))
    assert not offenders, f"pytest.monkeypatch is disallowed in ontology_download tests: {offenders}"
