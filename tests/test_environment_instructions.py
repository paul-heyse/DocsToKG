from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVRC = ROOT / ".envrc"
BOOTSTRAP = ROOT / "scripts" / "bootstrap_env.sh"
README = ROOT / "README.md"
DOCS_SETUP = ROOT / "docs" / "02-setup" / "index.md"
AGENTS = ROOT / "openspec" / "AGENTS.md"


def test_envrc_configures_virtualenv_and_pythonpath() -> None:
    text = ENVRC.read_text(encoding="utf-8")
    assert 'export VIRTUAL_ENV="$VENVP"' in text
    assert 'PATH_add "$VIRTUAL_ENV/bin"' in text
    assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in text


def test_bootstrap_script_installs_project() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'python -m venv "$VENV_PATH"' in text
    assert '"$VENV_PATH/bin/pip" install -e .' in text
    assert os.access(BOOTSTRAP, os.X_OK), "bootstrap script must be executable"


def test_documentation_mentions_bootstrap_and_direnv() -> None:
    readme = README.read_text(encoding="utf-8")
    docs_setup = DOCS_SETUP.read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")

    for content in (readme, docs_setup):
        assert "./scripts/bootstrap_env.sh" in content
        assert "direnv allow" in content

    assert "## Environment Activation" in agents
    assert "direnv exec . python" in agents
