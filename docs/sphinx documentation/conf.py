"""Sphinx configuration for DocsToKG.

This configuration is intentionally lightweight; it targets HTML output in
``docs/sphinx documentation/_build`` and enables Google-style docstring parsing
via napoleon so our codebase documentation mirrors the repository standards.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

project = "DocsToKG"
author = "DocsToKG contributors"
copyright = (
    f"{datetime.now():%Y}, {author}"  # noqa: A001 - Sphinx expects a variable named 'copyright'
)
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "pydantic": ("https://docs.pydantic.dev/latest/", {}),
}

nitpicky = False
default_role = "any"
