"""Sphinx configuration for DocsToKG documentation."""

from __future__ import annotations

import importlib
import os
import sys
from datetime import date

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore[assignment]

# --- paths -----------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)


def _read_version() -> str:
    pyproject = os.path.join(PROJECT_ROOT, "pyproject.toml")
    with open(pyproject, "rb") as handle:
        data = tomllib.load(handle)
    return data.get("project", {}).get("version", "0.0.0")


# --- project metadata ------------------------------------------------------
project = "DocsToKG"
author = "DocsToKG contributors"
release = _read_version()
copyright = f"{date.today():%Y}, {author}"
html_title = f"{project} {release}"

# --- extensions -------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx_sitemap",
    "sphinx_copybutton",
    "sphinx_design",
    "autoapi.extension",
]

autosummary_generate = True
autodoc_typehints = "description"
autosectionlabel_prefix_document = True
nitpicky = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# --- MyST configuration ----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

# --- theme -----------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = True
html_theme_options = {
    "show_prev_next": False,
    "navigation_depth": 3,
    "show_toc_level": 2,
}

html_baseurl = "https://paul-heyse.github.io/DocsToKG/"
sitemap_url_scheme = "{link}"

# --- intersphinx mappings --------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "httpx": ("https://www.python-httpx.org/en/stable/", None),
}

# --- mock heavy optional deps ----------------------------------------------
autodoc_mock_imports = [
    "faiss",
    "faiss_gpu",
    "torch",
    "cupy",
    "vllm",
    "rmm",
    "raft",
    "cuvs",
    "pyarrow",
]

# --- AutoAPI ---------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = [SRC_ROOT]
autoapi_root = "04-api"
autoapi_keep_files = False
autoapi_add_toctree_entry = True
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "private-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_include = ["DocsToKG.*"]
autoapi_ignore = ["*/tests/*", "*/__pycache__/*", "*/DocParsing/embedding/backends/dense/qwen_vllm.py"]
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"


# --- linkcode resolver -----------------------------------------------------
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Return a GitHub URL corresponding to the documented object."""

    if domain != "py":
        return None

    module_name = info.get("module")
    full_name = info.get("fullname")
    if not module_name:
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - import guards
        return None

    try:
        obj = module
        for part in (full_name or "").split("."):
            if not part:
                continue
            obj = getattr(obj, part)
        import inspect

        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:
        try:
            import inspect

            source_file = inspect.getsourcefile(module) or inspect.getfile(module)
            start_line = 1
        except Exception:
            return None

    relative = os.path.relpath(source_file, PROJECT_ROOT).replace("\\", "/")
    return f"https://github.com/paul-heyse/DocsToKG/blob/main/{relative}#L{start_line}"
