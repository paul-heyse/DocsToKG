"""
Sphinx Documentation Configuration

This configuration drives the DocsToKG technical documentation site,
enabling autodoc, autosummary, and Google-style docstrings so our
annotation standards surface consistently in HTML builds.

Key Features:
- Enables Napoleon, Intersphinx, and coverage reporting extensions
- Configures Read the Docs theme with custom styling hooks
- Mocks heavyweight dependencies (Faiss, Torch) for offline builds
"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DocsToKG"
copyright = "2025, DocsToKG Team"
author = "DocsToKG Team"

# The short X.Y version
version = "1.0.0"

# The full version, including alpha/beta/rc tags
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core autodoc functionality
    "sphinx.ext.autosummary",  # Generate autosummary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for NumPy/Google style docstrings
    "sphinx.ext.intersphinx",  # Link to external documentation
    "sphinx.ext.todo",  # Support for todo items
    "sphinx.ext.coverage",  # Collect doc coverage stats
    "sphinx.ext.githubpages",  # Publish to GitHub Pages
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": None,
    ".md": None,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Custom CSS
html_css_files = [
    "css/custom.css",
]

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-autodoc

# This value selects if automatically documented members are sorted
# alphabetical (value 'alphabetical'), by member type (value 'groupwise')
# or by source order (value 'bysource').
autodoc_member_order = "bysource"

# Default flags for autodoc directives
autodoc_default_flags = ["members"]

# Mock imports for external dependencies that may not be installed
autodoc_mock_imports = [
    "faiss",
    "numpy",
    "torch",
    "transformers",
    "sentence_transformers",
]

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True

# -- Options for napoleon (Google/NumPy style docstrings) -------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "faiss": ("https://faiss.ai/cpp_api/", None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for coverage ---------------------------------------------------
coverage_show_missing = True

# -- Custom configuration ---------------------------------------------------
# Add any custom configuration here

# Custom object types for cross-referencing
rst_epilog = """
.. _DocsToKG: https://github.com/paul-heyse/DocsToKG
.. _Faiss: https://faiss.ai/
.. _OpenSpec: https://openspec.dev/
"""

# Suppress certain warnings
suppress_warnings = [
    "image.nonlocal_uri",  # Suppress warnings for external images
]
