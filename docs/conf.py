"""Sphinx configuration for the deap_pettingzoo documentation.

The API reference is generated directly from source docstrings via
``sphinx.ext.autosummary`` (recursive) + ``sphinx.ext.autodoc``. Google-style
docstrings are understood through ``napoleon``, and the reST cross-reference
roles used throughout the codebase (``:class:``, ``:meth:``, ``:func:`` ...)
resolve natively. Hand-written guides under ``docs/`` are Markdown, parsed by
``myst_parser``.
"""

import os
import sys
from datetime import datetime

# Make the package importable by autodoc even without an editable install.
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "deap_pettingzoo"
author = "Alex"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("deap-pettingzoo")
except Exception:  # Not installed yet — fall back to the literal in __init__.
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# Autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Merge each class's __init__ docstring into the class body.
autoclass_content = "both"
# Render type hints in the description rather than the signature — keeps the
# signature line readable given the heavily-typed API.
autodoc_typehints = "description"
autodoc_preserve_defaults = True
# Streamlit executes app code at import (``st.set_page_config`` etc.), which
# floods the build log with "missing ScriptRunContext" warnings when autosummary
# imports the UI modules during recursive stub generation, so mock it.
autodoc_mock_imports = ["streamlit"]

# CI (the GitHub Pages workflow) installs only ``.[docs]`` and not the optional ML
# backends — and sets ``DOCS_MOCK_HEAVY=1``. ``neat`` is the only optional
# dependency imported at *module* level (in ``contrib/neat/*``), so it must be
# mocked for those modules to import without it.
if os.environ.get("DOCS_MOCK_HEAVY") == "1":
    autodoc_mock_imports += ["neat"]

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = True
napoleon_use_param = True
# Render ``Attributes:`` sections as ``:ivar:`` fields inside the class body
# rather than standalone attribute directives. On dataclasses (whose fields
# autodoc already introspects) the standalone form collides with autodoc's
# attribute, producing "duplicate object description" warnings.
napoleon_use_ivar = True

# Intersphinx — lets :class:`numpy.ndarray`, torch/gym refs, etc. resolve.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "pettingzoo": ("https://pettingzoo.farama.org/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
# Missing external references shouldn't fail the build.
nitpicky = False

# Suppress warnings about code blocks that can't be syntax-highlighted.
suppress_warnings = ["misc.highlighting_failure"]

# MyST (Markdown)
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline"]
myst_heading_anchors = 3
# Render ```mermaid fenced code blocks (the GitHub-flavored form used in the
# guides) as diagrams via the mermaid directive instead of plain code.
myst_fence_as_directive = ["mermaid"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # UML assets (PlantUML sources, jar, and sample result dumps) are not docs.
    "UML/**",
    "**/results/**",
]


def _untracked_markdown_docs():
    """Return docs-relative paths of git-untracked Markdown files.

    Only git-tracked Markdown docs are published. 
    Falls back to excluding nothing if git is unavailable.
    """
    import subprocess

    scaffold = {"index.md", "reference/index.md"}
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "*.md"],
            cwd=docs_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return []
    paths = (line.strip().replace("\\", "/") for line in out.splitlines())
    return [p for p in paths if p and p not in scaffold]


exclude_patterns += _untracked_markdown_docs()

# HTML output
html_theme = "furo"
html_title = f"deap_pettingzoo {version}"


# Recursive autosummary documents every module page with ``:members:``. Because
# package ``__init__`` files re-export names via ``__all__`` (e.g.
# ``from .recording import RecordingConfig``), the same object would otherwise be
# documented on both the package page and its defining-module page — producing
# "duplicate object description" warnings and ambiguous cross-references. The
# handlers below document each module-level object only on its home module.
_current_module = {"name": None}


def _track_current_module(app, what, name, obj, options, lines):
    """Remember which module page autodoc is currently rendering."""
    if what == "module":
        _current_module["name"] = name


def _skip_reexported_members(app, what, name, obj, skip, options):
    """Skip module-level members imported from elsewhere (re-exports).

    Only module members are filtered; class members are left untouched, so
    every class still documents its own methods/attributes normally.
    """
    if skip or what != "module":
        return skip
    home = getattr(obj, "__module__", None)
    current = _current_module["name"]
    if home and current and home != current:
        return True
    return skip


def setup(app):
    app.connect("autodoc-process-docstring", _track_current_module)
    app.connect("autodoc-skip-member", _skip_reexported_members)
