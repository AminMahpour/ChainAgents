"""Sphinx configuration for the ChainAgents documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "ChainAgents"
copyright = "Amin Mahpour"
author = "Amin Mahpour"
release = "1.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "exclude-imported-members": True,
    "show-inheritance": True,
}
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Docstring prose was not written to be published as API docs; don't fail on
# stray RST metacharacters inside it, re-exported-ambiguous type cross
# references, or autosectionlabel label collisions from signature lines.
suppress_warnings = ["docutils", "ref.python", "autosectionlabel"]
python_use_unqualified_type_names = True

exclude_patterns = [
    "_build",
    "superpowers",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = "ChainAgents Documentation"


def _guard_napoleon(handler):
    """Let napoleon's member check fail soft instead of killing the build.

    Pydantic's mock validator/serializer placeholders raise on any attribute
    access (including ``__qualname__``, which napoleon reads unguarded), and
    Sphinx's ``emit()`` invokes every registered handler regardless of earlier
    results, so the guard has to wrap napoleon's own listener.
    """

    def guarded(*args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except Exception:
            return None

    guarded.__name__ = getattr(handler, "__name__", "guarded")
    return guarded


def setup(app):
    listeners = app.events.listeners["autodoc-skip-member"]
    for i, listener in enumerate(listeners):
        if getattr(listener.handler, "__name__", "") == "_skip_member":
            listeners[i] = listener._replace(handler=_guard_napoleon(listener.handler))
