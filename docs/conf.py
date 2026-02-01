# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Advanced Ensemble U-MIDAS"
author = "Research Engineering"
copyright = f"{datetime.now().year}, {author}"

# The short X.Y version
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

# -- Autodoc options ---------------------------------------------------------
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Number equations (paper-style)
math_number_all = False

# -- MathJax configuration ---------------------------------------------------
# Configure MathJax 4 with no auto-numbering (use \tag{} in equations)
mathjax4_config = {
    'tex': {
        'inlinemath': [['$', '$'], ['\\(', '\\)']],
        'displaymath': [['$$', '$$'], ['\\[', '\\]']],
        'tags': 'none',  # Disable auto-numbering
        'processEscapes': True,
    },
}
