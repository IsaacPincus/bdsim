"""Sphinx configuration for the bdsim documentation.

Build with:   pip install sphinx myst-parser sphinx-rtd-theme
              sphinx-build -b html docs docs/_build/html

The compiled extension is mocked (`autodoc_mock_imports`), so the documentation
builds without compiling the C++ core -- useful in CI and for a quick local read.
"""
project = "bdsim"
author = "Isaac Pincus"
copyright = "Isaac Pincus"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The pure-Python layer imports the nanobind module at import time; mock it so the
# docs build with nothing compiled.
autodoc_mock_imports = ["bdsim._bdsim", "h5py", "netCDF4", "matplotlib"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False,
                           "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = "bdsim"

import os, sys
sys.path.insert(0, os.path.abspath("../python"))

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
