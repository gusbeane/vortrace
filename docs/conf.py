# Configuration file for the Sphinx documentation builder.

project = "vortrace"
copyright = "2024, Angus Beane, Matthew Smith"
author = "Angus Beane, Matthew Smith"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

# Napoleon settings
napoleon_google_docstyle = True
napoleon_numpy_docstyle = True

# Autodoc settings
autodoc_member_order = "bysource"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# nbsphinx settings
nbsphinx_execute = "never"

# Theme
html_theme = "furo"

# Exclude patterns
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
