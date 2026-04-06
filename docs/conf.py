# Configuration file for the Sphinx documentation builder.

project = "vortrace"
copyright = "2024, Angus Beane, Matthew Smith"
author = "Angus Beane, Matthew Smith"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
]

# Napoleon settings
napoleon_google_docstyle = True
napoleon_numpy_docstyle = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "none"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Theme
html_theme = "furo"

# Exclude patterns
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# ---------------------------------------------------------------------------
# Auto-generate tutorial images before the build starts
# ---------------------------------------------------------------------------
import subprocess
import sys
from pathlib import Path


def _generate_images(app):
    """Run generate_images.py if any output image is missing."""
    docs_dir = Path(__file__).resolve().parent
    image_dir = docs_dir / "images"
    expected = [
        "quickstart.png", "grid_projection.png",
        "multifield.png", "volume_rendering.png", "arbitrary_rays.png",
        "single_ray.png", "periodic.png", "io_loaded.png",
    ]
    if all((image_dir / name).exists() for name in expected):
        return
    print("Generating tutorial images...")
    subprocess.check_call(
        [sys.executable, str(docs_dir / "generate_images.py")],
    )


def setup(app):
    app.connect("builder-inited", _generate_images)
