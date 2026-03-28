"""Vortrace -- fast projections through Voronoi meshes."""

import logging

from vortrace.vortrace import ProjectionCloud
from vortrace import io
from vortrace import plot
from .Cvortrace import set_verbose, get_verbose  # type: ignore

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("vortrace")
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger("vortrace").addHandler(logging.NullHandler())

__all__ = ["ProjectionCloud", "io", "plot", "set_verbose", "get_verbose",
           "__version__"]
