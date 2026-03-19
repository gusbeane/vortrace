"""Vortrace -- fast projections through Voronoi meshes."""

from vortrace.vortrace import ProjectionCloud
from vortrace import io
from vortrace import plot

try:
    from .Cvortrace import set_verbose, get_verbose  # type: ignore
except ModuleNotFoundError:
    from importlib import import_module as _im
    _c = _im("Cvortrace")
    set_verbose = _c.set_verbose  # type: ignore
    get_verbose = _c.get_verbose  # type: ignore

__all__ = ["ProjectionCloud", "io", "plot", "set_verbose", "get_verbose"]
