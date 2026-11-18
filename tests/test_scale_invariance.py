"""Projections must not depend on the absolute scale of the coordinates.

Regression test for a bug found while comparing vortrace with COLT and
SKIRT on a high-resolution galaxy: the ray walk used an *absolute*
tolerance (1e-9 in squared distance) to decide whether two generators are
equidistant from a point on the ray.  For meshes whose cells are much
smaller than the coordinate unit (pc-sized cells in a kpc box, r^2 ~ 1e-6)
that tolerance is a sizeable fraction of a cell, so a third cell lying
between two cells could be mistaken for a tie and skipped, giving errors
of up to a few per cent in ~0.2% of the pixels.  The tolerance is now
relative to the local squared distance, which makes the walk scale
invariant.
"""
import h5py as h5
import numpy as np
import pytest

from vortrace import vortrace as vt

SNAP = "tests/test_data/galaxy_interaction.hdf5"


@pytest.fixture(scope="module")
def galaxy():
    with h5.File(SNAP, "r") as f:
        pos = np.array(f["PartType0/Coordinates"], dtype=float)
        rho = np.array(f["PartType0/Density"], dtype=float)
        mass = np.array(f["PartType0/Masses"], dtype=float)
        box = float(f["Parameters"].attrs["BoxSize"])
    return pos, rho, mass / rho, box


def _project(pos, rho, vol, box, scale, npix=96):
    """Column density of the central 75 kpc (in original units), with all
    lengths multiplied by ``scale``.  Densities are left unchanged, so the
    column density scales exactly by ``scale``."""
    pc = vt.ProjectionCloud(pos * scale, rho, vol=vol * scale**3,
                            boundbox=[0, box * scale] * 3)
    half = 37.5 * scale
    c = box / 2 * scale
    extent = [[c - half, c + half], [c - half, c + half]]
    image = pc.grid_projection(extent, npix, [0.0, box * scale], center=None)
    return image / scale


@pytest.mark.parametrize("scale", [1e-3, 1e-5, 1e3])
def test_grid_projection_is_scale_invariant(galaxy, scale):
    pos, rho, vol, box = galaxy
    ref = _project(pos, rho, vol, box, 1.0)
    img = _project(pos, rho, vol, box, scale)
    np.testing.assert_allclose(img, ref, rtol=1e-9)
