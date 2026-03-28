"""Shared test fixtures for the vortrace test suite."""

import h5py as h5
import numpy as np
import pytest


@pytest.fixture(scope="session")
def arepo_snap():
    """Load the galaxy_interaction snapshot once per test session.

    Returns a dict with keys: pos, dens, box_size.
    """
    snapname = "tests/test_data/galaxy_interaction.hdf5"
    with h5.File(snapname, mode="r") as f:
        pos = np.array(f["PartType0"]["Coordinates"])
        dens = np.array(f["PartType0"]["Density"])
        box_size = f["Parameters"].attrs["BoxSize"]
    return {"pos": pos, "dens": dens, "box_size": box_size}
