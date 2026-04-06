"""Tests for ProjectionCloud.slice() method."""

import numpy as np
import pytest
from vortrace.vortrace import ProjectionCloud


def make_cubic_lattice(n=5):
    """Create an n x n x n cubic lattice point cloud."""
    coords = np.arange(n, dtype=np.float64)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    fields = np.ones(len(pos), dtype=np.float64)
    vol = np.ones(len(pos), dtype=np.float64)
    boundbox = [-0.5, n - 0.5, -0.5, n - 0.5, -0.5, n - 0.5]
    return pos, fields, vol, boundbox


class TestSlice:

    def test_shape_single_field(self):
        pos, fields, vol, bb = make_cubic_lattice(n=5)
        pc = ProjectionCloud(pos, fields, boundbox=bb, vol=vol)
        data = pc.slice([-0.4, 4.4, -0.4, 4.4], 10, depth=2.0)
        assert data.shape == (10, 10)

    def test_shape_multi_field(self):
        pos, fields, vol, bb = make_cubic_lattice(n=5)
        fields_2d = np.column_stack([fields, 2.0 * fields])
        pc = ProjectionCloud(pos, fields_2d, boundbox=bb, vol=vol)
        data = pc.slice([-0.4, 4.4, -0.4, 4.4], 10, depth=2.0)
        assert data.shape == (10, 10, 2)

    def test_uniform_field(self):
        pos, fields, vol, bb = make_cubic_lattice(n=5)
        pc = ProjectionCloud(pos, fields, boundbox=bb, vol=vol)
        data = pc.slice([-0.4, 4.4, -0.4, 4.4], 10, depth=2.0)
        np.testing.assert_allclose(data, 1.0)

    def test_rectangular_nres(self):
        pos, fields, vol, bb = make_cubic_lattice(n=5)
        pc = ProjectionCloud(pos, fields, boundbox=bb, vol=vol)
        data = pc.slice([-0.4, 4.4, -0.4, 4.4], (8, 16), depth=2.0)
        assert data.shape == (8, 16)

    def test_bad_extent_raises(self):
        pos, fields, vol, bb = make_cubic_lattice(n=5)
        pc = ProjectionCloud(pos, fields, boundbox=bb, vol=vol)
        with pytest.raises(ValueError, match="extent must have 4 elements"):
            pc.slice([0, 1], 10, depth=2.0)

    def test_arepo_data(self, arepo_snap):
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        pc = ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        data = pc.slice([10., 90., 10., 90.], 64, depth=box_size / 2.)
        assert data.shape == (64, 64)
        assert np.all(data > 0)
