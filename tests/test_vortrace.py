"""Tests for the vortrace module.

Test cases for the main vortrace functionality.
"""
from vortrace import vortrace as vt
import h5py as h5
import numpy as np


class TestProjection:
    """Test class for projection functionality."""

    def read_arepo_snap(self, snapname):
        f = h5.File(snapname, mode='r')

        pos = np.array(f['PartType0']['Coordinates'])
        dens = np.array(f['PartType0']['Density'])
        box_size = f['Parameters'].attrs['BoxSize']

        f.close()

        return pos, dens, box_size

    def test_projection(self):
        snapname = 'tests/test_data/galaxy_interaction.hdf5'
        pos, dens, box_size = self.read_arepo_snap(snapname)

        length = 75.

        pc = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        extent = [box_size / 2. - length / 2., box_size / 2. + length / 2.]
        bounds = [0., box_size]
        npix = 128
        dat = pc.grid_projection(extent, npix, bounds, None)

        ref_dat = np.load('tests/test_data/galaxy_interaction-proj.npy')

        np.testing.assert_array_almost_equal(dat, ref_dat, decimal=14)

        pc.single_projection(np.array([0, 0, 0]),
                             np.array([box_size, box_size, box_size]))

def make_cubic_lattice(n=5):
    """Create an n x n x n cubic lattice point cloud."""
    coords = np.arange(n, dtype=np.float64)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    fields = np.ones(len(pos), dtype=np.float64)
    vol = np.ones(len(pos), dtype=np.float64)
    boundbox = [-0.5, n - 0.5, -0.5, n - 0.5, -0.5, n - 0.5]
    return pos, fields, vol, boundbox

class TestSameCell:
    def test_same_cell(self):
        """Test that rays starting and ending in the same cell are handled correctly.

        For a uniform field, the integral should equal the ray length."""

        pos, fields, vol, bb = make_cubic_lattice()
        pc = vt.ProjectionCloud(pos, fields, boundbox=bb, vol=vol)

        start = [0.5, 0.5, 0.5]
        end = [0.6, 0.6, 0.6]

        result = pc.single_projection(np.array(start), np.array(end))[0]
        expected = np.linalg.norm(np.array(end) - np.array(start))

        np.testing.assert_allclose(result, expected, rtol=1e-6)

        
