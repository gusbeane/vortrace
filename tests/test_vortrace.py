from vortrace import vortrace as vt
import h5py as h5
import numpy as np


class TestProjection:

    def read_arepo_snap(self, snapname):
        f = h5.File(snapname, mode='r')

        pos = np.array(f['PartType0']['Coordinates'])
        dens = np.array(f['PartType0']['Density'])
        BoxSize = f['Parameters'].attrs['BoxSize']

        f.close()

        return pos, dens, BoxSize

    def test_projection(self):
        snapname = 'tests/test_data/galaxy_interaction.hdf5'
        pos, dens, BoxSize = self.read_arepo_snap(snapname)

        L = 75.

        pc = vt.ProjectionCloud(
            pos, dens, boundbox=[0., BoxSize, 0., BoxSize, 0., BoxSize])
        extent= [BoxSize / 2. - L / 2., BoxSize / 2. + L / 2.]
        bounds = [0., BoxSize]
        npix = 128
        center = [BoxSize/2., BoxSize/2., BoxSize/2.]
        dat = pc.grid_projection(extent, npix, bounds, None)

        ref_dat = np.load('tests/test_data/galaxy_interaction-proj.npy')

        np.testing.assert_array_almost_equal(dat, ref_dat, decimal=14)
