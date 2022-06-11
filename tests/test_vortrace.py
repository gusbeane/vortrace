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
        xrng = yrng = [BoxSize / 2. - L / 2., BoxSize / 2. + L / 2.]
        zrng = [0., BoxSize]
        npix = [128, 128]
        dat = pc.projection(xrng, yrng, npix, zrng=zrng)

        ref_dat = np.load('tests/test_data/galaxy_interaction-proj.npy')

        np.testing.assert_array_equal(dat, ref_dat)

        # np.save('tests/test_data/galaxy_interaction-proj.npy', dat)
