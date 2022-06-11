"""Main vortrace functions.

Main entry point into vortrace functions.

Example:
    Example placeholders.

Todo:
    * Add examples.

"""

import Cvortrace
import numpy as np


class ProjectionCloud(object):
    """Object for making projections through Voronoi mesh.

    Organizes simple wrappers around the underlying Cvortrace package, which
    does all the heavy lifting.

    Example:
        Example placeholders.

    Todo:
        * Add examples.

"""

    def __init__(self, pos, dens, boundbox=None):
        pos = np.array(pos)
        dens = np.array(dens)

        if boundbox is None:
            boundbox = [
                pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(),
                pos[:, 1].max(), pos[:, 2].min(), pos[:, 2].max()
            ]

        self.boundbox = boundbox

        self._cloud = Cvortrace.PointCloud()
        self._cloud.loadPoints(pos, dens, boundbox)
        self._cloud.buildTree()

    def _make_grid(self, npix, xrng, yrng, zrng):
        pos_start = np.zeros((npix[0] * npix[1], 3))
        pos_end = np.zeros((npix[0] * npix[1], 3))

        delta_x = (xrng[1] - xrng[0]) / (npix[0] - 1)
        delta_y = (yrng[1] - yrng[0]) / (npix[1] - 1)

        for i in range(npix[0]):
            for j in range(npix[1]):
                pos_start[i * npix[0] +
                          j][0] = pos_end[i * npix[0] +
                                          j][0] = xrng[0] + delta_x * i
                pos_start[i * npix[0] +
                          j][1] = pos_end[i * npix[0] +
                                          j][1] = yrng[0] + delta_y * j
                pos_start[i * npix[0] + j][2] = zrng[0]
                pos_end[i * npix[0] + j][2] = zrng[1]

        return pos_start, pos_end

    def projection(self, xrng, yrng, npix, zrng=None):
        xrng = np.array(xrng)
        yrng = np.array(yrng)
        npix = np.array(npix)

        assert xrng.size == 2
        assert yrng.size == 2
        assert npix.size == 2

        if zrng is not None:
            zrng = np.array(zrng)
            assert zrng.size == 2
        else:
            zrng = np.array([self.boundbox[4], self.boundbox[5]])

        pos_start, pos_end = self._make_grid(npix, xrng, yrng, zrng)
        proj = Cvortrace.Projection(pos_start, pos_end)
        proj.makeProjection(self._cloud)
        dat = proj.returnProjection()

        print(dat)

        dat = np.reshape(dat, npix)
        return dat
