import Cvortrace
import numpy as np

class projection_cloud(object):
    def __init__(self, pos, dens, boundbox=None):
        pos = np.array(pos)
        dens = np.array(dens)

        if boundbox is None:
            boundbox = [pos[:,0].min(), pos[:,0].max(),
                        pos[:,1].min(), pos[:,1].max(),
                        pos[:,2].min(), pos[:,2].max()]

        self.boundbox = boundbox

        self.cloud = Cvortrace.PointCloud()
        self.cloud.loadPoints(pos, dens, boundbox)
        self.cloud.buildTree()
    
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
        
        print(0)

        extent = [xrng[0], xrng[1], yrng[0], yrng[1], zrng[0], zrng[1]]
        proj = Cvortrace.Projection(npix, extent)
        print(1)
        proj.makeProjection(self.cloud)
        print(2)
        dat = proj.returnProjection()
        print(3)

        dat = np.reshape(dat, npix)
        return dat


