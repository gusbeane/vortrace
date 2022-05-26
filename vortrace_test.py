import vortrace.vortrace as vt
import arepo
import numpy as np

def read_snap(snapname):
    sn = arepo.Snapshot(snapname)
    pos = sn.part0.pos.value
    dens = sn.part0.rho.value
    BoxSize = sn.BoxSize
    return pos, dens, BoxSize

snapname = "test_data/snap_200.hdf5"
pos, dens, BoxSize = read_snap(snapname)

L = 10.0

pc = vt.projection_cloud(pos, dens, boundbox=[0., BoxSize, 0., BoxSize, 0., BoxSize])
xrng = yrng = [BoxSize/2. - L/2.0, BoxSize/2. + L/2.0]
zrng = [0., BoxSize]
npix = [512,512]
dat = pc.projection(xrng, yrng, npix, zrng=zrng)
np.save('test_proj.npy', dat)

# boundbox = [0., BoxSize, 0., BoxSize, 0., BoxSize]

print(BoxSize)

# cloud.loadPoints(pos, dens, boundbox)
# cloud.buildTree()

# proj = Cvortrace.Projection(npix,boundbox)
# proj.makeProjection(cloud)
# dat = proj.returnProjection()
# proj.saveProjection('test_proj.dat')




# depth = 90.0
# slc_boundbox = [87.0,93.0,87.0,93.0]
# slc = Cvortrace.Slice(npix,slc_boundbox,depth)
# slc.makeSlice(cloud)
# slc.saveSlice('test_slice.dat')

