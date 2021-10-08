import vortrace
import arepo
import numpy as np

def read_snap(snapname):
    sn = arepo.Snapshot(snapname)
    pos = sn.part0.pos.value
    dens = sn.part0.rho.value
    return pos, dens

cloud = vortrace.PointCloud()
snapname = "test_data/snap_200.hdf5"
boundbox = [87.0,93.0,87.0,93.0,87.0,93.0]
pos, dens = read_snap(snapname)
cloud.loadPoints(pos, dens, boundbox)
cloud.buildTree()

npix = [256,256]
proj = vortrace.Projection(npix,boundbox)
proj.makeProjection(cloud)
proj.saveProjection('test_proj.dat')

depth = 90.0
slc_boundbox = [87.0,93.0,87.0,93.0]
slc = vortrace.Slice(npix,slc_boundbox,depth)
slc.makeSlice(cloud)
slc.saveSlice('test_slice.dat')