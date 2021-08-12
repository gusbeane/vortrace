import vortrace

cloud = vortrace.PointCloud()
snapname = "test_data/snap_200.hdf5"
boundbox = [87.0,93.0,87.0,93.0,87.0,93.0]
cloud.loadArepoSnapshot(snapname, boundbox)
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