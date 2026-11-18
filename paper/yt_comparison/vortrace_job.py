"""Exact projections with vortrace.  Run by run.py, one process per run."""
import glob
import os
import time

import h5py
import numpy as np
import vortrace as vt

import common

pos, rho, mass = [], [], []
for part in sorted(glob.glob(str(common.DATA / "snap_500.*.hdf5"))):
    with h5py.File(part) as f:
        pos.append(f["PartType0/Coordinates"][:].astype(float))
        rho.append(f["PartType0/Density"][:].astype(float))
        mass.append(f["PartType0/Masses"][:].astype(float))
pos, rho, mass = map(np.concatenate, (pos, rho, mass))

t0 = time.perf_counter()
cloud = vt.ProjectionCloud(pos, rho, vol=mass / rho,
                           boundbox=[0, common.BOX] * 3)
build_s = time.perf_counter() - t0

images, seconds = {}, {}
for name, (x, y, width, npix) in common.REGIONS.items():
    extent = [[x - width / 2, x + width / 2], [y - width / 2, y + width / 2]]
    t0 = time.perf_counter()
    images[name] = cloud.grid_projection(extent, npix, common.ZRANGE,
                                         center=None)
    seconds[name] = time.perf_counter() - t0

threads = os.environ["OMP_NUM_THREADS"]
np.savez(common.WORK / f"vortrace_{threads}.npz", build_s=build_s,
         projection_s=seconds["main"], **images)
