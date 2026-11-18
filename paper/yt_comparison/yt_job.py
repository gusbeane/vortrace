"""SPH-kernel projections with yt.  Run by run.py, one process per run.

yt treats every cell as an SPH particle with smoothing length 2 r_cell,
its default.  The result depends on yt's kernel defaults, so use the
version the paper used: pip install yt==4.4.2
"""
import os
import time

import numpy as np
import yt
from yt.frontends.arepo.api import ArepoHDF5Dataset
from yt.geometry.particle_geometry_handler import ParticleIndex

import common

yt.set_log_level("error")

# yt reads particles in chunks of 64^3 and projects each chunk separately,
# which here means 169 calls with a large fixed cost per call and thread.
# Use one chunk per snapshot file (4 calls); yt has no option for this.
ParticleIndex.chunksize = 0


def open_snapshot():
    """Open the snapshot and its particle index.  Returns (ds, seconds)."""
    t0 = time.perf_counter()
    ds = ArepoHDF5Dataset(str(common.SNAPSHOT), smoothing_factor=2.0)
    ds.index
    return ds, time.perf_counter() - t0


# yt builds its particle index the first time it opens a snapshot and
# saves it next to the snapshot.  Delete it to time the build, then open
# the snapshot again to time a later run that finds the saved index.
index_file = common.SNAPSHOT.with_name(common.SNAPSHOT.name + ".ewah")
index_file.unlink(missing_ok=True)
ds, index_build_s = open_snapshot()
ds, index_saved_s = open_snapshot()

# Integrate through the same z slab as vortrace.
lo, hi = ds.domain_left_edge.copy(), ds.domain_right_edge.copy()
lo[2] = ds.quan(common.ZRANGE[0], "code_length")
hi[2] = ds.quan(common.ZRANGE[1], "code_length")
slab = ds.box(lo, hi)

images, seconds = {}, {}
for name, (x, y, width, npix) in common.REGIONS.items():
    t0 = time.perf_counter()
    centre = ds.arr([x, y, 90.0], "code_length")
    proj = ds.proj(("gas", "density"), "z", center=centre, data_source=slab)
    frb = proj.to_frb((width, "code_length"), npix, center=centre)
    image = np.array(frb["gas", "density"].in_units("code_mass/code_length**2"))
    images[name] = image.T                      # yt is [y, x], vortrace [x, y]
    seconds[name] = time.perf_counter() - t0

threads = os.environ["OMP_NUM_THREADS"]
np.savez(common.WORK / f"yt_{threads}.npz", build_s=index_build_s,
         saved_index_s=index_saved_s, projection_s=seconds["main"], **images)
