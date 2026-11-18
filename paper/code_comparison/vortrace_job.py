"""The vortrace projection itself: read cells, build tree, project, write.

Started by ``run_vortrace.py`` as its own process so that it is timed the
same way as COLT and SKIRT.  Usage: python vortrace_job.py <workdir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import vortrace as vt

from common import HI, LO, NPIX

work = Path(sys.argv[1])
cells = np.load(work / "input.npz")
pos, rho, vol = cells["pos"], cells["rho"], cells["vol"]

t0 = time.perf_counter()
cloud = vt.ProjectionCloud(pos, rho, vol=vol,
                           boundbox=[LO[0], HI[0], LO[1], HI[1], LO[2], HI[2]])
t1 = time.perf_counter()
image = cloud.grid_projection([[LO[0], HI[0]], [LO[1], HI[1]]], NPIX,
                              [LO[2], HI[2]], center=None)
t2 = time.perf_counter()

np.save(work / "image.npy", image)
(work / "job_timings.json").write_text(
    json.dumps({"build_s": t1 - t0, "trace_s": t2 - t1}))
