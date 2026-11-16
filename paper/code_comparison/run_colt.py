"""Time the same projection with COLT.  Usage: python run_colt.py

COLT builds the Voronoi mesh from a Delaunay triangulation (CGAL) and then
traces one ray per pixel through it.
"""
import re
import time

import h5py
import numpy as np

import common
from common import HI, LO, NPIX

KPC = 3.085677581467192e21        # cm
CODE_MASS = 1e10 * 1.98841586e33  # g

work = common.workdir("colt")
pos, rho, vol = common.load_cells()
start = time.perf_counter()

# Input file: COLT wants cgs units and a temperature for every cell.
with h5py.File(work / "cells.hdf5", "w") as f:
    f.attrs["n_cells"] = np.int32(len(rho))
    for name, data, units in [
            ("r", pos * KPC, b"cm"),
            ("rho", rho * CODE_MASS / KPC**3, b"g/cm^3"),
            ("T", np.full(len(rho), 1e4), b"K"),
            # the mesh is bounded by the whole 180 kpc simulation box
            ("bbox", np.array([[0.0] * 3, [180.0 * KPC] * 3]), b"cm")]:
        f.create_dataset(name, data=data)
        f[name].attrs["units"] = units

# Parameter file: camera looking down z, image and depth equal to the box.
center = (LO + HI) / 2 * KPC
depth = (HI[2] - LO[2]) * KPC
(work / "config.yaml").write_text(f"""--- !projections
init_file: cells.hdf5
output_dir: .
output_base: proj
save_connections: false      # always rebuild the mesh
avoid_edges: false
field_weight_pairs: [[rho, avg]]
override_volume: true        # rho_avg = integral of rho along the ray / depth
adaptive: false
camera: [0, 0, 1]
camera_north: [0, 1, 0]
camera_center: [{center[0]}, {center[1]}, {center[2]}]
image_radius: {(HI[0] - LO[0]) / 2 * KPC}
n_pixels: {NPIX}
proj_depth: {depth}
""")

log = work / "log.txt"
run_s, peak_gib = common.run([common.EXTERNAL / "colt_build" / "colt",
                              "config.yaml"], log)

with h5py.File(work / "proj.hdf5", "r") as f:
    mean_density = f["proj/rho_avg"][0]             # g/cm^3, indexed [x, y]
image = mean_density * depth / (CODE_MASS / KPC**2)
total_s = time.perf_counter() - start


def colt_timer(name):
    """Seconds COLT reports for one of its timers, e.g. '... = 338.892 s'."""
    return float(re.search(name + r"\s*= \S+ = ([\d.e+-]+) s",
                           log.read_text(errors="replace")).group(1))


common.save("colt", image, ncells=len(rho),
            build_s=colt_timer("Initialization"),
            trace_s=colt_timer("Projections"),
            run_s=run_s, total_s=total_s, peak_gib=peak_gib)
