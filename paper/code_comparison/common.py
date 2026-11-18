"""Shared setup for the vortrace / COLT / SKIRT comparison.

All three codes project the same gas cells: those of the dwarf-galaxy
snapshot that lie inside the box LO..HI.  The image is the x-y face of that
box, NPIX pixels on a side, with one ray per pixel centre running along z
through the full depth of the box.

Units are the snapshot's: kpc, 1e10 Msun.  Images are column densities in
1e10 Msun / kpc^2, indexed [x, y].
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parents[1] / "data" / "external"   # COLT and SKIRT live here
SNAPSHOT_FILES = sorted((HERE.parents[1] / "data").glob("snap_500.*.hdf5"))

LO = np.array([87.0, 87.0, 88.0])    # kpc, corner of the box
HI = np.array([93.0, 93.0, 92.0])    # kpc, opposite corner
NPIX = 1024
NTHREADS = 32


def load_cells():
    """Positions [kpc], densities and volumes of the cells inside the box."""
    pos, rho, mass = [], [], []
    for name in SNAPSHOT_FILES:
        with h5py.File(name, "r") as f:
            pos.append(f["PartType0/Coordinates"][:].astype(float))
            rho.append(f["PartType0/Density"][:].astype(float))
            mass.append(f["PartType0/Masses"][:].astype(float))
    pos, rho, mass = map(np.concatenate, (pos, rho, mass))

    # The snapshot stores positions in single precision, so a few cells have
    # exactly the same position.  A Voronoi mesh needs distinct generators,
    # so nudge the second cell of each such pair by about 1e-6 kpc.
    _, first, count = np.unique(pos, axis=0, return_index=True,
                                return_counts=True)
    for i in first[count > 1]:
        same = np.where(np.all(pos == pos[i], axis=1))[0]
        for k, j in enumerate(same[1:], start=1):
            pos[j] += 1e-6 * k * np.array([1.0, 0.5, 0.25]) / 1.146

    inside = np.all((pos >= LO) & (pos < HI), axis=1)
    return pos[inside], rho[inside], (mass / rho)[inside]


def workdir(code):
    """Scratch directory for one code's input, log and output files."""
    path = EXTERNAL / "runs" / code
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(command, log):
    """Run a command with NTHREADS threads, writing its output to ``log``.

    Returns the wall-clock time [s] and the peak memory [GiB] of the
    process, the latter as reported by ``/usr/bin/time -v``.
    """
    env = dict(os.environ, OMP_NUM_THREADS=str(NTHREADS))
    start = time.perf_counter()
    with open(log, "w") as f:
        subprocess.run(["/usr/bin/time", "-v", *map(str, command)],
                       cwd=log.parent, env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    seconds = time.perf_counter() - start
    peak_kb = re.search(r"Maximum resident set size \(kbytes\): (\d+)",
                        log.read_text(errors="replace")).group(1)
    return seconds, int(peak_kb) / 2**20


def save(code, image, **numbers):
    """Store a code's image and its timings for ``compare.py``."""
    np.save(workdir(code) / "image.npy", image)
    (HERE / f"results_{code}.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))
