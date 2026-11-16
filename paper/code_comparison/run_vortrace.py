"""Time the vortrace projection.  Usage: python run_vortrace.py"""
import json
import sys
import time

import numpy as np

import common

work = common.workdir("vortrace")
pos, rho, vol = common.load_cells()

start = time.perf_counter()
np.savez(work / "input.npz", pos=pos, rho=rho, vol=vol)
run_s, peak_gib = common.run(
    [sys.executable, common.HERE / "vortrace_job.py", work], work / "log.txt")
image = np.load(work / "image.npy")
total_s = time.perf_counter() - start

job = json.loads((work / "job_timings.json").read_text())
common.save("vortrace", image, ncells=len(rho), build_s=job["build_s"],
            trace_s=job["trace_s"], run_s=run_s, total_s=total_s,
            peak_gib=peak_gib)
