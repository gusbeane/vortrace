"""Run both codes on 1 and 32 threads.  Usage: python run.py

Each run is its own process under /usr/bin/time, which reports the peak
memory.  Writes results.json; plot.py makes the table and figures.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

import common

common.WORK.mkdir(parents=True, exist_ok=True)
results = []
for threads in (1, 32):
    for code in ("vortrace", "yt"):
        log = common.WORK / f"{code}_{threads}.log"
        subprocess.run(["/usr/bin/time", "-v", "-o", log, sys.executable,
                        common.HERE / f"{code}_job.py"], check=True,
                       env={**os.environ, "OMP_NUM_THREADS": str(threads)})
        peak_kb = re.search(r"Maximum resident set size \(kbytes\): (\d+)",
                            log.read_text()).group(1)
        out = np.load(common.WORK / f"{code}_{threads}.npz")
        result = {"code": code, "threads": threads,
                  "build_s": float(out["build_s"]),
                  "projection_s": float(out["projection_s"]),
                  "peak_gib": int(peak_kb) / 2**20}
        if code == "yt":
            result["saved_index_s"] = float(out["saved_index_s"])
        print(result, flush=True)
        results.append(result)

(common.HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
