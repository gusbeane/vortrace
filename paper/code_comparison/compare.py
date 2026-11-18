"""Tabulate the timings and compare the three images.

Run after run_vortrace.py, run_colt.py and run_skirt.py.  Writes
timings.md and code_comparison.{pdf,png}.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

import common
from common import HI, LO

CODES = ["vortrace", "colt", "skirt"]
results = {c: json.loads((common.HERE / f"results_{c}.json").read_text())
           for c in CODES}
images = {c: np.load(common.workdir(c) / "image.npy") for c in CODES}

lines = ["| code | mesh or tree build [s] | ray tracing [s] | total [s] "
         "| peak memory [GiB] |", "|---|---:|---:|---:|---:|"]
for code, r in results.items():
    lines.append(f"| {code} | {r['build_s']:.1f} | {r['trace_s']:.1f} "
                 f"| {r['total_s']:.1f} | {r['peak_gib']:.1f} |")

lines += ["", "Relative difference from vortrace, per pixel:", "",
          "| code | 1st percentile | 99th percentile | pixels beyond 1e-6 "
          "| largest |", "|---|---:|---:|---:|---:|"]
difference = {}
for code in ["colt", "skirt"]:
    d = difference[code] = images[code] / images["vortrace"] - 1
    p1, p99 = np.percentile(d, [1, 99])
    lines.append(f"| {code} | {p1:.1e} | {p99:.1e} "
                 f"| {(abs(d) > 1e-6).sum()} | {abs(d).max():.1e} |")

ncells = results["vortrace"]["ncells"]
text = (f"# Code comparison\n\n{ncells} cells, {common.NPIX}^2 pixels, "
        f"{common.NTHREADS} threads.\n\n" + "\n".join(lines) + "\n")
(common.HERE / "timings.md").write_text(text)
print(text)

# Figure: the three images on top, the two difference maps below.
extent = [LO[0], HI[0], LO[1], HI[1]]
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True,
                         layout="constrained")
norm = LogNorm(*np.percentile(images["vortrace"], [1, 100]))
for ax, code in zip(axes[0], CODES):
    im = ax.imshow(images[code].T, origin="lower", extent=extent, norm=norm)
    ax.set_title(code)
fig.colorbar(im, ax=axes[0], label=r"column density [$10^{10}\,M_\odot$ kpc$^{-2}$]")
axes[1, 0].axis("off")
for ax, code in zip(axes[1, 1:], ["colt", "skirt"]):
    d = difference[code]
    limit = abs(d).max()
    im = ax.imshow(d.T, origin="lower", extent=extent, cmap="bwr",
                   norm=SymLogNorm(limit * 1e-3, vmin=-limit, vmax=limit))
    fig.colorbar(im, ax=ax, label=f"({code} - vortrace) / vortrace")
    ax.set_xlabel("x [kpc]")
axes[0, 0].set_ylabel("y [kpc]")
for suffix in ["pdf", "png"]:
    fig.savefig(common.HERE / f"code_comparison.{suffix}", dpi=150,
                bbox_inches="tight")
