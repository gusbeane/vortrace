"""Make the timings table and the two paper figures.  Usage: python plot.py

Reads results.json and the images saved by run.py.  Writes timings.md,
yt_comparison.pdf and error_histogram.pdf.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

import common

exact = np.load(common.WORK / "vortrace_1.npz")
sph = np.load(common.WORK / "yt_1.npz")
error = {name: sph[name] / exact[name] - 1 for name in common.REGIONS}

# ---- timings.md ----------------------------------------------------------
lines = ["# yt comparison", "",
         "| code | threads | index or tree build [s] | projection [s] "
         "| total [s] | peak memory [GiB] |", "|---|---:|---:|---:|---:|---:|"]
for r in json.loads((common.HERE / "results.json").read_text()):
    lines.append(f"| {r['code']} | {r['threads']} | {r['build_s']:.1f} "
                 f"| {r['projection_s']:.1f} "
                 f"| {r['build_s'] + r['projection_s']:.1f} "
                 f"| {r['peak_gib']:.1f} |")
    if r["code"] == "yt":
        saved = r["saved_index_s"]
lines += ["", "yt saves its index next to the snapshot.  Later runs load it "
          f"in {saved:.1f} s instead of building it.", "",
          "Relative error of yt, (yt - vortrace) / vortrace, per pixel of "
          "the main image:", "", "| percentile | 1 | 5 | 50 | 95 | 99 |",
          "|---|---:|---:|---:|---:|---:|",
          "| error | " + " | ".join(f"{p:+.3f}" for p in np.percentile(
              error["main"], [1, 5, 50, 95, 99])) + " |"]
(common.HERE / "timings.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- yt_comparison.pdf ---------------------------------------------------
# Three panels: vortrace, yt, relative error.  Each has two insets, which
# are separate projections of a small region and not zooms of the image.


def extent(name):
    """Image edges in kpc from the centre of the main image."""
    x, y, width, _ = common.REGIONS[name]
    return [x - 90 - width / 2, x - 90 + width / 2,
            y - 90 - width / 2, y - 90 + width / 2]


def density_norm(name, top=1.0):
    """Log colour scale for one region, shared by vortrace and yt."""
    return LogNorm(np.percentile(exact[name], 1),
                   top * max(exact[name].max(), sph[name].max()))


error_norm = Normalize(-0.25, 0.25)
inset_boxes = {"dense": [0.60, 0.02, 0.38, 0.38],     # axes fractions
               "void": [0.60, 0.60, 0.38, 0.38]}

fig = plt.figure(figsize=(17, 5.2))
grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.05, 0.22, 1, 0.05],
                        wspace=0.08, left=0.05, right=0.95, top=0.92,
                        bottom=0.12)
panels = [(grid[0, 0], exact, "vortrace (exact)"),
          (grid[0, 1], sph, r"yt (SPH kernel, $h = 2\,r_\mathrm{cell}$)"),
          (grid[0, 4], error, "(yt - vortrace) / vortrace")]
for cell, images, title in panels:
    is_error = images is error
    cmap, edge = ("bwr", "black") if is_error else ("viridis", "white")
    ax = fig.add_subplot(cell)
    # The main scale tops out above the brightest pixel so knots don't saturate.
    main = ax.imshow(images["main"].T, origin="lower", extent=extent("main"),
                     norm=error_norm if is_error else density_norm("main", 3),
                     cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x [kpc]")
    if images is exact:
        ax.set_ylabel("y [kpc]")
    else:
        ax.tick_params(labelleft=False)

    for name, box in inset_boxes.items():
        norm = error_norm if is_error else density_norm(name)
        inset = ax.inset_axes(box, xticks=[], yticks=[])
        image = inset.imshow(images[name].T, origin="lower",
                             extent=extent(name), norm=norm, cmap=cmap,
                             interpolation="nearest")
        for spine in inset.spines.values():
            spine.set(edgecolor=edge, linewidth=1.2)
        ax.indicate_inset_zoom(inset, edgecolor=edge, linewidth=1.2)
        if is_error:
            continue
        # Density insets have their own scale: a small colorbar beside each.
        above = name == "dense"
        bar_y = box[1] + box[3] + 0.02 if above else box[1] - 0.045
        bar = fig.colorbar(image, orientation="horizontal",
                           cax=ax.inset_axes([box[0], bar_y, box[2], 0.025]))
        bar.minorticks_off()
        bar.set_ticks([norm.vmin, norm.vmax],
                      labels=[f"{norm.vmin:.1e}", f"{norm.vmax:.1e}"])
        bar.ax.xaxis.set_ticks_position("top" if above else "bottom")
        bar.ax.tick_params(labelsize=7, color="white", labelcolor="white",
                           pad=1)
        for label, side in zip(bar.ax.get_xticklabels(), ("left", "right")):
            label.set_horizontalalignment(side)
        bar.outline.set_edgecolor("white")

    if images is sph:
        fig.colorbar(main, cax=fig.add_subplot(grid[0, 2]), label=(
            r"Column density [$10^{10}\;M_\odot\,\mathrm{kpc}^{-2}$]"))
    if is_error:
        fig.colorbar(main, cax=fig.add_subplot(grid[0, 5]), extend="both",
                     label="Relative error")
fig.savefig(common.HERE / "yt_comparison.pdf", dpi=200, bbox_inches="tight")

# ---- error_histogram.pdf -------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2), layout="tight")
values = error["main"].ravel()
ax.hist(values, bins=np.linspace(-1, 2, 121), histtype="step", color="k",
        lw=1.5, weights=np.full(values.size, 1 / values.size))
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set(yscale="log", xlim=(-1, 2), xlabel="(yt - vortrace) / vortrace",
       ylabel="fraction of pixels",
       title=r"yt (SPH kernel, $h = 2\,r_\mathrm{cell}$)")
fig.savefig(common.HERE / "error_histogram.pdf", dpi=200, bbox_inches="tight")
