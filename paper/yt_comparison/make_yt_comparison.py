#!/usr/bin/env python
"""Compare an exact vortrace projection with yt's SPH-kernel treatment.

``yt`` (and other SPH-oriented tools) project a Voronoi mesh by treating
every cell as an SPH particle whose smoothing length is a fixed multiple of
the cell radius.  This script projects an Arepo snapshot both ways, plots
the two column-density maps side by side with a map of the relative error
and a zoom on the densest region, plots a histogram of the relative error,
and records timings and summary statistics.  The SPH smoothing length
h = f r_cell is a free parameter of the kernel treatment; yt's default
f = 2 is used (``--hsml-factor``).

The snapshot used for the paper figures is an isolated-galaxy simulation
that is too large to ship with the repository (``data/snap_500.*.hdf5``,
about 10 GB in four parts).  Multi-part snapshots are read by pointing
``--snapshot`` at the first part.  The projection is restricted to a slab
in ``z`` (``--zrange``) and a square window (``--center``, ``--width``).

This script is deliberately *not* part of the ``vortrace`` package and
``yt`` is *not* a dependency of ``vortrace``.  The figure depends on yt's
kernel and normalisation defaults, so the yt version is pinned::

    pip install vortrace h5py matplotlib yt==4.4.2
    python paper/yt_comparison/make_yt_comparison.py \\
        --snapshot data/snap_500.0.hdf5

Outputs (written next to this script by default):

- ``yt_comparison.pdf``   -- vector figure used in the paper
- ``error_histogram.pdf`` -- histogram of the per-pixel relative error
- ``stats.json``          -- timings, settings and summary statistics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

import vortrace as vt

matplotlib.use("Agg")

YT_VERSION = "4.4.2"

SCRIPT_DIR = Path(__file__).resolve().parent            # paper/yt_comparison/
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SNAP = REPO_ROOT / "data" / "snap_500.0.hdf5"

LENGTH_UNIT = "kpc"
COL_DENSITY_UNIT = r"$10^{10}\;M_\odot\,\mathrm{kpc}^{-2}$"


def _import_yt():
    """Import yt lazily and enforce the version pin."""
    try:
        import yt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "This script needs yt, which is not a vortrace dependency. "
            f"Install it with: pip install yt=={YT_VERSION}"
        ) from exc
    if yt.__version__ != YT_VERSION:
        raise RuntimeError(
            f"yt {yt.__version__} is installed but this script is pinned to "
            f"yt=={YT_VERSION} because the figure depends on yt's kernel and "
            "normalisation defaults. Install it with: "
            f"pip install yt=={YT_VERSION}"
        )
    yt.set_log_level("error")
    return yt


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def snapshot_parts(path):
    """All parts of a (possibly multi-file) Arepo snapshot, sorted."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The paper snapshot is not part of the "
            "repository; pass --snapshot pointing at an Arepo HDF5 snapshot "
            "(the first part of a multi-part snapshot).")
    with h5py.File(path, "r") as f:
        nfiles = int(f["Header"].attrs.get("NumFilesPerSnapshot", 1))
    if nfiles == 1:
        return [path]
    stem = path.name.rsplit(".", 2)[0]          # snap_500.0.hdf5 -> snap_500
    parts = sorted(glob.glob(str(path.parent / f"{stem}.*.hdf5")),
                   key=lambda s: int(s.rsplit(".", 2)[1]))
    if len(parts) != nfiles:
        raise FileNotFoundError(
            f"Expected {nfiles} parts of {stem} next to {path}, "
            f"found {len(parts)}")
    return [Path(s) for s in parts]


def load_snapshot(path):
    """Load gas positions, density, cell volume and box size."""
    pos, rho, mass = [], [], []
    for part in snapshot_parts(path):
        with h5py.File(part, "r") as f:
            pos.append(np.asarray(f["PartType0/Coordinates"], dtype=float))
            rho.append(np.asarray(f["PartType0/Density"], dtype=float))
            mass.append(np.asarray(f["PartType0/Masses"], dtype=float))
    with h5py.File(path, "r") as f:
        box_size = float(f["Header"].attrs["BoxSize"])
    pos, rho, mass = map(np.concatenate, (pos, rho, mass))
    return pos, rho, mass / rho, box_size


def build_cloud(pos, rho, vol, box_size):
    """Build the vortrace point cloud (kD-tree). Returns (cloud, seconds)."""
    t0 = time.perf_counter()
    pc = vt.ProjectionCloud(
        pos, rho, vol=vol,
        boundbox=[0, box_size, 0, box_size, 0, box_size],
    )
    return pc, time.perf_counter() - t0


def project_vortrace(pc, *, center, width, zrange, npix):
    """Exact column density with vortrace. Returns ([x, y] image, seconds)."""
    extent = [[center[0] - width / 2, center[0] + width / 2],
              [center[1] - width / 2, center[1] + width / 2]]
    t0 = time.perf_counter()
    image = pc.grid_projection(extent, npix, list(zrange), center=None)
    return image, time.perf_counter() - t0


def project_yt(snap, *, center, width, zrange, npix, smoothing_factor):
    """SPH-kernel column density with yt.

    Returns ([x, y] image, timings, settings).
    """
    # Ask for the Arepo frontend explicitly (yt's auto-detection falls back
    # to the plain Gadget frontend for snapshots whose Config group lacks
    # the VORONOI flag, e.g. the trimmed test snapshot).  The Arepo frontend
    # gives each cell a smoothing length smoothing_factor * (3V / 4pi)^(1/3).
    # pylint: disable-next=import-outside-toplevel
    from yt.frontends.arepo.api import ArepoHDF5Dataset

    t0 = time.perf_counter()
    ds = ArepoHDF5Dataset(str(snap), smoothing_factor=smoothing_factor)
    # Force index construction so it is counted in "load".
    ds.index  # pylint: disable=pointless-statement
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    zmid = 0.5 * (zrange[0] + zrange[1])
    c = ds.arr([center[0], center[1], zmid], "code_length")
    # Restrict the integration to the z slab the same way vortrace does.
    le = ds.domain_left_edge.copy()
    re = ds.domain_right_edge.copy()
    le[2] = ds.quan(zrange[0], "code_length")
    re[2] = ds.quan(zrange[1], "code_length")
    prj = ds.proj(("gas", "density"), "z", center=c,
                  data_source=ds.box(le, re))
    frb = prj.to_frb((width, "code_length"), npix, center=c)
    image_yx = np.array(
        frb[("gas", "density")].in_units("code_mass/code_length**2"))
    t_proj = time.perf_counter() - t0

    settings = {
        "dataset_class": type(ds).__name__,
        "smoothing_factor": float(ds.smoothing_factor),
        "kernel_name": ds.kernel_name,
        "sph_smoothing_style": ds.sph_smoothing_style,
        "use_sph_normalization": bool(ds.use_sph_normalization),
        "field": "('gas', 'density') projected along z via "
                 "ds.proj(..., data_source=ds.box(z slab)).to_frb(...)",
    }
    # yt's FRB is indexed [y, x]; vortrace grids are [x, y].
    return image_yx.T, {"load_s": t_load, "projection_s": t_proj}, settings


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _window(cx, cy, half, ext):
    """Square window of half-width ``half`` around (cx, cy), kept inside ext."""
    cx = float(np.clip(cx, ext[0] + half, ext[1] - half))
    cy = float(np.clip(cy, ext[2] + half, ext[3] - half))
    return [cx - half, cx + half, cy - half, cy + half]


def make_figure(run, width, *, vmax_factor, err_lim, out_paths):
    """Two column-density maps plus a relative-error map, with zoom insets.

    The insets are not zooms of the main image: each inset region is
    projected separately, with both methods, at its own (finer) pixel
    size, so that the cell-scale structure is resolved.

    The figure is written to every path in ``out_paths`` (the format is
    taken from the extension).
    """
    img_vt, img_yt = run["img_vt"], run["img_yt"]
    ext = [-width / 2, width / 2, -width / 2, width / 2]

    # Shared colour scale.  The ceiling sits above the brightest pixel of
    # either image so that the dense knots do not saturate.
    positive = img_vt[img_vt > 0]
    vmin = np.percentile(positive, 1)
    vmax = vmax_factor * max(img_vt.max(), img_yt.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    err_norm = Normalize(vmin=-err_lim, vmax=err_lim)

    def rel_err(exact, kernel):
        return (kernel - exact) / exact

    # (inset name, position in axes fraction, side for its colorbar)
    inset_layout = [("dense", [0.60, 0.02, 0.38, 0.38], "above"),
                    ("void", [0.60, 0.60, 0.38, 0.38], "below")]

    # The column-density insets get their own log scale spanning the inset
    # region itself (the void is dark and the knot bright on the shared
    # scale), with a small colorbar beside each.
    inset_norms = {}
    for name, _, _ in inset_layout:
        ins = run["insets"][name]
        positive = ins["img_vt"][ins["img_vt"] > 0]
        inset_norms[name] = LogNorm(
            vmin=np.percentile(positive, 1),
            vmax=max(ins["img_vt"].max(), ins["img_yt"].max()))

    # Layout: image | image | colorbar | gap | image | colorbar
    # The gap column leaves room for the density colorbar's label.
    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.05, 0.22, 1, 0.05],
                          wspace=0.08, left=0.05, right=0.95,
                          top=0.92, bottom=0.12)
    ax_vt = fig.add_subplot(gs[0, 0])
    ax_yt = fig.add_subplot(gs[0, 1], sharey=ax_vt)
    cax_dens = fig.add_subplot(gs[0, 2])
    ax_err = fig.add_subplot(gs[0, 4], sharey=ax_vt)
    cax_err = fig.add_subplot(gs[0, 5])

    # (axes, main image, per-inset image, title, norm, cmap, frame colour)
    panels = [
        (ax_vt, img_vt, lambda ins: ins["img_vt"],
         "vortrace (exact)", norm, "viridis", "white"),
        (ax_yt, img_yt, lambda ins: ins["img_yt"],
         r"yt (SPH kernel, $h = %g\,r_\mathrm{cell}$)" % run["factor"],
         norm, "viridis", "white"),
        (ax_err, rel_err(img_vt, img_yt),
         lambda ins: rel_err(ins["img_vt"], ins["img_yt"]),
         "(yt - vortrace) / vortrace", err_norm, "bwr", "black"),
    ]
    images = []
    for ax, img, inset_img, title, nrm, cmap, edge in panels:
        images.append(ax.imshow(img.T, origin="lower", extent=ext, norm=nrm,
                                cmap=cmap, interpolation="nearest"))
        ax.set_title(title)
        ax.set_xlabel(f"x [{LENGTH_UNIT}]")

        for name, position, side in inset_layout:
            ins = run["insets"][name]
            window = ins["window"]
            own_scale = nrm is norm
            ins_norm = inset_norms[name] if own_scale else nrm
            axins = ax.inset_axes(position)
            im_ins = axins.imshow(inset_img(ins).T, origin="lower",
                                  extent=window, norm=ins_norm,
                                  cmap=cmap, interpolation="nearest")
            if own_scale:
                if side == "below":
                    cb_y = position[1] - 0.045
                else:
                    cb_y = position[1] + position[3] + 0.02
                cax_ins = ax.inset_axes([position[0], cb_y,
                                         position[2], 0.025])
                cb = fig.colorbar(im_ins, cax=cax_ins,
                                  orientation="horizontal")
                cb.minorticks_off()
                cb.set_ticks([ins_norm.vmin, ins_norm.vmax])
                cb.set_ticklabels([f"{ins_norm.vmin:.1e}",
                                   f"{ins_norm.vmax:.1e}"])
                if side == "above":
                    cb.ax.xaxis.set_ticks_position("top")
                cb.ax.tick_params(labelsize=7, color="white",
                                  labelcolor="white", pad=1)
                for lab, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
                    lab.set_horizontalalignment(ha)
                cb.outline.set_edgecolor("white")
            axins.set_xlim(window[0], window[1])
            axins.set_ylim(window[2], window[3])
            axins.set_xticks([])
            axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(1.2)
            ax.indicate_inset_zoom(axins, edgecolor=edge, linewidth=1.2)

    ax_vt.set_ylabel(f"y [{LENGTH_UNIT}]")
    for ax in (ax_yt, ax_err):
        ax.tick_params(labelleft=False)
    fig.colorbar(images[0], cax=cax_dens,
                 label=f"Column density [{COL_DENSITY_UNIT}]")
    fig.colorbar(images[2], cax=cax_err, extend="both",
                 label="Relative error")
    for out_path in out_paths:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_histogram(img_vt, img_yt, *, smoothing_factor, out_paths):
    """Histogram of the per-pixel relative error (yt - vortrace) / vortrace
    over every pixel of the main image."""
    rel_err = ((img_yt - img_vt) / img_vt).ravel()
    bins = np.linspace(-1.0, 2.0, 121)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(rel_err, bins=bins, histtype="step", color="k", lw=1.5,
            weights=np.full(rel_err.size, 1.0 / rel_err.size))
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("(yt - vortrace) / vortrace")
    ax.set_ylabel("fraction of pixels")
    ax.set_title(r"yt (SPH kernel, $h = %g\,r_\mathrm{cell}$)"
                 % smoothing_factor)
    fig.tight_layout()
    for out_path in out_paths:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _percentiles(values, pct=(1, 5, 50, 95, 99)):
    """Percentiles of an array as a JSON-friendly dict."""
    return {str(p): float(v) for p, v in zip(pct, np.percentile(values, pct))}


def summarise(run, width, dense):
    """Summary statistics comparing the two images."""
    img_vt, img_yt = run["img_vt"], run["img_yt"]
    pixel_area = (width / img_vt.shape[0]) ** 2

    ratio = img_yt / img_vt
    rel_err = ratio - 1
    inset_stats = {}
    for name, ins in run["insets"].items():
        r = ins["img_yt"] / ins["img_vt"]
        inset_stats[name] = {
            "window_kpc": [float(w) for w in ins["window"]],
            "npix": int(ins["img_vt"].shape[0]),
            "ratio_min": float(r.min()),
            "ratio_max": float(r.max()),
            "ratio_percentiles": _percentiles(r),
            "peak_ratio_yt_over_vortrace": float(
                ins["img_yt"].max() / ins["img_vt"].max()),
            "mean_column_density_vortrace": float(ins["img_vt"].mean()),
            "mean_column_density_yt": float(ins["img_yt"].mean()),
        }

    return {
        "total_mass_vortrace": float(img_vt.sum() * pixel_area),
        "total_mass_yt": float(img_yt.sum() * pixel_area),
        "peak_column_density_vortrace": float(img_vt.max()),
        "peak_column_density_yt": float(img_yt.max()),
        "peak_ratio_yt_over_vortrace": float(img_yt.max() / img_vt.max()),
        "ratio_yt_over_vortrace_percentiles": _percentiles(ratio),
        "ratio_at_vortrace_peak": float(
            ratio[np.unravel_index(np.argmax(img_vt), img_vt.shape)]),
        "insets": inset_stats,
        "dense_pixel_fraction": float(dense.mean()),
        "rel_err_all_percentiles": _percentiles(rel_err),
        "rel_err_dense_percentiles": _percentiles(rel_err[dense]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0])
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAP,
                   help="Arepo HDF5 snapshot; for a multi-part snapshot the "
                        "first part (default: data/snap_500.0.hdf5, which "
                        "is not shipped with the repository)")
    p.add_argument("--outdir", type=Path, default=SCRIPT_DIR,
                   help="Output directory (default: this script's directory)")
    p.add_argument("--center", type=float, nargs=2, default=[90.0, 90.0],
                   metavar=("X", "Y"),
                   help="Image centre in code length units (kpc)")
    p.add_argument("--zrange", type=float, nargs=2, default=[88.0, 92.0],
                   metavar=("ZMIN", "ZMAX"),
                   help="Integration limits along z in code length units")
    p.add_argument("--width", type=float, default=6.0,
                   help="Image width in code length units (kpc)")
    p.add_argument("--npix", type=int, default=1024, help="Pixels per side")
    p.add_argument("--dense-width", type=float, default=0.015,
                   help="Width of the dense-region inset in code length "
                        "units (kpc)")
    p.add_argument("--void-width", type=float, default=0.25,
                   help="Width of the void inset in code length units (kpc)")
    p.add_argument("--inset-npix", type=int, default=256,
                   help="Pixels per side of each inset, which is projected "
                        "separately at this resolution")
    p.add_argument("--void-center", type=float, nargs=2,
                   default=[-0.25, 2.0], metavar=("X", "Y"),
                   help="Centre of the second (void) inset, relative to the "
                        "image centre, in code length units")
    p.add_argument("--vmax-factor", type=float, default=3.0,
                   help="Colour-scale ceiling as a multiple of the brightest "
                        "pixel in either image")
    p.add_argument("--err-lim", type=float, default=0.25,
                   help="Symmetric limit of the relative-error colour scale")
    p.add_argument("--hsml-factor", type=float, default=2.0,
                   help="SPH smoothing length as a multiple of the cell "
                        "radius, h = f * r_cell (yt's default is 2)")
    p.add_argument("--dense-percentile", type=float, default=75.0,
                   help="Pixels above this percentile of the exact column "
                        "density count as 'dense' in the error statistics")
    p.add_argument("--cache", type=Path, default=None,
                   help="NPZ file in which to cache the projected images "
                        "(written after a full run, read instead of "
                        "re-projecting if it exists); lets the figures be "
                        "re-styled without redoing the projections")
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat each timed projection this many times; "
                        "report the best")
    return p.parse_args()


def main_settings(args):
    """Settings that determine the main images (part of the cache key).

    The vortrace image depends only on the geometry; the yt image also
    depends on the smoothing length, so the two are keyed separately.
    """
    geometry = {"center": list(args.center), "width": args.width,
                "zrange": list(args.zrange), "npix": args.npix}
    return {"vortrace": geometry,
            "yt": {**geometry, "hsml_factor": args.hsml_factor}}


def inset_specs(args):
    """Per-inset settings (the rest of the cache key)."""
    return {
        "dense": {"center": "peak", "width": args.dense_width,
                  "npix": args.inset_npix, "hsml_factor": args.hsml_factor},
        "void": {"center": list(args.void_center), "width": args.void_width,
                 "npix": args.inset_npix, "hsml_factor": args.hsml_factor},
    }


class Snapshot:
    """Lazily loaded snapshot and vortrace point cloud."""

    def __init__(self, path):
        self.path = path
        self._cloud = None
        self.parts = len(snapshot_parts(path))
        self.ncells = None
        self.box_size = None
        self.t_build = None

    def cloud(self):
        """Build (once) and return the vortrace point cloud."""
        if self._cloud is None:
            print(f"Loading {self.path} ({self.parts} part(s))")
            pos, rho, vol, self.box_size = load_snapshot(self.path)
            self.ncells = int(len(rho))
            print(f"  {self.ncells} gas cells, box {self.box_size} "
                  f"{LENGTH_UNIT}")
            print("Building vortrace point cloud ...")
            self._cloud, self.t_build = build_cloud(pos, rho, vol,
                                                    self.box_size)
            print(f"  build {self.t_build:.3f} s")
        return self._cloud


def _geometry(args):
    return {"center": tuple(args.center), "width": args.width,
            "zrange": tuple(args.zrange), "npix": args.npix}


def project_main_vortrace(args, snap):
    """Exact projection of the main image."""
    pc = snap.cloud()
    print(f"vortrace projection ({args.npix}^2, {args.width} {LENGTH_UNIT}, "
          f"z in {list(args.zrange)}) ...")
    runs = [project_vortrace(pc, **_geometry(args))
            for _ in range(args.repeat)]
    img_vt, t_proj = min(runs, key=lambda r: r[1])
    print(f"  projection {t_proj:.3f} s")
    return {"parts": snap.parts, "ncells": snap.ncells,
            "box_size": snap.box_size, "img_vt": img_vt,
            "timings_vt": {"build_s": snap.t_build, "projection_s": t_proj}}


def project_main_yt(args):
    """SPH-kernel projection of the main image."""
    print(f"yt projection (h = {args.hsml_factor:g} r_cell) ...")
    runs = [project_yt(args.snapshot, smoothing_factor=args.hsml_factor,
                       **_geometry(args))
            for _ in range(args.repeat)]
    img_yt, timings_yt, yt_settings = min(
        runs, key=lambda r: r[1]["projection_s"])
    print(f"  load {timings_yt['load_s']:.3f} s, "
          f"projection {timings_yt['projection_s']:.3f} s")
    return {"img_yt": img_yt, "timings_yt": timings_yt,
            "factor": args.hsml_factor, "yt_settings": yt_settings}


def project_inset(args, snap, run, name, spec):
    """Project one inset region at its own resolution, both ways."""
    width = args.width
    ext = [-width / 2, width / 2, -width / 2, width / 2]
    half = spec["width"] / 2
    if spec["center"] == "peak":
        img_vt = run["img_vt"]
        dx = width / img_vt.shape[0]
        peak = np.unravel_index(np.argmax(img_vt), img_vt.shape)
        window = _window(ext[0] + (int(peak[0]) + 0.5) * dx,
                         ext[2] + (int(peak[1]) + 0.5) * dx, half, ext)
    else:
        window = _window(spec["center"][0], spec["center"][1], half, ext)

    wc = (args.center[0] + 0.5 * (window[0] + window[1]),
          args.center[1] + 0.5 * (window[2] + window[3]))
    geo = {"center": wc, "width": spec["width"], "zrange": tuple(args.zrange),
           "npix": spec["npix"]}
    print(f"{name} inset ({spec['npix']}^2, {spec['width']} {LENGTH_UNIT} "
          f"at {[round(c, 3) for c in wc]}) ...")
    ins_vt, t_vt = project_vortrace(snap.cloud(), **geo)
    ins_yt, t_yt, _ = project_yt(args.snapshot,
                                 smoothing_factor=run["factor"], **geo)
    print(f"  vortrace {t_vt:.3f} s, yt {t_yt['projection_s']:.3f} s")
    return {"settings": spec, "window": window,
            "img_vt": ins_vt, "img_yt": ins_yt}


def save_cache(path, run, settings):
    """Write the main and inset projections to an NPZ file."""
    arrays = {"img_vt": run["img_vt"], "img_yt": run["img_yt"]}
    meta = {k: v for k, v in run.items()
            if k not in ("img_vt", "img_yt", "insets")}
    meta["insets"] = {}
    for name, ins in run["insets"].items():
        arrays[f"inset_{name}_vt"] = ins["img_vt"]
        arrays[f"inset_{name}_yt"] = ins["img_yt"]
        meta["insets"][name] = {"settings": ins["settings"],
                                "window": ins["window"]}
    meta["settings"] = settings
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def load_cache(path):
    """Read a cache written by ``save_cache``."""
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        run = {k: v for k, v in meta.items() if k != "settings"}
        run["img_vt"], run["img_yt"] = f["img_vt"], f["img_yt"]
        for name, ins in run["insets"].items():
            ins["img_vt"] = f[f"inset_{name}_vt"]
            ins["img_yt"] = f[f"inset_{name}_yt"]
    return run, meta["settings"]


def obtain_projections(args):
    """Return the projections, reusing whatever part of the cache still
    matches the requested settings and recomputing the rest."""
    settings = main_settings(args)
    specs = inset_specs(args)
    cached, cached_settings = None, None
    if args.cache is not None and args.cache.exists():
        cached, cached_settings = load_cache(args.cache)
        print(f"Loaded cache {args.cache}")

    snap = Snapshot(args.snapshot)
    run = {}
    vt_keys = ("parts", "ncells", "box_size", "img_vt", "timings_vt")
    yt_keys = ("img_yt", "timings_yt", "factor", "yt_settings")
    cached_settings = cached_settings or {}
    if cached is not None and cached_settings.get("vortrace") == \
            settings["vortrace"]:
        run.update({k: cached[k] for k in vt_keys})
        print("  main vortrace projection reused from cache")
    else:
        run.update(project_main_vortrace(args, snap))
    if cached is not None and cached_settings.get("yt") == settings["yt"]:
        run.update({k: cached[k] for k in yt_keys})
        print("  main yt projection reused from cache")
    else:
        run.update(project_main_yt(args))

    main_ok = cached_settings.get("vortrace") == settings["vortrace"]
    run["insets"] = {}
    for name, spec in specs.items():
        old = (cached or {}).get("insets", {}).get(name)
        if main_ok and old is not None and old.get("settings") == spec:
            run["insets"][name] = old
            print(f"  {name} inset reused from cache")
        else:
            run["insets"][name] = project_inset(args, snap, run, name, spec)

    if args.cache is not None:
        save_cache(args.cache, run, settings)
        print(f"Cached projections in {args.cache}")
    return run, settings


def main():
    """Entry point."""
    args = parse_args()
    yt = _import_yt()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run, settings = obtain_projections(args)

    img_vt, img_yt, factor = run["img_vt"], run["img_yt"], run["factor"]
    dense_threshold = float(np.percentile(img_vt, args.dense_percentile))
    dense = img_vt > dense_threshold

    outputs = [args.outdir / "yt_comparison.pdf"]
    make_figure(run, args.width, vmax_factor=args.vmax_factor,
                err_lim=args.err_lim, out_paths=outputs)
    for out in outputs:
        print(f"Wrote {out}")

    hist_outputs = [args.outdir / "error_histogram.pdf"]
    make_histogram(img_vt, img_yt, smoothing_factor=factor,
                   out_paths=hist_outputs)
    for out in hist_outputs:
        print(f"Wrote {out}")

    stats = {
        "snapshot": {
            "name": args.snapshot.name,
            "parts": run["parts"],
            "ncells": run["ncells"],
            "box_size": run["box_size"],
        },
        "settings": settings,
        "insets": {name: ins["settings"]
                   for name, ins in run["insets"].items()},
        "figure": {
            "vmax_factor": args.vmax_factor,
            "err_lim": args.err_lim,
            "dense_threshold": dense_threshold,
        },
        "timings": {
            "note": f"best of {args.repeat} run(s), wall-clock seconds, "
                    "main image only",
            "vortrace": run["timings_vt"],
            "yt": run["timings_yt"],
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "versions": {
            "vortrace": getattr(vt, "__version__", "unknown"),
            "yt": yt.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "yt_settings": run["yt_settings"],
        "stats": summarise(run, args.width, dense),
    }
    stats_path = args.outdir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")
    print(json.dumps(stats["stats"], indent=2))


if __name__ == "__main__":
    main()
