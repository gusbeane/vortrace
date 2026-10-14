#!/usr/bin/env python
"""Time vortrace against the mesh-building codes COLT and SKIRT.

The paper states that codes which obtain the exact integral by constructing
the Voronoi mesh (COLT, SKIRT) are expensive to run for a simple projection.
This script puts numbers behind that claim: it computes the *same*
column-density image with vortrace, COLT and SKIRT, checks that the three
exact methods agree, and records wall-clock timings and peak memory.

Two test galaxies are used:

- ``small``: the Arepo test-suite snapshot used throughout the documentation
  (``tests/test_data/galaxy_interaction.hdf5``, 7217 gas cells), projected
  as in the quickstart (75 kpc window, 256^2, full box depth).
- ``large``: the isolated dwarf-galaxy snapshot of the paper's yt figure
  (``data/snap_500.*.hdf5``, 27M cells, not shipped with the repository),
  projected as in ``paper/yt_comparison`` (6 kpc window, 1024^2, 4 kpc slab).

The paper's own 6 kpc, 1024^2 projection of the large galaxy could not be
reproduced with either mesh code on the 24 GiB workstation used here: COLT's
CGAL mesh needs ~1.3 kB per cell and SKIRT's Voro++ mesh ~0.8 kB per cell,
and 23 of the 27 million cells lie inside that 6 x 6 x 4 kpc column, so the
mesh codes run out of memory (SKIRT thrashed in swap and was killed).  The
large-galaxy comparison is therefore made on a 1 kpc window, handing the mesh
codes only the cells within 0.25 kpc (8 kpc in z) of the projected column
(``--width 1 --npix 256 --subbox-margin 0.25 --subbox-zmargin 8 --tag w1``);
vortrace still works on the full 27M-cell cloud.

COLT and SKIRT are driven through ``colt_projection.py`` and
``skirt_projection.py`` in this directory, which build the codes on first
use into ``data/external/`` (see their docstrings for the pinned versions).
Neither code is a dependency of ``vortrace``.

    python paper/code_comparison/make_code_comparison.py --galaxy small \\
        --repeat 3 --cache data/external/runs/cache_small.npz
    python paper/code_comparison/make_code_comparison.py --galaxy large \\
        --width 1 --npix 256 --subbox-margin 0.25 --subbox-zmargin 8 \\
        --tag w1 --cache data/external/runs/cache_large_w1.npz

Outputs (written next to this script by default):

- ``stats_<galaxy>.json`` -- timings, memory, settings and image agreement
- ``timings.md``          -- one table per galaxy for the paper text
- ``code_comparison_<galaxy>.pdf`` -- the three images and the differences
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, SymLogNorm  # noqa: E402

import vortrace as vt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import Geometry, ProjectionResult  # noqa: E402
# pylint: enable=wrong-import-position,wrong-import-order,ungrouped-imports

SCRIPT_DIR = common.SCRIPT_DIR
REPO_ROOT = common.REPO_ROOT

GALAXIES = {
    "small": {
        "snapshot": REPO_ROOT / "tests" / "test_data" / "galaxy_interaction.hdf5",
        "description": "Arepo test-suite galaxy interaction (documentation)",
        # centre/zrange are filled in from the box size at run time
        "width": 75.0, "npix": 256, "slab": None,
    },
    "large": {
        "snapshot": REPO_ROOT / "data" / "snap_500.0.hdf5",
        "description": "isolated dwarf galaxy (paper yt comparison)",
        "center": (90.0, 90.0), "width": 6.0, "npix": 1024,
        "slab": (88.0, 92.0),
    },
}


# ---------------------------------------------------------------------------
# vortrace
# ---------------------------------------------------------------------------

def project_vortrace(snap, geom, *, repeat=1):
    """Exact column density with vortrace (kD-tree build + projection)."""
    t0 = time.perf_counter()
    b = snap.box_size
    pc = vt.ProjectionCloud(snap.pos, snap.rho, vol=snap.vol,
                            boundbox=[0, b, 0, b, 0, b])
    t_build = time.perf_counter() - t0

    ext = geom.extent
    extent = [[ext[0], ext[1]], [ext[2], ext[3]]]
    image, t_proj = None, np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        img = pc.grid_projection(extent, geom.npix, list(geom.zrange),
                                 center=None)
        dt = time.perf_counter() - t0
        if dt < t_proj:
            image, t_proj = img, dt
    return pc, ProjectionResult(
        code="vortrace", image=image,
        timings={"build_s": t_build, "projection_s": t_proj,
                 "total_s": t_build + t_proj},
        settings={"version": importlib.metadata.version("vortrace"),
                  "note": "serial build (OpenMP not available with Apple "
                          "clang); kD-tree build timed separately"},
        nthreads=1)


def vortrace_reference(pc, geom):
    """vortrace image for an alternative geometry (e.g. a code's effective
    z range) using an already built cloud."""
    ext = geom.extent
    extent = [[ext[0], ext[1]], [ext[2], ext[3]]]
    return pc.grid_projection(extent, geom.npix, list(geom.zrange),
                              center=None)


# ---------------------------------------------------------------------------
# External codes
# ---------------------------------------------------------------------------

def external_module(code):
    """Import ``colt_projection`` or ``skirt_projection``."""
    return importlib.import_module(f"{code}_projection")


def external_kwargs(code, snap_ext, subbox):
    """Extra keyword arguments for a code's ``project``.

    SKIRT must be told the domain that contains the cells it is given
    (sites outside it are discarded and cells are clipped at its walls), and
    reads its binary column format far faster than text for large inputs.
    """
    kwargs = {}
    if code == "skirt":
        if subbox is not None:
            kwargs["domain"] = (subbox["lo"], subbox["hi"])
        if snap_ext.ncells > 1_000_000:
            kwargs["input_format"] = "scol"
    return kwargs


def run_external(code, snap, geom, *, nthreads, workdir, repeat=1, **kwargs):
    """Run one external code, best of ``repeat`` by total time."""
    mod = external_module(code)
    best = None
    for i in range(repeat):
        result = mod.project(snap, geom, nthreads=nthreads,
                             workdir=workdir, reuse_input=i > 0, **kwargs)
        if best is None or result.timings["total_s"] < best.timings["total_s"]:
            best = result
    return best


def external_geometry(code, geom, subbox, box_size):
    """Geometry a code will actually integrate.

    COLT limits each ray to ``proj_depth`` so it reproduces the slab.  SKIRT
    integrates through the whole z extent of its domain (the subbox or the
    full box), so its image is compared with a vortrace image over that z
    range.
    """
    if code == "skirt":
        zr = (subbox["lo"][2], subbox["hi"][2]) if subbox else (0.0, box_size)
        if tuple(zr) != tuple(geom.zrange):
            return Geometry(geom.center, geom.width, tuple(zr), geom.npix)
    return geom


# ---------------------------------------------------------------------------
# Figure and tables
# ---------------------------------------------------------------------------

def _symlog(lim):
    """Symmetric log colour scale spanning +-lim."""
    kwargs = {"linthresh": lim * 1e-3, "vmin": -lim, "vmax": lim}
    return SymLogNorm(**kwargs)  # pylint: disable=unexpected-keyword-arg


def make_figure(images, references, geom, out_paths, *, title):
    """Top row: column density from each code.  Bottom row: relative
    difference of each external code from its vortrace reference."""
    codes = list(images)
    n = len(codes)
    ext = geom.extent
    ext = [ext[0] - geom.center[0], ext[1] - geom.center[0],
           ext[2] - geom.center[1], ext[3] - geom.center[1]]

    positive = np.concatenate([im[im > 0].ravel() for im in images.values()])
    norm = LogNorm(vmin=np.percentile(positive, 1),
                   vmax=max(im.max() for im in images.values()))

    fig = plt.figure(figsize=(4.8 * n + 1.0, 8.8))
    # columns: image | cbar | gap, repeated; the top row only uses the last
    # colorbar, the diff row one colorbar per panel
    widths = ([1, 0.05, 0.3] * n)[:-1]
    gs = fig.add_gridspec(2, 3 * n - 1, width_ratios=widths,
                          wspace=0.1, hspace=0.18,
                          left=0.06, right=0.96, top=0.9, bottom=0.07)
    ax0 = None
    for j, code in enumerate(codes):
        ax = fig.add_subplot(gs[0, 3 * j], sharex=ax0, sharey=ax0)
        ax0 = ax0 or ax
        im = ax.imshow(images[code].T, origin="lower", extent=ext,
                       norm=norm, cmap="viridis", interpolation="nearest")
        ax.set_title(code)
        if j == 0:
            ax.set_ylabel(f"y [{common.LENGTH_UNIT}]")
        else:
            ax.tick_params(labelleft=False)
        if j == n - 1:
            fig.colorbar(im, cax=fig.add_subplot(gs[0, 3 * j + 1]),
                         label=f"Column density [{common.COL_DENSITY_UNIT}]")
        if code not in references:
            continue
        axd = fig.add_subplot(gs[1, 3 * j], sharex=ax0, sharey=ax0)
        ref = references[code]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(ref > 0, (images[code] - ref) / ref, 0.0)
        lim = max(np.abs(rel).max(), 1e-8)
        imd = axd.imshow(rel.T, origin="lower", extent=ext, cmap="bwr",
                         norm=_symlog(lim), interpolation="nearest")
        fig.colorbar(imd, cax=fig.add_subplot(gs[1, 3 * j + 1]),
                     label=f"({code} - vortrace) / vortrace")
        axd.set_xlabel(f"x [{common.LENGTH_UNIT}]")
        if j == 1:
            axd.set_ylabel(f"y [{common.LENGTH_UNIT}]")
        else:
            axd.tick_params(labelleft=False)
    fig.suptitle(title)
    for p in out_paths:
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_s(x):
    return "--" if x is None else (f"{x:.2f}" if x < 100 else f"{x:.0f}")


def _fmt_gb(x):
    return "--" if x is None else f"{x / 2**30:.1f}"


def timing_table(stats):
    """Markdown table of wall-clock times for one galaxy."""
    rows = ["| code | threads | mesh / tree [s] | ray tracing [s] | "
            "executable [s] | total incl. file I/O [s] | peak RSS [GiB] | "
            "max rel. diff. vs vortrace |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    vtt = stats["vortrace"]["timings"]
    rows.append(f"| vortrace | 1 | {_fmt_s(vtt['build_s'])} | "
                f"{_fmt_s(vtt['projection_s'])} | -- | {_fmt_s(vtt['total_s'])} | "
                f"{_fmt_gb(stats['vortrace'].get('peak_rss_bytes'))} | -- |")
    for run in stats["external"]:
        t = run["timings"]
        mesh = t.get("mesh_s")
        proj = next((t[k] for k in ("projection_s", "trace_s", "probe_s")
                     if k in t), None)
        rss = t.get("peak_rss_bytes", run["settings"].get("peak_rss_bytes"))
        if rss is None:
            rss = run["settings"].get("skirt_peak_memory_bytes")
        err = run["agreement"]["max_abs_rel_err"] if run.get("agreement") \
            else None
        rows.append(
            f"| {run['code']} | {run['nthreads']} | {_fmt_s(mesh)} | "
            f"{_fmt_s(proj)} | {_fmt_s(t.get('run_s'))} | "
            f"{_fmt_s(t['total_s'])} | {_fmt_gb(rss)} | "
            f"{'--' if err is None else f'{err:.1e}'} |")
    return "\n".join(rows)


def write_timings_md(path, galaxy, stats):
    """Replace (or append) this galaxy's section in timings.md."""
    header = f"## {galaxy}"
    sn, g = stats["snapshot"], stats["geometry"]
    cells = f"{sn['ncells']} cells"
    if sn.get("ncells_external") not in (None, sn["ncells"]):
        cells += f" ({sn['ncells_external']} handed to the mesh codes)"
    section = (f"{header}\n\n{sn['description']}: {cells}, "
               f"{g['npix']}^2 pixels, {g['width']} kpc window, "
               f"z in {g['zrange']} kpc.  Best of {stats['repeat']} run(s); "
               f"machine: {stats['machine']['platform']}, "
               f"{stats['machine']['cpu_count']} cores.\n\n"
               f"{timing_table(stats)}\n")
    path = Path(path)
    text = (path.read_text(encoding="utf-8") if path.exists()
            else "# Code comparison timings\n\n")
    parts = text.split("\n## ")
    kept = [parts[0]] + [p for p in parts[1:]
                         if not p.startswith(galaxy + "\n")]
    text = "\n## ".join(kept).rstrip("\n") + "\n\n" + section
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0])
    p.add_argument("--galaxy", choices=sorted(GALAXIES), default="small")
    p.add_argument("--snapshot", type=Path, default=None,
                   help="Override the snapshot path for the chosen galaxy")
    p.add_argument("--codes", nargs="*", default=["colt", "skirt"],
                   choices=["colt", "skirt"],
                   help="External codes to run (default: both)")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8],
                   help="Thread counts for the external codes "
                        "(vortrace is serial on this machine)")
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat each timed run this many times; report the "
                        "best")
    p.add_argument("--npix", type=int, default=None)
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--subbox-margin", type=float, default=None,
                   help="Hand the external codes only the cells within this "
                        "margin (kpc) of the projected region instead of "
                        "the full box; the margin must exceed the local "
                        "cell size by a wide factor (default: full box)")
    p.add_argument("--subbox-zmargin", type=float, default=None,
                   help="Margin along z for the subbox (default: same as "
                        "--subbox-margin).  SKIRT integrates the whole z "
                        "extent of what it is given, so a wide z margin "
                        "keeps its image close to the slab projection")
    p.add_argument("--tag", default=None,
                   help="Suffix for the output names, e.g. 'w1' gives "
                        "stats_large_w1.json (default: none)")
    p.add_argument("--outdir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--workdir", type=Path,
                   default=common.EXTERNAL_DIR / "runs",
                   help="Scratch directory for the external codes' input "
                        "and output files")
    p.add_argument("--cache", type=Path, default=None,
                   help="NPZ file in which to save the images after a run "
                        "(used by --figure-only)")
    p.add_argument("--figure-only", action="store_true",
                   help="Do not run anything: rebuild the figure and "
                        "timings.md from --cache and the existing stats file")
    return p.parse_args()


def figure_only(args, cfg, label):
    """Rebuild the figure and timings.md from a cache and stats file."""
    stats = json.loads((args.outdir / f"stats_{label}.json").read_text())
    data = np.load(args.cache)
    g = stats["geometry"]
    geom = Geometry(tuple(g["center"]), g["width"], tuple(g["zrange"]),
                    g["npix"])
    images = {k: data[k] for k in data.files if not k.startswith("ref:")}
    references = {k[4:]: data[k] for k in data.files if k.startswith("ref:")}
    shown = {"vortrace": images["vortrace"]}
    shown_refs = {}
    for code in args.codes:
        key = f"{code} ({min(args.threads)} thr)"
        if key in images:
            shown[code] = images[key]
            shown_refs[code] = references[key]
    make_figure(shown, shown_refs, geom,
                [args.outdir / f"code_comparison_{label}.pdf",
                 args.outdir / f"code_comparison_{label}.png"],
                title=f"{cfg['description']} "
                      f"({stats['snapshot']['ncells']} cells)")
    write_timings_md(args.outdir / "timings.md", label, stats)
    print(f"Rebuilt figure and timings.md for {label}")


def main():
    args = parse_args()
    cfg = GALAXIES[args.galaxy]
    label = args.galaxy + (f"_{args.tag}" if args.tag else "")
    snap_path = args.snapshot or cfg["snapshot"]
    if not Path(snap_path).exists():
        sys.exit(f"{snap_path} not found (the large snapshot is not shipped "
                 "with the repository; pass --snapshot)")

    if args.figure_only:
        figure_only(args, cfg, label)
        return

    print(f"Loading {snap_path}")
    snap, t_load = common.timed(common.load_snapshot, snap_path)
    b = snap.box_size
    print(f"  {snap.ncells} gas cells, box {b} kpc, {t_load:.1f} s")
    n_moved = common.separate_duplicates(snap)
    if n_moved:
        print(f"  {n_moved} cells with coincident positions displaced by "
              "1e-6 kpc (handed to every code)")

    center = tuple(cfg.get("center") or (b / 2, b / 2))
    zrange = tuple(cfg["slab"] or (0.0, b))
    geom = Geometry(center=center, width=args.width or cfg["width"],
                    zrange=zrange, npix=args.npix or cfg["npix"])
    print(f"Geometry: {geom}")

    stats = {
        "snapshot": {"name": Path(snap_path).name, "parts": snap.nparts,
                     "ncells": snap.ncells, "box_size": b,
                     "description": cfg["description"], "load_s": t_load,
                     "duplicate_cells_displaced": n_moved},
        "geometry": geom.to_dict(),
        "repeat": args.repeat,
        "machine": common.machine_info(),
        "loadavg_at_start": list(os.getloadavg()),
        "external": [],
    }

    # vortrace --------------------------------------------------------------
    print("vortrace ...")
    pc, res_vt = project_vortrace(snap, geom, repeat=args.repeat)
    print(f"  build {res_vt.timings['build_s']:.2f} s, projection "
          f"{res_vt.timings['projection_s']:.2f} s")
    # ru_maxrss is in bytes on macOS and kilobytes on Linux; it covers this
    # whole process (snapshot load included), which is what a user pays.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        maxrss *= 1024
    stats["vortrace"] = {"timings": res_vt.timings,
                         "settings": res_vt.settings,
                         "peak_rss_bytes": int(maxrss)}
    images = {"vortrace": res_vt.image}
    references = {}

    # cells handed to the mesh codes --------------------------------------
    snap_ext, subbox = snap, None
    if args.subbox_margin is not None:
        m = args.subbox_margin
        mz = args.subbox_zmargin if args.subbox_zmargin is not None else m
        ext = geom.extent
        lo = [max(ext[0] - m, 0.0), max(ext[2] - m, 0.0),
              max(geom.zrange[0] - mz, 0.0)]
        hi = [min(ext[1] + m, b), min(ext[3] + m, b),
              min(geom.zrange[1] + mz, b)]
        snap_ext = snap.subbox(lo, hi)
        subbox = {"margin_kpc": m, "zmargin_kpc": mz, "lo": lo, "hi": hi}
        stats["snapshot"]["subbox"] = subbox
        print(f"  subbox {lo} - {hi}: {snap_ext.ncells} cells")
    stats["snapshot"]["ncells_external"] = snap_ext.ncells

    # vortrace reference images for every geometry the external codes will
    # integrate, computed now so the kD-tree can be freed before the
    # memory-hungry mesh codes run.
    ref_images = {}
    for code in args.codes:
        eff = external_geometry(code, geom, subbox, b)
        if eff == geom:
            ref_images[code] = res_vt.image
        elif eff.to_dict() not in [g.to_dict() for g in ref_images.values()
                                   if isinstance(g, Geometry)]:
            print(f"  vortrace reference for {code} over z in {eff.zrange} ...")
            ref_images[code], t_ref = common.timed(vortrace_reference, pc, eff)
            print(f"    {t_ref:.1f} s")
        stats.setdefault("reference_geometries", {})[code] = eff.to_dict()
    del pc
    gc.collect()

    # external codes ------------------------------------------------------
    for code in args.codes:
        mod = external_module(code)
        exe = mod.ensure_built()
        print(f"{code}: {exe}")
        eff = external_geometry(code, geom, subbox, b)
        kwargs = external_kwargs(code, snap_ext, subbox)
        for nthreads in args.threads:
            workdir = args.workdir / f"{code}_{label}_t{nthreads}"
            print(f"  {code} with {nthreads} thread(s) ...")
            res = run_external(code, snap_ext, geom, nthreads=nthreads,
                               workdir=workdir, repeat=args.repeat, **kwargs)
            zr = res.settings.get("effective_zrange")
            if zr is not None and tuple(zr) != tuple(eff.zrange):
                raise RuntimeError(f"{code} integrated z in {zr}, expected "
                                   f"{eff.zrange}")
            ref = ref_images[code]
            agreement = common.compare_images(ref, res.image)
            print(f"    total {res.timings['total_s']:.1f} s, "
                  f"max |rel err| {agreement['max_abs_rel_err']:.2e}")
            stats["external"].append({
                "code": code, "nthreads": nthreads,
                "timings": res.timings, "settings": res.settings,
                "effective_geometry": eff.to_dict(),
                "agreement": agreement,
            })
            key = f"{code} ({nthreads} thr)"
            images[key] = res.image
            references[key] = ref
            # keep the stats file current in case a later run dies
            args.outdir.mkdir(parents=True, exist_ok=True)
            (args.outdir / f"stats_{label}.json").write_text(
                json.dumps(stats, indent=2, default=str))

    # outputs -------------------------------------------------------------
    stats["loadavg_at_end"] = list(os.getloadavg())
    args.outdir.mkdir(parents=True, exist_ok=True)
    stats_path = args.outdir / f"stats_{label}.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    write_timings_md(args.outdir / "timings.md", label, stats)
    if args.cache:
        np.savez_compressed(args.cache, **dict(images),
                            **{"ref:" + k: v for k, v in references.items()})

    # figure: one panel per code at the lowest thread count ----------------
    shown = {"vortrace": images["vortrace"]}
    shown_refs = {}
    for code in args.codes:
        key = f"{code} ({min(args.threads)} thr)"
        if key in images:
            shown[code] = images[key]
            shown_refs[code] = references[key]
    make_figure(shown, shown_refs, geom,
                [args.outdir / f"code_comparison_{label}.pdf",
                 args.outdir / f"code_comparison_{label}.png"],
                title=f"{cfg['description']} ({snap.ncells} cells)")
    print(f"Wrote {stats_path} and timings.md")


if __name__ == "__main__":
    main()
