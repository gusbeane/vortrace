"""Run COLT's ``projections`` module on an Arepo snapshot.

COLT (Cosmic Lyman-alpha Transfer, A. Smith; bitbucket.org/aaron_smith/colt)
builds the Voronoi mesh of the gas cells with CGAL (3D Delaunay triangulation)
and integrates the density along one parallel ray per pixel.  This module

- builds COLT from the source tree in ``data/external/colt``
  (``ensure_built``),
- converts a :class:`common.Snapshot` (Arepo code units) into a COLT
  initial-conditions file in cgs (``write_ics``),
- writes a ``!projections`` ``config.yaml`` for a camera looking along +z,
  runs the executable with ``OMP_NUM_THREADS`` threads, parses COLT's own
  timing output and returns the image in code units ``1e10 Msun kpc^-2``
  indexed ``[x, y]`` (``project``).

How the column density is obtained
----------------------------------
COLT's ``[rho, avg]`` projection returns, per pixel, ``int rho dl / L_norm``.
With ``override_volume: true`` the normalisation ``L_norm`` is the configured
``proj_depth`` (the *full* line-of-sight depth) instead of the clipped ray
length, so the column density is simply ``rho_avg * proj_depth``.  The
numerator is the exact Voronoi line integral through the ``[zmin, zmax]``
slab, clipped to the simulation bounding box (``bbox`` dataset in the IC
file, set to the full ``[0, box_size]^3`` box; the clipping shrinks the box by
a relative 1e-10 on each face, a negligible effect).  ``avoid_edges: false``
lets rays traverse the hull (edge) cells, whose Voronoi regions are unbounded
in COLT but are cut by the bounding box, exactly like vortrace's ``boundbox``
treatment.  ``save_connections: false`` forces the mesh to be rebuilt on every
run (otherwise COLT caches the connectivity in the IC file and later runs
would skip the mesh construction).

Camera orientation (verified against vortrace in ``--selftest``):
``camera: [0, 0, 1]`` is the ray propagation direction, ``camera_north:
[0, 1, 0]`` gives image y = +y and image x = cross(y, dir) = +x, and the
stored dataset ``proj/rho_avg`` has shape ``(n_cameras, nx, ny)`` so ``[0]``
is already indexed ``[x, y]``.

Run ``python paper/code_comparison/colt_projection.py --selftest`` for the
validation against vortrace on the small docs galaxy and ``--scaling N`` for
the synthetic timing test (peak RSS included).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Locations and build configuration
# ---------------------------------------------------------------------------
COLT_SRC = common.EXTERNAL_DIR / "colt"          # unpacked bitbucket tarball
COLT_BUILD = common.EXTERNAL_DIR / "colt_build"  # objects, defines, exe
COLT_EXE = COLT_BUILD / "colt"
DEPS_PREFIX = common.EXTERNAL_DIR / "deps"       # locally built yaml-cpp
YAML_CPP_SRC = DEPS_PREFIX / "src" / "yaml-cpp-0.8.0"
RUNS_DIR = common.EXTERNAL_DIR / "colt_runs"

# Compile-time definitions (written to colt_build/defines.yaml and read by
# COLT's Makefile).  long_neighbors makes the flat neighbour arrays 64-bit:
# n_neighbors_tot ~ 16 N would overflow int32 beyond ~1.3e8 cells (cells
# themselves stay int32, fine for N < 2.1e9).
COLT_DEFINES = {
    "GEOMETRY": "voronoi",
    "HAVE_CGAL": True,
    "NO_MPI": True,
    "long_neighbors": True,
}
# Make variables overriding the MACHINE=homebrew profile: local yaml-cpp
# prefix, homebrew mpfr (for -I/-L), and -O3 (the homebrew profile has no
# optimisation flag at all).
COLT_MAKE_VARS = {
    "YAML": str(DEPS_PREFIX),
    "MPFR": "/opt/homebrew/opt/mpfr",
    "CFLAGS": "-Xpreprocessor -fopenmp -std=c++20 -O3",
}

T_CONST_K = 1.0e4         # constant gas temperature in the ICs (COLT needs T)
OUTPUT_BASE = "proj"
DATASET = "proj/rho_avg"  # [n_cameras, nx, ny] mean density [g/cm^3]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _yaml_dump_defines(defines):
    lines = ["# Compile-time definitions for the vortrace comparison build"]
    for key, val in defines.items():
        if isinstance(val, bool):
            val = "true" if val else "false"
        lines.append(f"{key}: {val}")
    return "\n".join(lines) + "\n"


def _ensure_yaml_cpp():
    """Build the static yaml-cpp library into DEPS_PREFIX if it is missing."""
    lib = DEPS_PREFIX / "lib" / "libyaml-cpp.a"
    if lib.exists():
        return lib
    if not YAML_CPP_SRC.exists():
        raise RuntimeError(
            "yaml-cpp is not installed and its source is missing at "
            f"{YAML_CPP_SRC}. Download https://github.com/jbeder/yaml-cpp/"
            "archive/refs/tags/0.8.0.tar.gz and unpack it into "
            f"{YAML_CPP_SRC.parent}.")
    build = YAML_CPP_SRC / "build"
    build.mkdir(exist_ok=True)
    log = DEPS_PREFIX / "yaml-cpp-build.log"
    common.run_logged(
        ["cmake", "..", f"-DCMAKE_INSTALL_PREFIX={DEPS_PREFIX}",
         "-DYAML_BUILD_SHARED_LIBS=OFF", "-DYAML_CPP_BUILD_TESTS=OFF",
         "-DYAML_CPP_BUILD_TOOLS=OFF", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
        cwd=build, log_path=log, measure_rss=False)
    common.run_logged(["cmake", "--build", ".", "-j8"], cwd=build,
                      log_path=log, measure_rss=False)
    common.run_logged(["cmake", "--install", "."], cwd=build, log_path=log,
                      measure_rss=False)
    return lib


def ensure_built(nthreads_build=8):
    """Build COLT if the executable is missing; return the executable path."""
    if COLT_EXE.exists():
        return COLT_EXE
    if not COLT_SRC.exists():
        raise FileNotFoundError(f"COLT source not found at {COLT_SRC}")
    _ensure_yaml_cpp()
    COLT_BUILD.mkdir(parents=True, exist_ok=True)
    defines = COLT_BUILD / "defines.yaml"
    defines.write_text(_yaml_dump_defines(COLT_DEFINES), encoding="utf-8")
    cmd = ["make", f"-j{nthreads_build}", f"DEFS={defines}",
           f"BUILD={COLT_BUILD / 'build'}", f"EXE={COLT_EXE}"]
    cmd += [f"{k}={v}" for k, v in COLT_MAKE_VARS.items()]
    common.run_logged(cmd, cwd=COLT_SRC, log_path=COLT_BUILD / "make.log",
                      measure_rss=False)
    if not COLT_EXE.exists():
        raise RuntimeError(f"COLT build finished but {COLT_EXE} is missing")
    return COLT_EXE


def colt_provenance():
    """Source and build provenance for the stats file."""
    info = {"source_dir": str(COLT_SRC),
            "source": "bitbucket.org/aaron_smith/colt (master tarball)",
            "exe": str(COLT_EXE)}
    makefile = COLT_SRC / "Makefile"
    if makefile.exists():
        mtime = _dt.datetime.fromtimestamp(makefile.stat().st_mtime)
        info["source_mtime"] = mtime.isoformat(timespec="seconds")
    defines = COLT_BUILD / "defines.yaml"
    if defines.exists():
        info["defines"] = defines.read_text(encoding="utf-8")
    ct_h = COLT_BUILD / "build" / "compile_time.h"
    if ct_h.exists():
        info["compile_time_h"] = ct_h.read_text(encoding="utf-8")
    info["make_vars"] = dict(COLT_MAKE_VARS)
    return info


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def write_ics(snap, path):
    """Write a COLT Voronoi IC file (cgs) from a Snapshot in code units.

    Positions are kept in the simulation frame (no shift); the ``bbox``
    dataset declares the full ``[0, box_size]^3`` box as the domain.  COLT
    checks every ``units`` attribute, so they must be present and exact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = snap.ncells
    r = np.ascontiguousarray(snap.pos, dtype=np.float64) * common.KPC_CM
    rho = np.ascontiguousarray(snap.rho, dtype=np.float64)
    rho = rho * common.UNIT_DENSITY_CGS
    b = snap.box_size * common.KPC_CM
    bbox = np.array([[0.0, 0.0, 0.0], [b, b, b]], dtype=np.float64)
    tmp = path.with_suffix(".tmp.hdf5")
    with h5py.File(tmp, "w") as f:
        f.attrs["n_cells"] = np.int32(n) if n < 2**31 else np.int64(n)
        f.create_dataset("r", data=r)
        f["r"].attrs["units"] = b"cm"
        f.create_dataset("rho", data=rho)
        f["rho"].attrs["units"] = b"g/cm^3"
        f.create_dataset("T", data=np.full(n, T_CONST_K, dtype=np.float64))
        f["T"].attrs["units"] = b"K"
        f.create_dataset("bbox", data=bbox)
        f["bbox"].attrs["units"] = b"cm"
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def make_config(geom, ics_rel, *, output_dir="output",
                output_base=OUTPUT_BASE):
    """COLT ``!projections`` configuration for a +z camera (values in cm)."""
    cx, cy = geom.center
    kpc = common.KPC_CM
    return {
        "init_file": ics_rel,
        "output_dir": output_dir,
        "output_base": output_base,
        # Mesh handling
        "save_connections": False,  # always rebuild; never cache into ICs
        "avoid_edges": False,       # trace through hull cells to the bbox
        # Fields
        "field_weight_pairs": [["rho", "avg"]],
        "override_volume": True,    # normalise by proj_depth
        "adaptive": False,
        # Camera: rays run along +z from zmin to zmax; image x=+x, y=+y
        "camera": [0, 0, 1],
        "camera_north": [0, 1, 0],
        "camera_center": [cx * kpc, cy * kpc, geom.zmid * kpc],
        "image_radius": 0.5 * geom.width * kpc,
        "n_pixels": int(geom.npix),
        "proj_depth": geom.depth * kpc,
    }


def _yaml_scalar(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in val) + "]"
    if isinstance(val, float):
        return repr(float(val))
    return str(val)


def dump_config(config):
    """Serialise the config dict as a COLT YAML document (no PyYAML)."""
    lines = ["# COLT config file (generated by colt_projection.py)",
             "--- !projections"]
    for key, val in config.items():
        lines.append(f"{key}: {_yaml_scalar(val)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
_TIMER_RE = re.compile(
    r"^\s*(?:\(\*\)|\|)?\s*([A-Za-z][A-Za-z_ ]*?)\s*=\s*"
    r"(?:\d+ days )?\d+:\d\d:\d+\.\d+ = ([0-9.eE+-]+) s")
_MESH_RE = re.compile(
    r"^\s*(n_edges|n_inner_edges|n_neib_tot|n_circ_tot)\s*=\s*(\d+)")


def parse_colt_log(text):
    """COLT's reported timers (seconds) and mesh counts from a run log."""
    reported, mesh = {}, {}
    section = None
    for raw in text.replace("\b", "").splitlines():
        line = raw.rstrip()
        if line.startswith("CGAL timings"):
            section = "cgal"
        elif line.startswith("Clock timings"):
            section = "clock"
        m = _TIMER_RE.match(line)
        if m:
            name = m.group(1).strip()
            if section == "cgal":
                name = "CGAL " + name
            reported[name] = float(m.group(2))
            continue
        m = _MESH_RE.match(line)
        if m:
            mesh[m.group(1)] = int(m.group(2))
    return reported, mesh


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _time_l_works():
    """Whether ``/usr/bin/time -l`` runs here (sandboxes may block it)."""
    try:
        proc = subprocess.run(["/usr/bin/time", "-l", "true"],
                              capture_output=True, check=False)
        return proc.returncode == 0
    except OSError:
        return False


def project(snap, geom, *, nthreads, workdir, exe=None, reuse_ics=False,
            reuse_input=None, timeout=None):
    """Column-density image of ``snap`` through ``geom`` computed by COLT.

    ``workdir`` receives ``ics/snapshot.hdf5``, ``config.yaml``, ``colt.log``
    and ``output/proj.hdf5``.  With ``reuse_ics`` (alias ``reuse_input``) an
    existing IC file is kept.  The peak resident set size of the COLT process
    (bytes) is recorded in ``settings["peak_rss_bytes"]``: from
    ``/usr/bin/time -l`` via ``common.run_logged`` when that works, otherwise
    from ``getrusage(RUSAGE_CHILDREN)`` (the maximum over all child processes
    of this Python process so far); ``settings["peak_rss_source"]`` says
    which.
    """
    t_start = time.perf_counter()
    exe = Path(exe) if exe is not None else ensure_built()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ics = workdir / "ics" / "snapshot.hdf5"
    timings = {}

    if (reuse_ics or reuse_input) and ics.exists():
        timings["write_ics_s"] = 0.0
    else:
        _, timings["write_ics_s"] = common.timed(write_ics, snap, ics)

    config = make_config(geom, os.path.relpath(ics, workdir))
    (workdir / "config.yaml").write_text(dump_config(config),
                                         encoding="utf-8")
    out_dir = workdir / config["output_dir"]
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{config['output_base']}.hdf5"
    if out_file.exists():
        out_file.unlink()

    cmd = [str(exe), "config.yaml"]
    env = {"OMP_NUM_THREADS": str(int(nthreads))}
    log_path = workdir / "colt.log"
    use_time = _time_l_works()
    _, timings["run_s"] = common.run_logged(
        cmd, cwd=workdir, log_path=log_path, env=env, timeout=timeout,
        measure_rss=use_time)
    if use_time:
        rss = common.run_logged.last_peak_rss_bytes
        rss_source = "/usr/bin/time -l"
    else:   # macOS reports ru_maxrss in bytes
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rss_source = ("getrusage(RUSAGE_CHILDREN).ru_maxrss "
                      "(max over all children so far)")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    reported, mesh = parse_colt_log(log_text)
    if "CGAL Calculation" in reported:
        timings["mesh_s"] = reported["CGAL Calculation"]
    if "Projections" in reported:
        timings["trace_s"] = reported["Projections"]
    if "Read time" in reported:
        timings["read_s"] = reported["Read time"]   # includes the mesh
    if "Write time" in reported:
        timings["write_output_s"] = reported["Write time"]
    timings["colt_reported"] = reported

    t_read = time.perf_counter()
    with h5py.File(out_file, "r") as f:
        dset = f[DATASET]
        rho_avg = np.asarray(dset[0], dtype=np.float64)   # (nx, ny) g/cm^3
        units = dset.attrs.get("units", b"")
        depth_cm = float(f["proj"].attrs["depth"])
    if isinstance(units, bytes):
        units = units.decode()
    if units not in ("", "g/cm^3"):
        raise RuntimeError(f"Unexpected units for {DATASET}: {units!r}")
    if not np.isclose(depth_cm, config["proj_depth"], rtol=1e-12):
        raise RuntimeError(f"COLT used depth {depth_cm} cm, expected "
                           f"{config['proj_depth']}")
    # override_volume -> rho_avg = int rho dl / proj_depth
    image = rho_avg * depth_cm / common.UNIT_COLDENS_CGS   # [x, y]
    timings["read_output_s"] = time.perf_counter() - t_read
    timings["total_s"] = time.perf_counter() - t_start

    settings = {
        "config": config,
        "config_yaml": dump_config(config),
        "dataset": DATASET,
        "dataset_units": units,
        "column_density": "rho_avg * depth (override_volume: true)",
        "T_const_K": T_CONST_K,
        "ic_file": str(ics),
        "workdir": str(workdir),
        "ncells": snap.ncells,
        "mesh": mesh,
        "peak_rss_bytes": rss,
        "peak_rss_source": rss_source,
        "colt": colt_provenance(),
    }
    return common.ProjectionResult("colt", image, timings, settings,
                                   int(nthreads))


# ---------------------------------------------------------------------------
# Validation and scaling tests
# ---------------------------------------------------------------------------

def _vortrace_reference(snap, geom):
    import vortrace as vt  # pylint: disable=import-outside-toplevel
    b = snap.box_size
    pc = vt.ProjectionCloud(snap.pos, snap.rho, vol=snap.vol,
                            boundbox=[0, b, 0, b, 0, b])
    cx, cy = geom.center
    h = geom.width / 2
    extent = [[cx - h, cx + h], [cy - h, cy + h]]
    return pc.grid_projection(extent, geom.npix, list(geom.zrange),
                              center=None)


def _orientation_report(img_ref, img):
    i_ref = np.unravel_index(np.argmax(img_ref), img_ref.shape)
    i_col = np.unravel_index(np.argmax(img), img.shape)
    # Marginal peak positions tell transposes and flips apart.
    px_ref = np.argmax(img_ref.sum(axis=1))
    py_ref = np.argmax(img_ref.sum(axis=0))
    px = np.argmax(img.sum(axis=1))
    py = np.argmax(img.sum(axis=0))
    err_t = np.abs(img.T - img_ref).max()
    err = np.abs(img - img_ref).max()
    return {"argmax_vortrace": tuple(int(i) for i in i_ref),
            "argmax_colt": tuple(int(i) for i in i_col),
            "x_marginal_peak": (int(px_ref), int(px)),
            "y_marginal_peak": (int(py_ref), int(py)),
            "transposed_is_better": bool(err_t < err)}


def selftest(nthreads=8):
    """Compare COLT with vortrace on the small docs galaxy."""
    snap_path = (common.REPO_ROOT / "tests" / "test_data"
                 / "galaxy_interaction.hdf5")
    snap = common.load_snapshot(snap_path)
    b = snap.box_size
    print(f"Snapshot: {snap_path.name}, {snap.ncells} cells, box {b} kpc")
    exe = ensure_built()
    ok = True
    cases = (("full_depth", (0.0, b)),
             ("slab_200kpc", (b / 2 - 100.0, b / 2 + 100.0)))
    for tag, zrange in cases:
        geom = common.Geometry(center=(b / 2, b / 2), width=75.0,
                               zrange=zrange, npix=256)
        img_vt, t_vt = common.timed(_vortrace_reference, snap, geom)
        res = project(snap, geom, nthreads=nthreads,
                      workdir=RUNS_DIR / f"selftest_{tag}", exe=exe)
        cmp = common.compare_images(img_vt, res.image)
        timings = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in res.timings.items() if k != "colt_reported"}
        print(f"\n[{tag}] zrange={zrange}")
        print("  compare_images:", cmp)
        print("  orientation:", _orientation_report(img_vt, res.image))
        print(f"  timings: vortrace {t_vt:.3f} s; COLT", timings)
        print("  COLT reported:", res.timings["colt_reported"])
        print("  mesh:", res.settings["mesh"])
        if cmp["max_abs_rel_err"] is None or cmp["max_abs_rel_err"] > 1e-6:
            ok = False
    print("\nSELFTEST", "PASSED" if ok else "FAILED (max rel err > 1e-6)")
    return ok


def scaling_test(n_points=1_000_000, *, box=100.0, npix=256,
                 thread_counts=(1, 8), seed=42, reuse_ics=False):
    """Time COLT on N uniformly random points in a box (density 1)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n_points, 3))
    rho = np.ones(n_points)
    mass = np.full(n_points, box**3 / n_points)   # rho = 1 -> volumes
    snap = common.Snapshot(Path(f"synthetic_uniform_{n_points}"), pos, rho,
                           mass, box, 1)
    geom = common.Geometry(center=(box / 2, box / 2), width=box,
                           zrange=(0.0, box), npix=npix)
    exe = ensure_built()
    workdir = RUNS_DIR / f"scaling_{n_points}"
    results = {}
    for i, nt in enumerate(thread_counts):
        res = project(snap, geom, nthreads=nt, workdir=workdir, exe=exe,
                      reuse_ics=(reuse_ics or i > 0))
        rss = res.settings["peak_rss_bytes"]
        results[nt] = {
            "timings": {k: v for k, v in res.timings.items()
                        if k != "colt_reported"},
            "colt_reported": res.timings["colt_reported"],
            "peak_rss_GiB": None if rss is None else rss / 2**30,
            "peak_rss_source": res.settings["peak_rss_source"],
            "mesh": res.settings["mesh"],
            "image_mean": float(res.image.mean()),
            "image_std": float(res.image.std()),
        }
        print(f"\n[{n_points} points, {nt} threads]")
        for k, v in results[nt].items():
            print(f"  {k}: {v}")
    # Exact expectation for a uniform density box: column = rho * depth = box
    print(f"\nExpected uniform column density {box:.1f}; COLT image mean "
          f"{results[thread_counts[0]]['image_mean']:.6f}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="validate against vortrace")
    ap.add_argument("--scaling", type=int, metavar="N",
                    help="synthetic scaling test with N points")
    ap.add_argument("--nthreads", type=int, default=8,
                    help="threads for --selftest")
    ap.add_argument("--threads", default="1,8",
                    help="comma-separated thread counts for --scaling (one "
                         "process per count gives exact per-run peak RSS "
                         "when /usr/bin/time is unavailable)")
    ap.add_argument("--reuse-ics", action="store_true",
                    help="reuse existing scaling ICs")
    ap.add_argument("--build", action="store_true", help="only build COLT")
    args = ap.parse_args(argv)
    if args.build:
        print(ensure_built())
    if args.scaling:
        counts = tuple(int(t) for t in args.threads.split(","))
        scaling_test(args.scaling, thread_counts=counts,
                     reuse_ics=args.reuse_ics)
    if args.selftest or not (args.build or args.scaling):
        return 0 if selftest(args.nthreads) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
