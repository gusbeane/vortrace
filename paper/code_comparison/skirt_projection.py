"""Exact column-density projections through an Arepo Voronoi mesh with SKIRT 9.

SKIRT (https://skirt.ugent.be) is a Monte Carlo radiative-transfer code that
can import a hydrodynamical snapshot as a ``VoronoiMeshMedium``: it builds the
Voronoi tessellation of the generating sites with Voro++ and can then
integrate the medium density along parallel lines of sight with a
``DensityProbe`` + ``ParallelProjectionForm`` -- an exact mesh integral,
which is what vortrace computes.  This module builds SKIRT, writes the input
files, runs ``skirt`` and returns the projected image as a
``common.ProjectionResult`` in code units (1e10 Msun kpc^-2).

Build and runtime layout (everything under ``data/external``)::

    SKIRT9/          source: GitHub SKIRT/SKIRT9 master tarball, commit
                     d8e79d41a3d97abeeeaae4023acd202a231808f3 (2026-10-12)
    git -> SKIRT9    symlink: SKIRT looks for built-in resources relative
                     to the executable at ../../../git/SKIRT/resources
    SKIRT9_build/    out-of-tree CMake Release build (BUILD_SKIRT=ON,
                     BUILD_MAKE_UP=OFF); executable SKIRT/main/skirt
    resources/       external resource packs (SKIRT9_Resources_Core v8)
    skirt_runs/      work directories of the projections run from here

SKIRT facts this wrapper relies on (verified in the SKIRT 9 source):

* ``ParallelProjectionForm`` sends exactly one ray through the *centre* of
  every pixel when ``numSampling="1"`` (the same quadrature vortrace uses).
  With inclination = azimuth = 0 and **roll = 90 deg** the observer sits on
  the +z axis and the image axes coincide with model x and y, so the FITS
  image (stored [y, x]) transposed is the vortrace [x, y] image.
* The ray starts at z = 10 (fovX + fovY) and must lie *outside* the domain,
  so the medium is written with z shifted to put the domain centre at z = 0;
  ``project`` raises if the field of view is too small for the depth.
* ``VoronoiMeshSnapshot::buildMesh`` **discards** sites outside the medium
  domain and clips the cells at the domain walls (non-periodic Voro++
  container).  Rays always integrate through the whole domain, so a domain
  equal to the z slab would change the cells next to the slab faces.  The
  domain therefore defaults to the whole box and the z range SKIRT really
  integrated is reported as ``settings["effective_zrange"]``.
* Output is a FITS file with 32-bit float pixels in ``BUNIT = Msun/pc2``
  (``ExtragalacticUnits``), so agreement with a double-precision reference
  is limited to ~6e-8 relative.
* The version banner SKIRT prints contains the output of ``git describe``
  run by its CMake script; with the tarball unpacked inside this repository
  that describes *vortrace*, not SKIRT.  ``SKIRT_SOURCE_COMMIT`` records the
  true SKIRT commit.

No third-party FITS reader is needed; a minimal one is implemented below.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pylint: disable=wrong-import-position
import common  # noqa: E402
from common import (EXTERNAL_DIR, UNIT_COLDENS_MSUN_PC2,  # noqa: E402
                    UNIT_DENSITY_MSUN_PC3, Geometry, ProjectionResult,
                    Snapshot, run_logged)
# pylint: enable=wrong-import-position

SKIRT_SRC = EXTERNAL_DIR / "SKIRT9"
# SKIRT/SKIRT9 master as of 2026-10-12 (the tarball that was built)
SKIRT_SOURCE_COMMIT = "d8e79d41a3d97abeeeaae4023acd202a231808f3"
SKIRT_GIT_LINK = EXTERNAL_DIR / "git"      # SKIRT resolves resources via ../git
SKIRT_BUILD = EXTERNAL_DIR / "SKIRT9_build"
SKIRT_EXE = SKIRT_BUILD / "SKIRT" / "main" / "skirt"
SKIRT_RESOURCES = EXTERNAL_DIR / "resources"
SKIRT_CORE_PACK = SKIRT_RESOURCES / "SKIRT9_Resources_Core"
RESOURCE_URL = ("https://sciences.ugent.be/skirtextdat/SKIRT9/Resources/"
                "SKIRT9_Resources_Core_v8.zip")
RUNS_DIR = EXTERNAL_DIR / "skirt_runs"

CMAKE_OPTIONS = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SKIRT=ON",
                 "-DBUILD_MAKE_UP=OFF", "-DBUILD_DOX_STYLE=OFF"]

KPC_PC = 1000.0
PC_M = 3.08567758e16          # SKIRT's Constants::pc()
AU_M = 1.49597871e11          # SKIRT's Constants::AU()
PROBE_NAME = "coldens"
SKI_NAME = "skirt_proj"

# BUNIT -> factor converting the FITS values to 1e10 Msun kpc^-2.  SKIRT
# converts input (Msun/pc3, pc) and output (Msun/pc2) with the same
# constants, so for the default unit system no physical constant enters.
_COLDENS_TO_CODE = {
    "Msun/pc2": 1.0 / UNIT_COLDENS_MSUN_PC2,
    "Msun/AU2": (PC_M / AU_M) ** 2 / UNIT_COLDENS_MSUN_PC2,
    "g/cm2": 1.0 / common.UNIT_COLDENS_CGS,
    "kg/m2": 0.1 / common.UNIT_COLDENS_CGS,
}


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def ensure_built(nthreads_build=8):
    """Build SKIRT 9 if the executable is missing; return the executable path.

    The build is a plain CMake Release build of the ``skirt`` target only
    (``BUILD_MAKE_UP=OFF``, no Qt).  Also makes sure the ``git`` symlink and
    the Core resource pack SKIRT needs at runtime are in place.
    """
    if not SKIRT_SRC.is_dir():
        raise FileNotFoundError(
            f"SKIRT source not found at {SKIRT_SRC}; download the SKIRT9 "
            "master tarball from https://github.com/SKIRT/SKIRT9 and unpack "
            "it there")
    if not SKIRT_GIT_LINK.exists():
        SKIRT_GIT_LINK.symlink_to(SKIRT_SRC.name)
    if not SKIRT_EXE.exists():
        SKIRT_BUILD.mkdir(parents=True, exist_ok=True)
        run_logged(["cmake", *CMAKE_OPTIONS, str(SKIRT_SRC)], cwd=SKIRT_BUILD,
                   log_path=SKIRT_BUILD / "cmake_configure.log",
                   measure_rss=False)
        run_logged(["make", f"-j{int(nthreads_build)}"], cwd=SKIRT_BUILD,
                   log_path=SKIRT_BUILD / "make.log", measure_rss=False)
        if not SKIRT_EXE.exists():
            raise RuntimeError(
                f"SKIRT build finished but {SKIRT_EXE} is missing")
    if not (SKIRT_CORE_PACK / "version.txt").exists():
        _download_core_resources()
    return SKIRT_EXE


def _download_core_resources():
    """Fetch the SKIRT9_Resources_Core pack (what downloadResources.sh does)."""
    SKIRT_RESOURCES.mkdir(parents=True, exist_ok=True)
    zip_path = SKIRT_RESOURCES / "SKIRT9_Resources_Core_v8.zip"
    try:
        subprocess.run(["curl", "-sSL", "-o", str(zip_path), RESOURCE_URL],
                       check=True)
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d",
                        str(SKIRT_RESOURCES)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Could not download the SKIRT Core resource pack from "
            f"{RESOURCE_URL}; download it manually and unzip it into "
            f"{SKIRT_RESOURCES}") from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()


def skirt_version(exe):
    """The version banner SKIRT prints ('SKIRT v9.0 (git ... built on ...)').

    See the module docstring: the ``git`` part describes the enclosing
    repository, not SKIRT; ``SKIRT_SOURCE_COMMIT`` is the real commit.
    """
    proc = subprocess.run([str(exe), "-h"], capture_output=True, text=True,
                          check=False)
    m = re.search(r"Welcome to (SKIRT v[^\n]*\))", proc.stdout + proc.stderr)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

def write_medium_file(snap, path, *, z_offset=0.0, precision="%.17e"):
    """Write the gas cells as a SKIRT ``VoronoiMeshMedium`` input file.

    Columns: x, y, z in pc and the cell mass density in Msun/pc^3 (SKIRT
    ``massType="MassDensity"``), i.e. exactly the Arepo quantities converted
    with the exact factors 1e3 (kpc -> pc) and 1e10/1e9 (code density ->
    Msun/pc^3).  ``z_offset`` (kpc) is subtracted from z so the SKIRT domain
    can be centred on z = 0 (see module docstring).

    ``path`` ending in ``.txt`` gives SKIRT's column text format (header
    lines declare the units); ``.scol`` gives SKIRT's binary "stored
    columns" format, which SKIRT reads about ten times faster (values are
    exact doubles; ``precision`` is ignored).  With the text format,
    ``%.17e`` reproduces the doubles exactly; ``%.10e`` perturbs the site
    positions by ~1e-11 relative, which was measured to raise the maximum
    pixel error against vortrace from 6e-8 (the float32 FITS limit) to
    1.6e-7 on the test galaxy.
    """
    path = Path(path)
    pos_pc = np.asarray(snap.pos, dtype=np.float64) * KPC_PC
    pos_pc = pos_pc.copy()
    pos_pc[:, 2] -= z_offset * KPC_PC
    rho = np.asarray(snap.rho, dtype=np.float64) * UNIT_DENSITY_MSUN_PC3
    table = np.column_stack([pos_pc, rho])
    names = ["x", "y", "z", "rho"]
    units = ["pc", "pc", "pc", "Msun/pc3"]
    if path.suffix == ".scol":
        _write_scol(path, table, names, units)
        return
    header = ("Arepo gas cells for SKIRT VoronoiMeshMedium "
              "(massType MassDensity)\n"
              f"source snapshot: {snap.path}\n"
              f"z offset subtracted: {z_offset!r} kpc\n")
    for i, (n, u) in enumerate(zip(names, units), start=1):
        header += f"column {i}: {n} ({u})\n"
    np.savetxt(path, table, fmt=precision, header=header.rstrip("\n"))


def _write_scol(path, table, names, units):
    """SKIRT stored-columns binary format (SKIRT/tools/StoredColumns.hpp)."""
    def tag(s):
        s = s.encode("ascii")
        if not 1 <= len(s) <= 8:
            raise ValueError(f"scol tag {s!r} must be 1-8 characters")
        return s.ljust(8, b" ")
    nrows, ncols = table.shape
    with open(path, "wb") as f:
        f.write(b"SKIRT X\n")
        f.write(struct.pack("<Q", 0x010203040A0BFEFF))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", nrows))
        f.write(struct.pack("<Q", ncols))
        for n in names:
            f.write(tag(n))
        for u in units:
            f.write(tag(u))
        np.ascontiguousarray(table, dtype="<f8").tofile(f)
        f.write(b"SCOLEND\n")


_SKI_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Exact column density through an Arepo Voronoi mesh;
     written by vortrace/paper/code_comparison/skirt_projection.py -->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9"
                            producer="vortrace code_comparison">
  <MonteCarloSimulation userLevel="Expert" simulationMode="ExtinctionOnly"
                        numPackets="0">
    <random type="Random">
      <Random seed="0"/>
    </random>
    <units type="Units">
      <ExtragalacticUnits/>
    </units>
    <cosmology type="Cosmology">
      <LocalUniverseCosmology/>
    </cosmology>
    <sourceSystem type="SourceSystem">
      <SourceSystem minWavelength="0.09 micron" maxWavelength="100 micron"
                    wavelengths="0.55 micron" sourceBias="0.5"/>
    </sourceSystem>
    <mediumSystem type="MediumSystem">
      <MediumSystem>
        <media type="Medium">
          <VoronoiMeshMedium filename="{medium_filename}"
                             {box}
                             massType="MassDensity" massFraction="1"
                             importMetallicity="false"
                             importTemperature="false"
                             importVelocity="false"
                             importMagneticField="false"
                             importVariableMixParams="false" useColumns="">
            <materialMix type="MaterialMix">
              <MeanInterstellarDustMix/>
            </materialMix>
          </VoronoiMeshMedium>
        </media>
        <samplingOptions type="SamplingOptions">
          <SamplingOptions numDensitySamples="1" numPropertySamples="1"/>
        </samplingOptions>
        <grid type="SpatialGrid">
          <VoronoiMeshSpatialGrid {box}
                                  policy="ImportedMesh"/>
        </grid>
      </MediumSystem>
    </mediumSystem>
    <instrumentSystem type="InstrumentSystem">
      <InstrumentSystem/>
    </instrumentSystem>
    <probeSystem type="ProbeSystem">
      <ProbeSystem>
        <probes type="Probe">
          <DensityProbe probeName="{probe_name}" aggregation="Type"
                        probeAfter="Setup">
            <form type="Form">
              <ParallelProjectionForm inclination="0 deg" azimuth="0 deg"
                                      roll="90 deg"
                                      fieldOfViewX="{width_pc!r} pc"
                                      numPixelsX="{npix}"
                                      centerX="{cx!r} pc"
                                      fieldOfViewY="{width_pc!r} pc"
                                      numPixelsY="{npix}"
                                      centerY="{cy!r} pc"
                                      numSampling="1"/>
            </form>
          </DensityProbe>
        </probes>
      </ProbeSystem>
    </probeSystem>
  </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""


def write_ski(path, *, medium_filename, domain_lo_pc, domain_hi_pc,
              center_pc, width_pc, npix, probe_name=PROBE_NAME):
    """Write the SKIRT parameter (.ski) file for one column-density image.

    ``domain_lo_pc``/``domain_hi_pc`` are the medium domain corners in SKIRT
    coordinates (pc, already z-shifted), ``center_pc`` the image centre
    (x, y) in pc and ``width_pc`` the field of view in pc.
    """
    path = Path(path)
    lo = [float(v) for v in domain_lo_pc]
    hi = [float(v) for v in domain_hi_pc]
    cx, cy = (float(v) for v in center_pc)
    box = (f'minX="{lo[0]!r} pc" maxX="{hi[0]!r} pc" '
           f'minY="{lo[1]!r} pc" maxY="{hi[1]!r} pc" '
           f'minZ="{lo[2]!r} pc" maxZ="{hi[2]!r} pc"')
    ski = _SKI_TEMPLATE.format(medium_filename=medium_filename, box=box,
                               probe_name=probe_name, width_pc=float(width_pc),
                               npix=int(npix), cx=cx, cy=cy)
    path.write_text(ski, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def read_fits_image(path):
    """Minimal FITS primary-HDU reader: returns (data, header dict).

    Handles the files SKIRT writes (BITPIX -32/-64, NAXIS 2 or 3 with a
    single plane, BSCALE/BZERO).  ``data`` is a float64 array indexed
    [y, x] as stored in the file.
    """
    header = {}
    raw = Path(path).read_bytes()
    pos = 0
    while True:
        block = raw[pos:pos + 2880]
        if len(block) < 2880:
            raise ValueError(f"{path}: truncated FITS header")
        pos += 2880
        done = False
        for i in range(0, 2880, 80):
            card = block[i:i + 80].decode("ascii", errors="replace")
            key = card[:8].strip()
            if key == "END":
                done = True
                break
            if card[8:10] != "= ":
                continue
            value = card[10:]
            if "/" in value and not value.lstrip().startswith("'"):
                value = value.split("/", 1)[0]
            value = value.strip()
            if value.startswith("'"):
                value = value[1:value.rfind("'")].rstrip()
            header[key] = value
        if done:
            break
    bitpix = int(header["BITPIX"])
    naxis = int(header["NAXIS"])
    shape = [int(header[f"NAXIS{k}"]) for k in range(naxis, 0, -1)]  # C order
    dtype = {-32: ">f4", -64: ">f8", 16: ">i2", 32: ">i4", 64: ">i8"}[bitpix]
    count = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=pos)
    data = data.astype(np.float64).reshape(shape)
    data = (data * float(header.get("BSCALE", 1.0))
            + float(header.get("BZERO", 0.0)))
    if naxis == 3:
        if shape[0] != 1:
            raise ValueError(
                f"{path}: expected a single image plane, got {shape[0]}")
        data = data[0]
    return data, header


_LOG_RE = re.compile(r"^(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d)\.(\d{3})"
                     r"\s(.)\s(.*)$")
_PEAK_RE = re.compile(r"Peak memory usage: ([\d.]+) ([KMGT]?B)")


def parse_skirt_log(path):
    """Parse SKIRT's ``<prefix>_log.txt`` into timings and statistics.

    Returns a dict with the phase durations in seconds derived from the log
    timestamps (``mesh_s``: Voro++ tessellation incl. verification,
    ``read_input_s``: reading the medium file, ``search_s``: building the
    block/kd-tree search structure, ``probe_s``: the projection probe,
    ``setup_s``/``skirt_total_s``: SKIRT's own setup and total elapsed),
    SKIRT's own peak-memory figure (``peak_memory_bytes``, from
    getrusage ru_maxrss) and the site statistics SKIRT logs (``stats``).
    """
    entries = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = _LOG_RE.match(line)
        if not m:
            continue
        d, mo, y, hh, mm, ss, ms, level, msg = m.groups()
        t = _dt.datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss),
                         int(ms) * 1000)
        entries.append((t, level, msg.strip()))

    def first(pattern, start=0):
        rx = re.compile(pattern)
        for k in range(start, len(entries)):
            if rx.search(entries[k][2]):
                return k
        return None

    def span(start_pat, end_pat):
        i = first(start_pat)
        if i is None:
            return None
        j = first(end_pat, i + 1)
        if j is None:
            return None
        return (entries[j][0] - entries[i][0]).total_seconds()

    out = {
        # reading + parsing the medium file
        "read_input_s": span(r"VoronoiMeshMedium reads Voronoi sites",
                             r"Done reading"),
        # Voro++ tessellation of all cells incl. SKIRT's verification pass
        "mesh_s": span(r"Constructing Voronoi tessellation",
                       r"Done computing Voronoi tessellation"),
        # block lists + kd-trees used to locate the cell containing a point
        "search_s": span(r"Building data structures to accelerate searching",
                         r"Number of search trees"),
        # the probe is the only "setup output" in our ski file
        "probe_s": span(r"Starting setup output", r"Finished setup output"),
        "setup_s": span(r"Starting setup\.\.\.", r"Finished setup in"),
        "skirt_total_s": span(r"Starting simulation", r"Finished simulation"),
        "peak_memory_bytes": None,
    }
    k = first(_PEAK_RE.pattern)
    if k is not None:
        val, unit = _PEAK_RE.search(entries[k][2]).groups()
        out["peak_memory_bytes"] = int(
            float(val) * 1024 ** "BKMGT".index(unit[0]))
    stats = {}
    patterns = {
        "num_sites": r"Number of sites: (\d+)",
        "num_sites_outside_domain": r"Number of sites outside domain: (\d+)",
        "num_sites_too_nearby": r"Number of sites too nearby others: (\d+)",
        "num_sites_retained": r"Number of sites retained: (\d+)",
        "num_cells": r"Done computing Voronoi tessellation with (\d+) cells",
        "avg_neighbors": r"Average number of neighbors per cell: ([\d.]+)",
    }
    for key, pat in patterns.items():
        k = first(pat)
        if k is not None:
            val = re.search(pat, entries[k][2]).group(1)
            stats[key] = float(val) if "." in val else int(val)
    stats["invalid_cell_warnings"] = [
        e[2] for e in entries if "invalid Voronoi cells" in e[2]]
    stats["warnings"] = [e[2] for e in entries if e[1] == "!"]
    out["stats"] = stats
    return out


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------

def project(snap, geom, *, nthreads, workdir, exe=None, reuse_input=False,
            domain=None, input_format="txt", precision="%.17e",
            measure_memory=True, name=SKI_NAME):
    """Project ``snap`` with SKIRT according to ``geom``.

    Parameters
    ----------
    snap, geom : common.Snapshot, common.Geometry
    nthreads : int
        Threads for ``skirt -t``.
    workdir : path
        Directory for the medium file, .ski file, SKIRT log and FITS output.
    exe : path, optional
        SKIRT executable (default: ``ensure_built()``).
    reuse_input : bool
        Reuse an existing medium file in ``workdir`` instead of rewriting it.
    domain : (lo, hi), optional
        Medium domain corners in kpc (code coordinates).  Default is the full
        periodic box ``[0, box_size]^3`` -- the same as vortrace's
        ``boundbox``.  Sites outside the domain are discarded by SKIRT and
        the cells are clipped at its walls, so pass the bounding box of the
        cells you hand over (e.g. the ``subbox`` limits).  SKIRT integrates
        through the *entire* z extent of the domain, whatever ``geom.zrange``
        says; the range actually integrated is returned in
        ``settings["effective_zrange"]`` and must be used for the vortrace
        reference.
    input_format : "txt" or "scol"
        Text column file (human readable) or SKIRT's binary stored-columns
        file (much faster to read for tens of millions of cells).
    precision : str
        ``numpy.savetxt`` format for the text file.
    measure_memory : bool
        Let ``common.run_logged`` wrap skirt in ``/usr/bin/time -l`` to record
        the peak RSS (``settings["peak_rss_bytes"]``).  SKIRT's own
        ru_maxrss figure is recorded either way
        (``settings["skirt_peak_memory_bytes"]``).
    """
    t_start = time.perf_counter()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(exe) if exe is not None else ensure_built()

    b = float(snap.box_size)
    if domain is None:
        lo, hi = np.zeros(3), np.full(3, b)
    else:
        lo, hi = (np.asarray(v, dtype=float) for v in domain)
    zmin_dom, zmax_dom = float(lo[2]), float(hi[2])
    z_offset = 0.5 * (zmin_dom + zmax_dom)
    if geom.zrange[0] < zmin_dom - 1e-9 or geom.zrange[1] > zmax_dom + 1e-9:
        raise ValueError(f"geom.zrange {geom.zrange} is not inside the "
                         f"domain z range ({zmin_dom}, {zmax_dom})")
    # ParallelProjectionForm starts each ray at z = 10 (fovX + fovY) in model
    # coordinates; that point must lie outside the (z-centred) domain.
    ray_start = 10.0 * (2 * geom.width)
    half_depth = 0.5 * (zmax_dom - zmin_dom)
    if ray_start <= half_depth:
        raise ValueError(
            f"SKIRT starts the rays at z = 20*width = {ray_start} kpc, "
            f"inside the domain half-depth {half_depth} kpc; use a wider "
            "field of view or a shallower domain")

    timings = {}
    suffix = {"txt": ".txt", "scol": ".scol"}[input_format]
    medium_path = workdir / f"medium{suffix}"
    input_reused = reuse_input and medium_path.exists()
    if input_reused:
        timings["write_input_s"] = 0.0
    else:
        _, timings["write_input_s"] = common.timed(
            write_medium_file, snap, medium_path, z_offset=z_offset,
            precision=precision)

    cx, cy = geom.center
    ski_path = write_ski(
        workdir / f"{name}.ski", medium_filename=medium_path.name,
        domain_lo_pc=(lo - [0, 0, z_offset]) * KPC_PC,
        domain_hi_pc=(hi - [0, 0, z_offset]) * KPC_PC,
        center_pc=(cx * KPC_PC, cy * KPC_PC), width_pc=geom.width * KPC_PC,
        npix=geom.npix)

    # remove stale outputs so we never read an old image
    stale = list(workdir.glob(f"{name}_*.fits"))
    stale.append(workdir / f"{name}_log.txt")
    for old in stale:
        if old.exists():
            old.unlink()

    cmd = [str(exe), "-t", str(int(nthreads)), "-b", "-i", str(workdir),
           "-o", str(workdir), str(ski_path)]
    console_log = workdir / "console.log"
    # common.run_logged wraps the command in ``/usr/bin/time -l`` when
    # measure_rss is true and exposes the peak RSS afterwards.
    _, timings["run_s"] = run_logged(cmd, cwd=workdir, log_path=console_log,
                                     measure_rss=measure_memory)
    peak_rss = getattr(run_logged, "last_peak_rss_bytes", None)

    log_path = workdir / f"{name}_log.txt"
    log_info = (parse_skirt_log(log_path) if log_path.exists()
                else {"stats": {}})
    for key in ("mesh_s", "probe_s", "read_input_s", "search_s", "setup_s",
                "skirt_total_s"):
        if log_info.get(key) is not None:
            timings[key] = log_info[key]

    fits_files = sorted(workdir.glob(f"{name}_{PROBE_NAME}_*_Sigma.fits"))
    if len(fits_files) != 1:
        raise RuntimeError(f"expected one *_Sigma.fits in {workdir}, found "
                           f"{[f.name for f in fits_files]}; see {log_path}")
    t0 = time.perf_counter()
    data, header = read_fits_image(fits_files[0])
    bunit = header.get("BUNIT", "")
    if bunit not in _COLDENS_TO_CODE:
        raise RuntimeError(f"unexpected BUNIT {bunit!r} in {fits_files[0]}")
    image = np.ascontiguousarray(data.T) * _COLDENS_TO_CODE[bunit]   # [x, y]
    if image.shape != (geom.npix, geom.npix):
        raise RuntimeError(f"unexpected image shape {image.shape}")
    timings["read_output_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_start

    settings = {
        "ski_file": str(ski_path),
        "ski_contents": ski_path.read_text(encoding="utf-8"),
        "skirt_exe": str(exe),
        "skirt_version": skirt_version(exe),
        "skirt_source_commit": SKIRT_SOURCE_COMMIT,
        "skirt_log": str(log_path),
        "medium_file": str(medium_path),
        "input_format": input_format,
        "input_precision": precision if input_format == "txt" else "float64",
        "input_reused": bool(input_reused),
        "ncells_handed_over": int(snap.ncells),
        "domain_kpc": [lo.tolist(), hi.tolist()],
        "z_offset_kpc": z_offset,
        "effective_zrange": [zmin_dom, zmax_dom],
        "requested_zrange": list(geom.zrange),
        "fits_file": str(fits_files[0]),
        "BUNIT": bunit,
        "BITPIX": int(header.get("BITPIX", 0)),
        "orientation": "inclination 0, azimuth 0, roll 90 deg: image x = "
                       "model x, image y = model y; FITS [y, x] transposed "
                       "to [x, y]",
        "rays_per_pixel": 1,
        # peak RSS of the skirt process from /usr/bin/time -l (None if the
        # wrapper was disabled or unavailable) and SKIRT's own ru_maxrss
        "peak_rss_bytes": peak_rss,
        "skirt_peak_memory_bytes": log_info.get("peak_memory_bytes"),
        "skirt_stats": log_info.get("stats", {}),
    }
    return ProjectionResult(code="SKIRT", image=image, timings=timings,
                            settings=settings, nthreads=int(nthreads))


# ---------------------------------------------------------------------------
# Validation and scaling tests
# ---------------------------------------------------------------------------

def vortrace_reference(snap, geom):
    """vortrace grid projection with the full box as boundbox."""
    import vortrace as vt  # pylint: disable=import-outside-toplevel
    b = float(snap.box_size)
    cx, cy = geom.center
    w = geom.width
    pc = vt.ProjectionCloud(snap.pos, snap.rho, vol=snap.vol,
                            boundbox=[0, b, 0, b, 0, b])
    extent = [[cx - w / 2, cx + w / 2], [cy - w / 2, cy + w / 2]]
    return pc.grid_projection(extent, geom.npix, list(geom.zrange),
                              center=None)


def selftest(nthreads=8, npix=256, measure_memory=True):
    """Validate against vortrace on the small docs galaxy; print statistics."""
    snap_path = (common.REPO_ROOT / "tests" / "test_data"
                 / "galaxy_interaction.hdf5")
    snap = common.load_snapshot(snap_path)
    b = snap.box_size
    geom = Geometry(center=(b / 2, b / 2), width=75.0, zrange=(0.0, b),
                    npix=npix)
    print(f"snapshot: {snap_path} ({snap.ncells} cells, box {b} kpc)")
    print(f"geometry: {geom}")

    img_vt, t_vt = common.timed(vortrace_reference, snap, geom)
    res = project(snap, geom, nthreads=nthreads,
                  workdir=RUNS_DIR / "selftest_galaxy",
                  measure_memory=measure_memory)
    stats = common.compare_images(img_vt, res.image)
    print("compare_images(vortrace, SKIRT):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"vortrace reference time: {t_vt:.3f} s")
    print("SKIRT timings:", {k: round(v, 4) for k, v in res.timings.items()})
    print("SKIRT version:", res.settings["skirt_version"],
          "commit", res.settings["skirt_source_commit"])
    print("SKIRT stats:", res.settings["skirt_stats"])
    rss_time = (res.settings["peak_rss_bytes"] or 0) / 2**20
    rss_skirt = (res.settings["skirt_peak_memory_bytes"] or 0) / 2**20
    print(f"peak RSS (MiB): time -l: {rss_time:.1f}  "
          f"SKIRT-reported: {rss_skirt:.1f}")
    ivt = np.unravel_index(np.argmax(img_vt), img_vt.shape)
    isk = np.unravel_index(np.argmax(res.image), res.image.shape)
    print(f"argmax vortrace {ivt}, SKIRT {isk}")
    # orientation check against transposed / flipped alternatives
    alts = {"as is": res.image, "transposed": res.image.T,
            "flip x": res.image[::-1, :], "flip y": res.image[:, ::-1]}
    for label, alt in alts.items():
        rel = np.abs(alt - img_vt)[img_vt > 0] / img_vt[img_vt > 0]
        print(f"  orientation {label:>10}: max rel err {rel.max():.3e}")
    return stats, res


def synthetic_snapshot(n, box=100.0, seed=42):
    """Uniform random sites with unit density in a ``box`` kpc box."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * box
    rho = np.ones(n)
    mass = np.full(n, box**3 / n)
    return Snapshot(Path(f"synthetic_uniform_{n}"), pos, rho, mass, box, 1)


def scaling_test(n=1_000_000, thread_counts=(1, 8), npix=256, check=True,
                 measure_memory=True):
    """Time SKIRT on ``n`` uniform random points at several thread counts."""
    snap = synthetic_snapshot(n)
    b = snap.box_size
    geom = Geometry(center=(b / 2, b / 2), width=b, zrange=(0.0, b),
                    npix=npix)
    workdir = RUNS_DIR / f"scaling_uniform_{n}"
    results = {}
    img_vt = None
    if check:
        img_vt, t_vt = common.timed(vortrace_reference, snap, geom)
        print(f"vortrace reference: {t_vt:.2f} s (8 threads by default)")
    for k, nth in enumerate(thread_counts):
        res = project(snap, geom, nthreads=nth, workdir=workdir,
                      reuse_input=(k > 0), measure_memory=measure_memory)
        results[nth] = res
        rss = res.settings["peak_rss_bytes"]
        tag = ""
        if not rss:
            rss = res.settings["skirt_peak_memory_bytes"]
            tag = " (SKIRT-reported)"
        rss_gib = rss / 2**30 if rss else float("nan")
        timing_str = ", ".join(f"{a}: {v:.2f}"
                               for a, v in res.timings.items())
        print(f"nthreads={nth}: timings {{{timing_str}}}, "
              f"peak RSS {rss_gib:.2f} GiB{tag}")
        print("   stats:", res.settings["skirt_stats"])
        if img_vt is not None:
            st = common.compare_images(img_vt, res.image)
            print(f"   vs vortrace: max rel err {st['max_abs_rel_err']:.3e}, "
                  f"mass ratio {st['total_mass_ratio']:.8f}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--selftest", action="store_true",
                        help="validate against vortrace on the small galaxy "
                             "(default)")
    parser.add_argument("--scaling", action="store_true",
                        help="run the 1e6-point scaling test")
    parser.add_argument("--n", type=int, default=1_000_000,
                        help="number of points for --scaling")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8],
                        help="thread counts for --scaling")
    parser.add_argument("--nthreads", type=int, default=8,
                        help="threads for --selftest")
    parser.add_argument("--no-time", action="store_true",
                        help="do not wrap skirt in /usr/bin/time -l (it "
                             "fails in sandboxes); SKIRT's own peak-memory "
                             "figure is still recorded")
    args = parser.parse_args(argv)
    if not args.scaling:
        args.selftest = True
    if args.selftest:
        selftest(nthreads=args.nthreads, measure_memory=not args.no_time)
    if args.scaling:
        scaling_test(n=args.n, thread_counts=tuple(args.threads),
                     measure_memory=not args.no_time)


if __name__ == "__main__":
    main()
