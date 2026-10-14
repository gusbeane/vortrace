"""Shared helpers for the vortrace vs. COLT vs. SKIRT timing comparison.

Everything here is deliberately *not* part of the ``vortrace`` package.  The
comparison scripts read an Arepo snapshot in Gadget/Arepo code units (kpc,
1e10 Msun), hand the mesh-generating points and densities to each code, and
compare the resulting column-density images in code units
(1e10 Msun kpc^-2).

Conventions used by every wrapper (``colt_projection.py``,
``skirt_projection.py``, and the vortrace path in ``make_code_comparison.py``):

- ``pos`` is an (N, 3) float64 array of cell positions in kpc,
  ``rho`` the gas density in 1e10 Msun kpc^-3, ``mass`` the cell mass in
  1e10 Msun, ``box_size`` the periodic box side in kpc.  The box spans
  [0, box_size] along every axis.
- A projection is defined by ``center=(cx, cy)`` in kpc, a square ``width``
  in kpc, ``zrange=(zmin, zmax)`` in kpc, and ``npix`` pixels per side.  Rays
  travel along +z through the slab ``zmin <= z <= zmax``.  Pixel ``[i, j]`` of
  the returned image covers ``x in [cx - width/2 + i*dx, ... + (i+1)*dx]``
  and likewise for ``y`` with ``j`` (so images are indexed ``[x, y]``, as
  vortrace returns them).
- Every wrapper returns ``ProjectionResult`` with the image in code units.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXTERNAL_DIR = REPO_ROOT / "data" / "external"   # where COLT/SKIRT are built

# ---------------------------------------------------------------------------
# Units (Gadget/Arepo code units -> cgs / astronomical)
# ---------------------------------------------------------------------------
KPC_CM = 3.085677581467192e21          # 1 kpc in cm
PC_CM = KPC_CM / 1000.0
MSUN_G = 1.98841586e33                 # 1 Msun in g
UNIT_LENGTH_CM = KPC_CM                # code length unit
UNIT_MASS_G = 1e10 * MSUN_G            # code mass unit
UNIT_MASS_MSUN = 1e10
UNIT_DENSITY_CGS = UNIT_MASS_G / UNIT_LENGTH_CM**3          # g cm^-3
UNIT_COLDENS_CGS = UNIT_MASS_G / UNIT_LENGTH_CM**2          # g cm^-2
UNIT_DENSITY_MSUN_PC3 = UNIT_MASS_MSUN / 1000.0**3          # Msun pc^-3
UNIT_COLDENS_MSUN_PC2 = UNIT_MASS_MSUN / 1000.0**2          # Msun pc^-2

LENGTH_UNIT = "kpc"
COL_DENSITY_UNIT = r"$10^{10}\;M_\odot\,\mathrm{kpc}^{-2}$"


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

def snapshot_parts(path):
    """All parts of a (possibly multi-file) Arepo snapshot, sorted."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
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


@dataclasses.dataclass
class Snapshot:
    """Gas cells of an Arepo snapshot in code units."""
    path: Path
    pos: np.ndarray        # (N, 3) kpc
    rho: np.ndarray        # (N,) 1e10 Msun kpc^-3
    mass: np.ndarray       # (N,) 1e10 Msun
    box_size: float        # kpc
    nparts: int

    @property
    def vol(self):
        """Cell volumes in kpc^3."""
        return self.mass / self.rho

    @property
    def ncells(self):
        return int(len(self.rho))

    def subbox(self, lo, hi):
        """Cells with lo <= pos < hi (component-wise), as a new Snapshot.

        Used to hand the mesh-based codes only the cells near the projected
        region when the full box is too large for them.  The margin must be
        wide enough that the Voronoi mesh inside the projected region is
        unaffected by the cut; the driver records the margin used.
        """
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        keep = np.all((self.pos >= lo) & (self.pos < hi), axis=1)
        return Snapshot(self.path, self.pos[keep], self.rho[keep],
                        self.mass[keep], self.box_size, self.nparts)


def separate_duplicates(snap, offset=1e-6):
    """Displace exactly coincident cell positions so every generator is
    distinct.

    Snapshots that store positions in single precision can contain pairs of
    cells with bit-identical coordinates in dense regions.  A Voronoi mesh
    is undefined for coincident generators (CGAL refuses them, SKIRT drops
    one of the two), so the second member of every duplicate group is moved
    by ``offset`` (kpc; 1e-6 kpc = 1 mpc, far below any cell size) along a
    fixed direction.  The same perturbed positions are handed to every code.
    Returns the number of cells moved; ``snap.pos`` is modified in place.
    """
    _, first, counts = np.unique(snap.pos, axis=0, return_index=True,
                                 return_counts=True)
    moved = 0
    for i in first[counts > 1]:
        same = np.where(np.all(snap.pos == snap.pos[i], axis=1))[0]
        for k, j in enumerate(same[1:], start=1):
            snap.pos[j] += offset * k * np.array([1.0, 0.5, 0.25]) / 1.146
            moved += 1
    return moved


def load_snapshot(path):
    """Load gas positions, density and mass from an Arepo HDF5 snapshot."""
    pos, rho, mass = [], [], []
    parts = snapshot_parts(path)
    for part in parts:
        with h5py.File(part, "r") as f:
            pos.append(np.asarray(f["PartType0/Coordinates"], dtype=float))
            rho.append(np.asarray(f["PartType0/Density"], dtype=float))
            mass.append(np.asarray(f["PartType0/Masses"], dtype=float))
    with h5py.File(parts[0], "r") as f:
        box_size = float(f["Header"].attrs["BoxSize"])
    pos, rho, mass = map(np.concatenate, (pos, rho, mass))
    return Snapshot(Path(path), pos, rho, mass, box_size, len(parts))


# ---------------------------------------------------------------------------
# Projection geometry and results
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Geometry:
    """A square column-density image through a z slab (code units, kpc)."""
    center: tuple          # (cx, cy)
    width: float
    zrange: tuple          # (zmin, zmax)
    npix: int

    @property
    def extent(self):
        """[xmin, xmax, ymin, ymax] in kpc."""
        cx, cy = self.center
        h = self.width / 2
        return [cx - h, cx + h, cy - h, cy + h]

    @property
    def depth(self):
        return self.zrange[1] - self.zrange[0]

    @property
    def zmid(self):
        return 0.5 * (self.zrange[0] + self.zrange[1])

    @property
    def pixel_width(self):
        return self.width / self.npix

    def to_dict(self):
        return {"center": list(self.center), "width": self.width,
                "zrange": list(self.zrange), "npix": self.npix}


@dataclasses.dataclass
class ProjectionResult:
    """Column-density image from one code plus timings and provenance.

    ``image`` is (npix, npix), indexed [x, y], in 1e10 Msun kpc^-2.
    ``timings`` is a flat dict of wall-clock seconds.  Every wrapper fills in
    at least ``total_s`` (everything from handing over the data to having the
    image in memory, including writing input files and reading output) and
    ``run_s`` (the external executable's wall-clock time).  Where the code
    reports a separate mesh-construction time, add ``mesh_s``.
    ``settings`` records the configuration handed to the code (file names,
    config keys, version/commit) so the run can be reproduced.
    """
    code: str
    image: np.ndarray
    timings: dict
    settings: dict
    nthreads: int


def timed(func, *args, **kwargs):
    """Run ``func`` and return (result, wall-clock seconds)."""
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    return out, time.perf_counter() - t0


def _peak_rss_from_log(text):
    """Peak resident set size in bytes from ``/usr/bin/time -l`` (macOS) or
    GNU ``time -v`` output, or None if absent."""
    rss = None
    for line in text.splitlines():
        low = line.lower()
        if "maximum resident set size" in low:
            token = low.split("maximum resident set size")[0].strip()
            if not token:                       # GNU time: "...(kbytes): N"
                token = line.split(":")[-1].strip()
                scale = 1024
            else:                               # macOS: "N  maximum resident set size"
                scale = 1
            try:
                rss = int(float(token.split()[0])) * scale
            except ValueError:
                pass
    return rss


_TIME_L_WORKS = None


def _time_l_works():
    """Whether ``/usr/bin/time -l`` (macOS) runs here; it fails inside
    sandboxes that forbid the sysctl it needs.  Probed once."""
    global _TIME_L_WORKS  # pylint: disable=global-statement
    if _TIME_L_WORKS is None:
        _TIME_L_WORKS = False
        if platform.system() == "Darwin" and os.path.exists("/usr/bin/time"):
            try:
                proc = subprocess.run(["/usr/bin/time", "-l", "true"],
                                      capture_output=True, check=False)
                _TIME_L_WORKS = (proc.returncode == 0 and
                                 b"maximum resident set size" in proc.stderr)
            except OSError:
                pass
    return _TIME_L_WORKS


def run_logged(cmd, *, cwd, log_path, env=None, timeout=None,
               measure_rss=True):
    """Run an external command, teeing stdout/stderr to ``log_path``.

    Returns (returncode, wall-clock seconds).  Raises ``RuntimeError`` with
    the tail of the log on failure.  When ``measure_rss`` is true and
    ``/usr/bin/time`` exists, the command runs under ``/usr/bin/time -l``
    (macOS) so its peak resident set size is appended to the log; it is
    then available as ``run_logged.last_peak_rss_bytes`` (None if unknown).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    cmd = [str(c) for c in cmd]
    use_time = measure_rss and _time_l_works()
    if use_time:
        cmd = ["/usr/bin/time", "-l"] + cmd
    run_logged.last_peak_rss_bytes = None
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        proc = subprocess.run(cmd, cwd=str(cwd),
                              stdout=log, stderr=subprocess.STDOUT,
                              env=full_env, timeout=timeout, check=False)
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    elapsed = time.perf_counter() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rss = _peak_rss_from_log(text) if use_time else None
    if rss is None and after > before:
        # ru_maxrss of children is the max over all children so far; it
        # is only attributable to this command when it grew.  Bytes on
        # macOS, kilobytes on Linux.
        rss = int(after) * (1 if platform.system() == "Darwin" else 1024)
    run_logged.last_peak_rss_bytes = rss
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed with exit code {proc.returncode}; "
            f"log at {log_path}:\n{text[-4000:]}")
    return proc.returncode, elapsed


run_logged.last_peak_rss_bytes = None


def machine_info():
    """Platform description for the stats file."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def compare_images(reference, other):
    """Summary statistics of ``other`` relative to ``reference``.

    Both images must have the same shape.  Pixels where the reference is
    zero are excluded from the relative-error statistics.
    """
    reference = np.asarray(reference, dtype=float)
    other = np.asarray(other, dtype=float)
    if reference.shape != other.shape:
        raise ValueError(f"shape mismatch {reference.shape} vs {other.shape}")
    nonzero = reference > 0
    rel = (other[nonzero] - reference[nonzero]) / reference[nonzero]
    pct = (1, 5, 50, 95, 99)
    return {
        "total_mass_ratio": float(other.sum() / reference.sum()),
        "peak_ratio": float(other.max() / reference.max()),
        "max_abs_rel_err": float(np.abs(rel).max()) if rel.size else None,
        "rel_err_percentiles": {str(p): float(v)
                                for p, v in zip(pct, np.percentile(rel, pct))},
        "rms_rel_err": float(np.sqrt(np.mean(rel**2))) if rel.size else None,
        "zero_pixels_reference": int((~nonzero).sum()),
    }
