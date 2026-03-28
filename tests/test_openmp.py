"""Tests for OpenMP support.

Verifies that projections produce identical results regardless of the
number of OpenMP threads, and provides an optional performance benchmark.

Default tests (fast):
    pytest tests/test_openmp.py -v

Full scaling benchmark (slow, needs -s for output):
    pytest -m benchmark tests/test_openmp.py -v -s
"""
import os
import subprocess
import sys
import tempfile
import time

import h5py as h5
import numpy as np
import pytest

from vortrace import vortrace as vt


_SUBPROCESS_SCRIPT = """\
import os
os.environ['OMP_NUM_THREADS'] = '{nthreads}'
import time
import h5py as h5
import numpy as np
from vortrace import vortrace as vt

f = h5.File('{snap}', mode='r')
pos = np.array(f['PartType0']['Coordinates'])
dens = np.array(f['PartType0']['Density'])
box_size = f['Parameters'].attrs['BoxSize']
f.close()

length = 75.0
pc = vt.ProjectionCloud(pos, dens,
    boundbox=[0., box_size, 0., box_size, 0., box_size])
extent = [box_size / 2. - length / 2., box_size / 2. + length / 2.]
bounds = [0., box_size]

t0 = time.perf_counter()
dat = pc.grid_projection(extent, {npix}, bounds, None)
elapsed = time.perf_counter() - t0

if '{outfile}':
    np.save('{outfile}', dat)
print(f'ELAPSED:{{elapsed:.6f}}')
"""


def _run_projection_subprocess(snapname, npix, nthreads, outfile=''):
    """Run a grid projection in a subprocess with a clean OMP environment.

    Returns the elapsed wall-clock time in seconds.
    """
    script = _SUBPROCESS_SCRIPT.format(
        nthreads=nthreads, snap=snapname, npix=npix, outfile=outfile)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(nthreads)
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Subprocess failed (nthreads={nthreads}):\n"
        f"stderr: {result.stderr}")
    for line in result.stdout.splitlines():
        if line.startswith('ELAPSED:'):
            return float(line.split(':')[1])
    return None


def _read_arepo_snap(snapname):
    """Read particle data from an AREPO snapshot."""
    with h5.File(snapname, mode='r') as f:
        pos = np.array(f['PartType0']['Coordinates'])
        dens = np.array(f['PartType0']['Density'])
        box_size = f['Parameters'].attrs['BoxSize']
    return pos, dens, box_size


def _make_projection(pos, dens, box_size, npix=128):
    """Run a grid projection and return the result array."""
    length = 75.0
    pc = vt.ProjectionCloud(
        pos, dens,
        boundbox=[0., box_size, 0., box_size, 0., box_size])
    extent = [box_size / 2. - length / 2., box_size / 2. + length / 2.]
    bounds = [0., box_size]
    return pc.grid_projection(extent, npix, bounds, None)


class TestOpenMPCorrectness:
    """Verify that OpenMP parallelisation does not change results.

    These run by default and should finish in a few seconds.
    """

    snapname = 'tests/test_data/galaxy_interaction.hdf5'

    def test_single_vs_multi_thread(self):
        """Results must be identical with 1 thread and 2 threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = os.path.join(tmpdir, 'result_1thread.npy')
            out2 = os.path.join(tmpdir, 'result_2thread.npy')

            # Launch both subprocesses concurrently (small 32x32 grid)
            configs = [('1', out1), ('2', out2)]
            procs = []
            for nthreads, outfile in configs:
                script = _SUBPROCESS_SCRIPT.format(
                    nthreads=nthreads, snap=self.snapname,
                    npix=32, outfile=outfile)
                env = os.environ.copy()
                env['OMP_NUM_THREADS'] = nthreads
                procs.append(subprocess.Popen(
                    [sys.executable, '-c', script],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

            for proc, (nthreads, _) in zip(procs, configs):
                stdout, stderr = proc.communicate()
                assert proc.returncode == 0, (
                    f"Subprocess failed (nthreads={nthreads}):\n"
                    f"stderr: {stderr.decode()}")

            dat1 = np.load(out1)
            dat2 = np.load(out2)

        np.testing.assert_array_equal(
            dat1, dat2,
            err_msg="Results differ between 1 thread and 2 threads")

    def test_openmp_matches_reference(self):
        """Multi-threaded result must match the stored reference data."""
        pos, dens, box_size = _read_arepo_snap(self.snapname)
        dat = _make_projection(pos, dens, box_size, npix=128)
        ref_dat = np.load('tests/test_data/galaxy_interaction-proj.npy')
        np.testing.assert_array_almost_equal(dat, ref_dat, decimal=14)

    def test_openmp_gives_speedup(self):
        """Verify that 2 threads is faster than 1 thread on a small grid."""
        npix = 64
        t1 = _run_projection_subprocess(self.snapname, npix, nthreads=1)
        t2 = _run_projection_subprocess(self.snapname, npix, nthreads=2)
        assert t1 is not None and t2 is not None
        speedup = t1 / t2
        assert speedup > 1.3, (
            f"Expected >1.3x speedup with 2 threads, got {speedup:.2f}x — "
            f"OpenMP may not be working (t1={t1:.2f}s, t2={t2:.2f}s)")


def _thread_counts(max_nt=16):
    """Return thread counts to benchmark, capped by available cores."""
    ncpu = os.cpu_count() or 4
    counts = [1, 2, 4, 8, 16]
    return [n for n in counts if n <= min(max_nt, ncpu)]


def _print_scaling_table(npix, timings, threads):
    """Print a formatted scaling table."""
    nrays = npix * npix
    t1 = timings[1]

    hdr_times = ''.join(f' {"t="+str(n):>10}' for n in threads)
    hdr_speedups = ''.join(f' {"s("+str(n)+")":>8}' for n in threads[1:])
    print(f"\n--- OpenMP Projection Benchmark ({npix}x{npix} grid, "
          f"{nrays} rays) ---")
    print(f"{'':>8}{hdr_times}{hdr_speedups}")
    print("-" * (9 + 11 * len(threads) + 9 * len(threads[1:])))

    vals = ''.join(f' {timings[n]:10.3f}s' for n in threads)
    speedups = ''.join(f' {t1/timings[n]:8.2f}x' for n in threads[1:])
    print(f"{'time':>8}{vals}{speedups}")

    effs = ''.join(f' {t1/timings[n]/n*100:9.1f}%' for n in threads)
    print(f"{'eff':>8}{effs}")
    print(f"\nPer-ray: {t1/nrays*1e6:.0f} us (1 thread)")
    print("--------------------------------------------------")


class TestOpenMPPerformance:
    """Full scaling benchmark across many thread counts and grid sizes.

    Skipped by default — run with:
        pytest -m benchmark tests/test_openmp.py -v -s
    """

    snapname = 'tests/test_data/galaxy_interaction.hdf5'

    @pytest.mark.benchmark
    def test_projection_scaling(self):
        """Measure wall-clock scaling from 1 to 16 threads at 128x128."""
        npix = 128
        threads = _thread_counts(max_nt=16)

        timings = {}
        for nt in threads:
            elapsed = _run_projection_subprocess(self.snapname, npix, nt)
            assert elapsed is not None, f"No timing returned for {nt} threads"
            timings[nt] = elapsed

        _print_scaling_table(npix, timings, threads)

        max_nt = max(threads)
        if max_nt > 1:
            speedup = timings[1] / timings[max_nt]
            assert speedup > 1.5, (
                f"Expected >1.5x speedup at {max_nt} threads, "
                f"got {speedup:.2f}x — OpenMP may not be working")

    @pytest.mark.benchmark
    def test_projection_scaling_large(self):
        """Measure wall-clock scaling from 1 to 16 threads at 256x256."""
        npix = 256
        threads = _thread_counts(max_nt=16)

        timings = {}
        for nt in threads:
            elapsed = _run_projection_subprocess(self.snapname, npix, nt)
            assert elapsed is not None, f"No timing returned for {nt} threads"
            timings[nt] = elapsed

        _print_scaling_table(npix, timings, threads)

        max_nt = max(threads)
        if max_nt > 1:
            speedup = timings[1] / timings[max_nt]
            assert speedup > 1.5, (
                f"Expected >1.5x speedup at {max_nt} threads, "
                f"got {speedup:.2f}x — OpenMP may not be working")
