"""Tests for OpenMP support.

Verifies that projections produce identical results regardless of the
number of OpenMP threads, and provides a simple performance benchmark.
"""
import os
import subprocess
import sys
import time

import h5py as h5
import numpy as np
import pytest

from vortrace import vortrace as vt


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
    """Verify that OpenMP parallelisation does not change results."""

    snapname = 'tests/test_data/galaxy_interaction.hdf5'

    def test_single_vs_multi_thread(self):
        """Results must be identical with 1 thread and multiple threads."""
        pos, dens, box_size = _read_arepo_snap(self.snapname)

        # Run with 1 thread in a subprocess to get a clean OMP environment
        script = (
            "import os; os.environ['OMP_NUM_THREADS'] = '{nthreads}';"
            "import numpy as np;"
            "import h5py as h5;"
            "from vortrace import vortrace as vt;"
            "f = h5.File('{snap}', mode='r');"
            "pos = np.array(f['PartType0']['Coordinates']);"
            "dens = np.array(f['PartType0']['Density']);"
            "box_size = f['Parameters'].attrs['BoxSize'];"
            "f.close();"
            "length = 75.0;"
            "pc = vt.ProjectionCloud(pos, dens, "
            "  boundbox=[0., box_size, 0., box_size, 0., box_size]);"
            "extent = [box_size/2.-length/2., box_size/2.+length/2.];"
            "bounds = [0., box_size];"
            "dat = pc.grid_projection(extent, 128, bounds, None);"
            "np.save('{outfile}', dat)"
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = os.path.join(tmpdir, 'result_1thread.npy')
            out_multi = os.path.join(tmpdir, 'result_multi.npy')

            for nthreads, outfile in [('1', out1), ('4', out_multi)]:
                cmd = script.format(
                    nthreads=nthreads, snap=self.snapname, outfile=outfile)
                env = os.environ.copy()
                env['OMP_NUM_THREADS'] = nthreads
                result = subprocess.run(
                    [sys.executable, '-c', cmd],
                    env=env, capture_output=True, text=True)
                assert result.returncode == 0, (
                    f"Subprocess failed (nthreads={nthreads}):\n"
                    f"stderr: {result.stderr}")

            dat1 = np.load(out1)
            dat_multi = np.load(out_multi)

        np.testing.assert_array_equal(
            dat1, dat_multi,
            err_msg="Results differ between 1 thread and 4 threads")

    def test_openmp_matches_reference(self):
        """Multi-threaded result must match the stored reference data."""
        pos, dens, box_size = _read_arepo_snap(self.snapname)
        dat = _make_projection(pos, dens, box_size, npix=128)
        ref_dat = np.load('tests/test_data/galaxy_interaction-proj.npy')
        np.testing.assert_array_almost_equal(dat, ref_dat, decimal=14)


class TestOpenMPPerformance:
    """Benchmark projection performance with different thread counts.

    These tests are skipped by default — run with:
        pytest -m benchmark tests/test_openmp.py -v -s
    """

    snapname = 'tests/test_data/galaxy_interaction.hdf5'

    @pytest.mark.benchmark
    def test_projection_scaling(self):
        """Measure wall-clock time with 1, 2, and 4 OpenMP threads."""
        script = (
            "import os; os.environ['OMP_NUM_THREADS'] = '{nthreads}';"
            "import time, numpy as np, h5py as h5;"
            "from vortrace import vortrace as vt;"
            "f = h5.File('{snap}', mode='r');"
            "pos = np.array(f['PartType0']['Coordinates']);"
            "dens = np.array(f['PartType0']['Density']);"
            "box_size = f['Parameters'].attrs['BoxSize'];"
            "f.close();"
            "length = 75.0;"
            "pc = vt.ProjectionCloud(pos, dens, "
            "  boundbox=[0., box_size, 0., box_size, 0., box_size]);"
            "extent = [box_size/2.-length/2., box_size/2.+length/2.];"
            "bounds = [0., box_size];"
            "npix = 256;"
            "t0 = time.perf_counter();"
            "dat = pc.grid_projection(extent, npix, bounds, None);"
            "elapsed = time.perf_counter() - t0;"
            "print(f'ELAPSED:{{elapsed:.4f}}')"
        )

        timings = {}
        for nthreads in ['1', '2', '4']:
            cmd = script.format(nthreads=nthreads, snap=self.snapname)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = nthreads
            result = subprocess.run(
                [sys.executable, '-c', cmd],
                env=env, capture_output=True, text=True)
            assert result.returncode == 0, (
                f"Subprocess failed (nthreads={nthreads}):\n"
                f"stderr: {result.stderr}")

            for line in result.stdout.splitlines():
                if line.startswith('ELAPSED:'):
                    elapsed = float(line.split(':')[1])
                    timings[int(nthreads)] = elapsed
                    break

        # Print results
        print("\n--- OpenMP Projection Benchmark (256x256 grid) ---")
        for nt, t in sorted(timings.items()):
            speedup = timings[1] / t if 1 in timings else float('nan')
            print(f"  {nt} thread(s): {t:.4f}s  (speedup: {speedup:.2f}x)")
        print("--------------------------------------------------")

        # Basic sanity: multi-threaded should not be dramatically slower
        if 1 in timings and 4 in timings:
            assert timings[4] < timings[1] * 2.0, (
                "4-thread run is more than 2x slower than single-threaded — "
                "OpenMP may not be working correctly")
