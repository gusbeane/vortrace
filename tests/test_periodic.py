"""Tests for periodic boundary conditions (issue #23)."""

import numpy as np
import pytest
import warnings
from vortrace.vortrace import ProjectionCloud
from vortrace import io


class TestPeriodicNearestNeighbor:
    """Test that periodic wrapping finds the correct nearest neighbor."""

    def test_periodic_nearest_neighbor(self):
        """Query near one edge should find particle near the opposite edge."""
        # 10x10x10 periodic box, grid shifted right by 0.25 in x
        box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        coords = np.arange(0.5, 10.0, 1.0)  # 0.5, 1.5, ..., 9.5
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
        pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        pos[:, 0] += 0.25  # x-coords: 0.75, 1.75, ..., 9.75
        fields = np.arange(len(pos), dtype=np.float64)

        # Query at (0.1, 5.5, 5.5):
        #   nearest non-periodic: (0.75, 5.5, 5.5) at index 55, dist=0.65
        #   nearest periodic:     (9.75, 5.5, 5.5) at index 955, wrapped dist=0.35
        query = [0.1, 5.5, 5.5]

        cloud_per = ProjectionCloud(pos, fields, boundbox=box,
                                    periodic=True, filter=False)
        cloud_nper = ProjectionCloud(pos, fields, boundbox=box,
                                     periodic=False, filter=False)

        assert cloud_nper._cloud.queryTree(query) == 55
        assert cloud_per._cloud.queryTree(query) == 955


    def test_periodic_bruteforce_27_images(self):
        """Validate periodic NN against brute-force 27-image reference."""
        from scipy.spatial import KDTree

        Lx, Ly, Lz = 1.0, 0.5, 0.5
        box = [0.0, Lx, 0.0, Ly, 0.0, Lz]
        rng = np.random.default_rng(123)

        n_particles = 1000
        pos = np.column_stack([
            rng.uniform(0, Lx, n_particles),
            rng.uniform(0, Ly, n_particles),
            rng.uniform(0, Lz, n_particles),
        ])
        fields = np.arange(n_particles, dtype=np.float64)

        cloud = ProjectionCloud(pos, fields, boundbox=box,
                                periodic=True, filter=False)

        # Build scipy tree for brute-force reference
        ref_tree = KDTree(pos)

        shifts = np.array([
            (dx * Lx, dy * Ly, dz * Lz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ])

        n_queries = 1000
        queries = np.column_stack([
            rng.uniform(0, Lx, n_queries),
            rng.uniform(0, Ly, n_queries),
            rng.uniform(0, Lz, n_queries),
        ])

        for i in range(n_queries):
            q = queries[i]
            # Brute-force: query all 27 images, keep closest
            best_dist = np.inf
            best_idx = -1
            for s in shifts:
                dist, idx = ref_tree.query(q + s)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            result = cloud._cloud.queryTree(q.tolist())
            assert result == best_idx, (
                f"query {i}: got {result}, expected {best_idx}"
            )


class TestPeriodicFilterWarning:
    """Test warning when periodic=True with filter=True."""

    def test_periodic_filter_warning(self):
        """periodic=True, filter=True should warn about filter=False."""
        rng = np.random.default_rng(42)
        pos = rng.random((50, 3)) * 10.0
        fields = np.ones(50)
        box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProjectionCloud(pos, fields, boundbox=box,
                            periodic=True, filter=True)
            user_warnings = [x for x in w
                             if issubclass(x.category, UserWarning)]
            msgs = [str(x.message) for x in user_warnings]
            assert any("filter=False" in m for m in msgs)

    def test_periodic_no_warning_explicit_filter_false(self):
        """periodic=True, filter=False should emit no periodic warning."""
        rng = np.random.default_rng(42)
        pos = rng.random((50, 3)) * 10.0
        fields = np.ones(50)
        box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProjectionCloud(pos, fields, boundbox=box,
                            periodic=True, filter=False)
            user_warnings = [x for x in w
                             if issubclass(x.category, UserWarning)]
            msgs = [str(x.message) for x in user_warnings]
            # No warning about filter should appear
            assert not any("filter=False" in m for m in msgs)


class TestPeriodicNoFiltering:
    """Test that periodic=True loads all particles without filtering."""

    def test_periodic_no_filtering(self):
        """When periodic=True, orig_ids should be identity [0..N-1]."""
        rng = np.random.default_rng(42)
        n = 100
        pos = rng.random((n, 3)) * 10.0
        fields = np.ones(n)
        box = [2.0, 8.0, 2.0, 8.0, 2.0, 8.0]  # smaller than data extent

        cloud = ProjectionCloud(pos, fields, boundbox=box,
                                periodic=True, filter=False)

        # All particles should be loaded
        assert len(cloud.orig_ids) == n
        np.testing.assert_array_equal(cloud.orig_ids, np.arange(n))


class TestPeriodicIORoundtrip:
    """Test save/load of periodic clouds."""

    def test_npz_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        pos = rng.random((50, 3)) * 10.0
        fields = np.ones(50)
        box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]

        cloud = ProjectionCloud(pos, fields, boundbox=box,
                                periodic=True, filter=False)
        path = tmp_path / "periodic.npz"
        io.save_cloud(path, cloud)
        loaded = io.load_cloud(path)

        assert loaded.periodic is True
        assert loaded.filter is False
        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)

    def test_hdf5_roundtrip(self, tmp_path):
        pytest.importorskip("h5py")
        rng = np.random.default_rng(42)
        pos = rng.random((50, 3)) * 10.0
        fields = np.ones(50)
        box = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]

        cloud = ProjectionCloud(pos, fields, boundbox=box,
                                periodic=True, filter=False)
        path = tmp_path / "periodic.hdf5"
        io.save_cloud(path, cloud, fmt="hdf5")
        loaded = io.load_cloud(path)

        assert loaded.periodic is True
        assert loaded.filter is False
        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)


class TestNonperiodicBackwardCompat:
    """Test that existing non-periodic usage is unchanged."""

    def test_default_behavior(self):
        """Cloud without periodic/filter args should behave as before."""
        rng = np.random.default_rng(42)
        pos = rng.random((200, 3)) * 100.0
        fields = np.ones(200)
        box = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
        vol = np.ones(200) * 10.0

        cloud = ProjectionCloud(pos, fields, boundbox=box, vol=vol)

        assert cloud.periodic is False
        assert cloud.filter is True
        # Filtering should have happened (though all points are in-box here)
        assert cloud._cloud.get_periodic() is False
