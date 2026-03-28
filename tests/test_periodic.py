"""Tests for periodic boundary conditions."""

import numpy as np
import pytest
import warnings
from vortrace import vortrace as vt
from vortrace.Cvortrace import PointCloud


def _make_grid_cloud(box_size, spacing, field_val=1.0, periodic=False):
    """Create a uniform cubic grid of particles filling [0, box_size]^3."""
    coords = np.arange(spacing / 2.0, box_size, spacing)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    fields = np.full(len(pos), field_val)
    boundbox = [0.0, box_size, 0.0, box_size, 0.0, box_size]
    return vt.ProjectionCloud(pos, fields, boundbox=boundbox,
                              periodic=periodic)


def _brute_force_periodic_knn(pos, query, box_size, k):
    """Brute-force k-NN with periodic wrapping over 27 images.

    Returns (ids, dists2) sorted by ascending squared distance,
    with unique particle IDs only (minimum distance kept).
    """
    L = np.array([box_size, box_size, box_size])
    best = {}  # particle_id -> min squared distance
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                shift = np.array([ix, iy, iz]) * L
                shifted_query = query + shift
                dists2 = np.sum((pos - shifted_query) ** 2, axis=1)
                for pid, d2 in enumerate(dists2):
                    if pid not in best or d2 < best[pid]:
                        best[pid] = d2
    # Sort by distance and take top k
    sorted_items = sorted(best.items(), key=lambda x: x[1])
    ids = np.array([item[0] for item in sorted_items[:k]])
    dists2 = np.array([item[1] for item in sorted_items[:k]])
    return ids, dists2


class TestPeriodicNearestNeighbor:
    """Test periodic nearest-neighbor queries at the C++ level."""

    def test_1nn_near_boundary(self):
        """Particle near x=0 should be found from query near x=L."""
        box = 10.0
        # A few particles spread out, one very close to x=0
        pos = np.array([
            [0.1, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [8.0, 5.0, 5.0],
        ])
        fields = np.array([1.0, 2.0, 3.0])
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        cloud = PointCloud()
        cloud.loadPoints(pos, fields, boundbox, periodic=True)
        cloud.buildTree()

        # Query near x=L: periodic distance to particle 0 is 0.2
        query = np.array([9.9, 5.0, 5.0])
        nn = cloud.queryTree(query)
        assert nn == 0, f"Expected particle 0 (at 0.1), got {nn}"

    def test_1nn_interior_unchanged(self):
        """Interior queries give the same result periodic or not."""
        box = 10.0
        rng = np.random.RandomState(42)
        pos = rng.uniform(0.5, 9.5, size=(100, 3))
        fields = np.ones(100)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        cloud_p = PointCloud()
        cloud_p.loadPoints(pos, fields, boundbox, periodic=True)
        cloud_p.buildTree()

        cloud_np = PointCloud()
        cloud_np.loadPoints(pos, fields, boundbox)
        cloud_np.buildTree()

        # Query well interior — both should agree
        query = np.array([5.0, 5.0, 5.0])
        assert cloud_p.queryTree(query) == cloud_np.queryTree(query)


class TestPeriodicKNNDeduplication:
    """Test that k-NN returns unique particle IDs with correct distances."""

    def test_knn_near_corner(self):
        """Query near a corner where many images overlap.

        Particles near the corner should not appear as duplicates.
        Verify against brute-force 27-image reference.
        """
        box = 10.0
        rng = np.random.RandomState(123)
        pos = rng.uniform(0.0, box, size=(200, 3))
        fields = np.ones(200)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        cloud = PointCloud()
        cloud.loadPoints(pos, fields, boundbox, periodic=True)
        cloud.buildTree()

        query = np.array([0.1, 0.1, 0.1])
        k = 5

        ref_ids, ref_dists2 = _brute_force_periodic_knn(pos, query, box, k)

        ids, dists2 = cloud.queryTree(query, k=k)

        # (a) All returned IDs are unique
        assert len(set(ids)) == k, \
            f"Duplicate IDs in k-NN result: {ids}"

        # (b) IDs match brute-force reference
        np.testing.assert_array_equal(ids, ref_ids)

        # (c) Distances match brute-force reference
        np.testing.assert_allclose(dists2, ref_dists2, rtol=1e-12)

    def test_knn_unique_ids_via_ray(self):
        """Fire a ray near a periodic boundary and verify segment cell IDs
        are all valid (no duplicates from periodic image confusion)."""
        box = 5.0
        spacing = 0.5
        pc = _make_grid_cloud(box, spacing, field_val=1.0, periodic=True)

        # Ray along x-axis very close to x=0 boundary
        start = np.array([0.01, 2.5, 2.5])
        end = np.array([4.99, 2.5, 2.5])
        dens, cell_ids, _, ds = pc.single_projection(start, end)

        # All cell IDs should be unique consecutive cells
        assert len(cell_ids) == len(set(cell_ids)), \
            "Duplicate cell IDs in ray segments"

        # Integral should be close to ray length (uniform density=1)
        ray_length = np.linalg.norm(end - start)
        np.testing.assert_allclose(dens, ray_length, rtol=1e-10)


class TestPeriodicNoFiltering:
    """When periodic=True, all particles should be loaded."""

    def test_all_particles_loaded(self):
        """No spatial filtering in periodic mode — all particles are kept."""
        box = 10.0
        rng = np.random.RandomState(99)
        pos = rng.uniform(0.0, box, size=(500, 3))
        fields = np.ones(500)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        cloud = PointCloud()
        cloud.loadPoints(pos, fields, boundbox, periodic=True)
        cloud.buildTree()

        assert cloud.get_npart() == 500, \
            f"Expected 500 particles, got {cloud.get_npart()}"


class TestPeriodicRay:
    """Test ray integration with periodic boundaries."""

    def test_ray_uniform_density(self):
        """Ray through uniform-density periodic box should give
        integral = ray_length * density."""
        box = 5.0
        spacing = 0.5
        density = 2.5
        pc = _make_grid_cloud(box, spacing, field_val=density,
                              periodic=True)

        start = np.array([0.5, 2.5, 2.5])
        end = np.array([4.5, 2.5, 2.5])
        dens, _, _, _ = pc.single_projection(start, end)

        ray_length = np.linalg.norm(end - start)
        np.testing.assert_allclose(dens, ray_length * density, rtol=1e-10)

    def test_ray_near_boundary(self):
        """Ray starting very near x=0 should integrate correctly."""
        box = 5.0
        spacing = 0.5
        pc = _make_grid_cloud(box, spacing, field_val=1.0, periodic=True)

        start = np.array([0.01, 2.5, 2.5])
        end = np.array([4.99, 2.5, 2.5])
        dens, _, _, _ = pc.single_projection(start, end)

        ray_length = np.linalg.norm(end - start)
        np.testing.assert_allclose(dens, ray_length, rtol=1e-10)


class TestPeriodicValidation:
    """Test periodic-mode guards for out-of-bounds and small extent."""

    def test_error_if_particles_outside_box(self):
        """Particles outside the bounding box should raise an error."""
        box = 10.0
        pos = np.array([[5.0, 5.0, 5.0],
                         [5.0, 5.0, 11.0]])  # outside z
        fields = np.ones(2)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        with pytest.raises(ValueError, match="outside the bounding box"):
            vt.ProjectionCloud(pos, fields, boundbox=boundbox, periodic=True)

    def test_error_if_particles_below_box(self):
        """Particles below the bounding box lower bound should raise."""
        box = 10.0
        pos = np.array([[5.0, 5.0, 5.0],
                         [-0.1, 5.0, 5.0]])  # outside x (below)
        fields = np.ones(2)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        with pytest.raises(ValueError, match="outside the bounding box"):
            vt.ProjectionCloud(pos, fields, boundbox=boundbox, periodic=True)

    def test_no_error_if_particles_inside_box(self):
        """All particles inside the box should not raise."""
        box = 10.0
        rng = np.random.default_rng(42)
        pos = rng.uniform(0.1, 9.9, size=(50, 3))
        fields = np.ones(50)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        # Should not raise
        vt.ProjectionCloud(pos, fields, boundbox=boundbox, periodic=True)

    def test_no_error_for_non_periodic_outside_box(self):
        """Non-periodic mode should not error for particles outside box."""
        box = 10.0
        pos = np.array([[5.0, 5.0, 5.0],
                         [5.0, 5.0, 15.0]])  # outside z
        fields = np.ones(2)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        # Should not raise — non-periodic just filters
        vt.ProjectionCloud(pos, fields, boundbox=boundbox, periodic=False)

    def test_warning_if_extent_small_relative_to_box(self):
        """Warn if particles span < 60% of the box in any dimension."""
        box = 100.0
        # Particles clustered in a small region (extent ~10 in each dim)
        rng = np.random.default_rng(42)
        pos = rng.uniform(45.0, 55.0, size=(50, 3))
        fields = np.ones(50)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vt.ProjectionCloud(pos, fields, boundbox=boundbox, periodic=True)
            extent_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "60%" in str(x.message)
            ]
            assert len(extent_warnings) >= 1

    def test_no_warning_if_extent_fills_box(self):
        """No warning when particles span most of the box."""
        box = 10.0
        rng = np.random.default_rng(42)
        pos = rng.uniform(0.1, 9.9, size=(200, 3))
        fields = np.ones(200)
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vt.ProjectionCloud(pos, fields, boundbox=boundbox, periodic=True)
            extent_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "60%" in str(x.message)
            ]
            assert len(extent_warnings) == 0


class TestPeriodicBackwardCompat:
    """Non-periodic behavior must be unchanged."""

    def test_default_is_not_periodic(self):
        """Default periodic=False."""
        box = 10.0
        pos = np.array([[5.0, 5.0, 5.0]])
        fields = np.array([1.0])
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        cloud = PointCloud()
        cloud.loadPoints(pos, fields, boundbox)
        assert cloud.get_periodic() is False


class TestPeriodicIORoundtrip:
    """Test save/load preserves the periodic flag."""

    def test_npz_roundtrip(self, tmp_path):
        box = 5.0
        spacing = 1.0
        pc = _make_grid_cloud(box, spacing, periodic=True)
        path = str(tmp_path / "cloud.npz")
        pc.save(path, fmt="npz")

        loaded = vt.ProjectionCloud.load(path)
        assert loaded.periodic is True

    def test_legacy_load(self, tmp_path):
        """Files saved without periodic flag should load as False."""
        box = 5.0
        pos = np.array([[2.5, 2.5, 2.5]])
        fields = np.array([1.0])
        boundbox = [0.0, box, 0.0, box, 0.0, box]

        # Save manually without periodic key
        path = str(tmp_path / "legacy.npz")
        np.savez(path, pos=pos, fields=fields,
                 boundbox=np.array(boundbox))

        loaded = vt.ProjectionCloud.load(path)
        assert loaded.periodic is False
