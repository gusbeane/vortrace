"""Tests for traced_projection (batch and single-ray segment queries)."""

import numpy as np
import pytest
from vortrace import vortrace as vt


def make_cubic_lattice(n=5):
    """Create an n x n x n cubic lattice point cloud."""
    coords = np.arange(n, dtype=np.float64)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    fields = np.ones(len(pos), dtype=np.float64)
    vol = np.ones(len(pos), dtype=np.float64)
    boundbox = [-0.5, n - 0.5, -0.5, n - 0.5, -0.5, n - 0.5]
    return pos, fields, vol, boundbox


@pytest.fixture
def lattice_cloud():
    pos, fields, vol, bb = make_cubic_lattice()
    return vt.ProjectionCloud(pos, fields, boundbox=bb, vol=vol)


@pytest.fixture
def lattice_cloud_multifield():
    pos, fields, vol, bb = make_cubic_lattice()
    fields_2d = np.column_stack([fields, 2 * fields])
    return vt.ProjectionCloud(pos, fields_2d, boundbox=bb, vol=vol)


class TestSingleRay:
    """traced_projection with a single ray should match legacy behavior."""

    def test_single_1d_input(self, lattice_cloud):
        start = np.array([2.0, 2.0, -0.5])
        end = np.array([2.0, 2.0, 4.5])
        val, cell_ids, s_vals, ds_vals = lattice_cloud.traced_projection(
            start, end)
        ray_length = np.linalg.norm(end - start)
        np.testing.assert_allclose(val, ray_length, rtol=1e-6)
        np.testing.assert_allclose(np.sum(ds_vals), ray_length, rtol=1e-12)

    def test_single_2d_input(self, lattice_cloud):
        start = np.array([[2.0, 2.0, -0.5]])
        end = np.array([[2.0, 2.0, 4.5]])
        val, cell_ids, s_vals, ds_vals = lattice_cloud.traced_projection(
            start, end)
        ray_length = np.linalg.norm(end - start)
        np.testing.assert_allclose(val, ray_length, rtol=1e-6)

    def test_single_returns_scalar_for_one_field(self, lattice_cloud):
        start = np.array([2.0, 2.0, -0.5])
        end = np.array([2.0, 2.0, 4.5])
        val, _, _, _ = lattice_cloud.traced_projection(start, end)
        assert np.isscalar(val) or val.ndim == 0

    def test_return_midpoint_false(self, lattice_cloud):
        start = np.array([2.0, 2.0, -0.5])
        end = np.array([2.0, 2.0, 4.5])
        _, _, s_mid, _ = lattice_cloud.traced_projection(start, end,
                                                         return_midpoint=True)
        _, _, s_enter, _ = lattice_cloud.traced_projection(
            start, end, return_midpoint=False)
        # Midpoints should be larger than entry points for all but the last
        assert len(s_mid) == len(s_enter)
        assert not np.array_equal(s_mid, s_enter)


class TestBatch:
    """traced_projection with multiple rays."""

    def _make_rays(self, n=8):
        starts = np.zeros((n, 3), dtype=np.float64)
        ends = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            starts[i] = [2.0, 2.0, -0.5]
            ends[i] = [2.0, 2.0, 4.5]
        # Vary a couple of rays.
        starts[1] = [1.0, 2.0, -0.5]
        ends[1] = [1.0, 2.0, 4.5]
        starts[2] = [3.0, 3.0, -0.5]
        ends[2] = [3.0, 3.0, 4.5]
        return starts, ends

    def test_batch_list_format(self, lattice_cloud):
        starts, ends = self._make_rays()
        values, cid_list, sv_list, ds_list = lattice_cloud.traced_projection(
            starts, ends)
        n = len(starts)
        assert values.shape[0] == n
        assert len(cid_list) == n
        assert len(sv_list) == n
        assert len(ds_list) == n
        for i in range(n):
            assert len(cid_list[i]) == len(ds_list[i])
            assert len(sv_list[i]) == len(ds_list[i])

    def test_batch_flatten_format(self, lattice_cloud):
        starts, ends = self._make_rays()
        result = lattice_cloud.traced_projection(starts, ends, flatten=True)
        values, cell_ids, s_vals, ds_vals, offsets = result
        n = len(starts)
        assert offsets[0] == 0
        assert offsets[-1] == len(cell_ids)
        assert len(offsets) == n + 1
        assert len(cell_ids) == len(s_vals) == len(ds_vals)

    def test_batch_matches_sequential(self, lattice_cloud):
        """Batch result should match calling traced_projection per ray."""
        starts, ends = self._make_rays(4)
        values, cid_list, sv_list, ds_list = lattice_cloud.traced_projection(
            starts, ends)

        for i in range(len(starts)):
            val_i, cid_i, sv_i, ds_i = lattice_cloud.traced_projection(
                starts[i], ends[i])
            np.testing.assert_allclose(values[i], val_i, rtol=1e-12)
            np.testing.assert_array_equal(cid_list[i], cid_i)
            np.testing.assert_array_equal(sv_list[i], sv_i)
            np.testing.assert_array_equal(ds_list[i], ds_i)

    def test_batch_values_match_projection(self, lattice_cloud):
        """Reduced values should match projection() output."""
        starts, ends = self._make_rays(4)
        values, _, _, _ = lattice_cloud.traced_projection(starts, ends)
        proj_values = lattice_cloud.projection(starts, ends)
        np.testing.assert_allclose(values, proj_values, rtol=1e-12)

    def test_uniform_density_integral(self, lattice_cloud):
        """For uniform field=1, integral should equal ray length."""
        starts, ends = self._make_rays(4)
        values, _, _, ds_list = lattice_cloud.traced_projection(starts, ends)
        for i in range(len(starts)):
            ray_length = np.linalg.norm(ends[i] - starts[i])
            np.testing.assert_allclose(values[i], ray_length, rtol=1e-6)
            np.testing.assert_allclose(np.sum(ds_list[i]), ray_length,
                                       rtol=1e-12)


class TestMultiFieldBatch:
    def test_multifield_values_shape(self, lattice_cloud_multifield):
        n = 4
        starts = np.tile([2.0, 2.0, -0.5], (n, 1))
        ends = np.tile([2.0, 2.0, 4.5], (n, 1))
        values, _, _, _ = lattice_cloud_multifield.traced_projection(
            starts, ends)
        assert values.shape == (n, 2)

    def test_multifield_scaling(self, lattice_cloud_multifield):
        n = 4
        starts = np.tile([2.0, 2.0, -0.5], (n, 1))
        ends = np.tile([2.0, 2.0, 4.5], (n, 1))
        values, _, _, _ = lattice_cloud_multifield.traced_projection(
            starts, ends)
        np.testing.assert_allclose(values[:, 1], 2 * values[:, 0], rtol=1e-12)


class TestReductionModes:
    def test_max_reduction(self, lattice_cloud):
        starts = np.array([[2.0, 2.0, -0.5], [1.0, 1.0, -0.5]])
        ends = np.array([[2.0, 2.0, 4.5], [1.0, 1.0, 4.5]])
        values, _, _, _ = lattice_cloud.traced_projection(
            starts, ends, reduction='max')
        proj_values = lattice_cloud.projection(starts, ends, reduction='max')
        np.testing.assert_allclose(values, proj_values, rtol=1e-12)

    def test_min_reduction(self, lattice_cloud):
        starts = np.array([[2.0, 2.0, -0.5], [1.0, 1.0, -0.5]])
        ends = np.array([[2.0, 2.0, 4.5], [1.0, 1.0, 4.5]])
        values, _, _, _ = lattice_cloud.traced_projection(
            starts, ends, reduction='min')
        proj_values = lattice_cloud.projection(starts, ends, reduction='min')
        np.testing.assert_allclose(values, proj_values, rtol=1e-12)


class TestErrorHandling:
    def test_mismatched_shapes(self, lattice_cloud):
        with pytest.raises(ValueError):
            lattice_cloud.traced_projection(
                np.zeros((3, 3)), np.zeros((4, 3)))

    def test_wrong_columns(self, lattice_cloud):
        with pytest.raises(ValueError):
            lattice_cloud.traced_projection(
                np.zeros((3, 2)), np.zeros((3, 2)))

    def test_bad_1d_shape(self, lattice_cloud):
        with pytest.raises(ValueError):
            lattice_cloud.traced_projection(np.zeros(4), np.zeros(4))


class TestSingleProjectionRemoved:
    def test_single_projection_raises(self, lattice_cloud):
        with pytest.raises(AttributeError, match="traced_projection"):
            lattice_cloud.single_projection(
                np.array([0, 0, 0.]), np.array([1, 1, 1.]))
