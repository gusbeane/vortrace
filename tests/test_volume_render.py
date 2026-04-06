"""Tests for volume rendering reduction mode."""

import numpy as np
import pytest
from vortrace import vortrace as vt
from vortrace.Cvortrace import ReductionMode


class TestVolumeRenderBasic:
    """Test volume rendering with synthetic data."""

    def _make_uniform_cloud(self, rgba, n=500, box=100.0):
        """Create a cloud where every cell has the same RGBA values."""
        rng = np.random.default_rng(42)
        pos = rng.random((n, 3)) * box
        fields = np.tile(rgba, (n, 1))  # shape (n, 4)
        boundbox = [0., box, 0., box, 0., box]
        return vt.ProjectionCloud(pos, fields, boundbox=boundbox), box

    def test_transparent_medium(self):
        """Alpha=0 everywhere: result should be (0, 0, 0)."""
        pc, box = self._make_uniform_cloud([1.0, 0.5, 0.25, 0.0])
        start = np.array([box / 2, box / 2, 0.1])
        end = np.array([box / 2, box / 2, box - 0.1])

        rgb, _, _, _ = pc.traced_projection(start, end, reduction='volume')
        assert rgb.shape == (3,)
        np.testing.assert_allclose(rgb, [0.0, 0.0, 0.0], atol=1e-15)

    def test_uniform_field_analytic(self):
        """Uniform RGBA: compare to analytic integral.

        For uniform emission C and absorption alpha over total path L:
          result_c = C * (1 - exp(-alpha * L))
        (each segment contributes, and the geometric series sums to this).
        """
        R, G, B, alpha = 0.8, 0.4, 0.2, 0.01
        pc, box = self._make_uniform_cloud([R, G, B, alpha])
        start = np.array([box / 2, box / 2, 0.1])
        end = np.array([box / 2, box / 2, box - 0.1])

        rgb, _, _, ds_vals = pc.traced_projection(
            start, end, reduction='volume')
        total_L = np.sum(ds_vals)

        expected = np.array([R, G, B]) * (1 - np.exp(-alpha * total_L))
        np.testing.assert_allclose(rgb, expected, rtol=1e-10)

    def test_high_opacity_saturates(self):
        """Very high alpha: result should approach (R, G, B)."""
        R, G, B = 1.0, 0.5, 0.25
        alpha = 1000.0  # extremely opaque
        pc, box = self._make_uniform_cloud([R, G, B, alpha])
        start = np.array([box / 2, box / 2, 0.1])
        end = np.array([box / 2, box / 2, box - 0.1])

        rgb, _, _, _ = pc.traced_projection(start, end, reduction='volume')
        np.testing.assert_allclose(rgb, [R, G, B], rtol=1e-6)

    def test_reversed_ray_differs(self):
        """Reversing the ray direction should give different results
        (unless the field is perfectly uniform).
        """
        rng = np.random.default_rng(123)
        n = 500
        box = 100.0
        pos = rng.random((n, 3)) * box
        # Non-uniform: alpha varies with position
        alpha = 0.01 + 0.05 * rng.random(n)
        fields = np.column_stack([
            rng.random(n), rng.random(n), rng.random(n), alpha
        ])
        boundbox = [0., box, 0., box, 0., box]
        pc = vt.ProjectionCloud(pos, fields, boundbox=boundbox)

        start = np.array([box / 2, box / 2, 0.1])
        end = np.array([box / 2, box / 2, box - 0.1])

        rgb_fwd, _, _, _ = pc.traced_projection(start, end, reduction='volume')
        rgb_rev, _, _, _ = pc.traced_projection(end, start, reduction='volume')

        # Forward and reverse should differ for non-uniform fields
        assert not np.allclose(rgb_fwd, rgb_rev, atol=1e-10)


class TestVolumeRenderValidation:
    """Test that volume rendering validates nfields == 4 and field ranges."""

    def test_rgb_out_of_range_raises(self):
        """RGB values > 1 should raise ValueError."""
        rng = np.random.default_rng(1)
        pos = rng.random((100, 3)) * 10
        fields = np.column_stack([
            rng.random(100) * 2,  # R in [0, 2] — some > 1
            rng.random(100),
            rng.random(100),
            rng.random(100) * 0.1,
        ])
        pc = vt.ProjectionCloud(pos, fields, boundbox=[0, 10, 0, 10, 0, 10])
        with pytest.raises(ValueError, match=r"R, G, B.*\[0, 1\]"):
            pc.grid_projection([2, 8], 8, [0, 10], None, reduction='volume')

    def test_negative_rgb_raises(self):
        """Negative RGB values should raise ValueError."""
        rng = np.random.default_rng(1)
        pos = rng.random((100, 3)) * 10
        fields = np.column_stack([
            -0.1 * np.ones(100),  # R < 0
            rng.random(100),
            rng.random(100),
            rng.random(100) * 0.1,
        ])
        pc = vt.ProjectionCloud(pos, fields, boundbox=[0, 10, 0, 10, 0, 10])
        with pytest.raises(ValueError, match=r"R, G, B.*\[0, 1\]"):
            pc.projection(
                np.array([[5, 5, 0.1]]), np.array([[5, 5, 9.9]]),
                reduction='volume')

    def test_negative_alpha_raises(self):
        """Negative alpha should raise ValueError."""
        rng = np.random.default_rng(1)
        pos = rng.random((100, 3)) * 10
        fields = np.column_stack([
            rng.random(100),
            rng.random(100),
            rng.random(100),
            -0.1 * np.ones(100),  # alpha < 0
        ])
        pc = vt.ProjectionCloud(pos, fields, boundbox=[0, 10, 0, 10, 0, 10])
        with pytest.raises(ValueError, match="alpha >= 0"):
            pc.traced_projection([5, 5, 0.1], [5, 5, 9.9],
                                 reduction='volume')

    def test_nfields_1_raises(self):
        rng = np.random.default_rng(1)
        pos = rng.random((100, 3)) * 10
        fields = rng.random(100)
        pc = vt.ProjectionCloud(pos, fields, boundbox=[0, 10, 0, 10, 0, 10])
        with pytest.raises(ValueError, match="4 fields"):
            pc.grid_projection([2, 8], 8, [0, 10], None, reduction='volume')

    def test_nfields_3_raises(self):
        rng = np.random.default_rng(1)
        pos = rng.random((100, 3)) * 10
        fields = rng.random((100, 3))
        pc = vt.ProjectionCloud(pos, fields, boundbox=[0, 10, 0, 10, 0, 10])
        with pytest.raises(ValueError, match="4 fields"):
            pc.projection(
                np.array([[5, 5, 0.1]]), np.array([[5, 5, 9.9]]),
                reduction='volume')

    def test_nfields_2_traced_projection_raises(self):
        rng = np.random.default_rng(1)
        pos = rng.random((100, 3)) * 10
        fields = rng.random((100, 2))
        pc = vt.ProjectionCloud(pos, fields, boundbox=[0, 10, 0, 10, 0, 10])
        with pytest.raises(ValueError, match="4 fields"):
            pc.traced_projection([5, 5, 0.1], [5, 5, 9.9],
                                 reduction='volume')


class TestVolumeRenderProjection:
    """Test volume rendering through grid and batch projection methods."""

    def test_grid_projection_shape(self, arepo_snap):
        """Grid projection with volume rendering returns (npix, npix, 3)."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]

        # Construct 4 fields: R=dens/max, G=dens/max, B=dens/max,
        # alpha=dens/max
        dmax = dens.max()
        fields = np.column_stack([
            dens / dmax, dens / dmax, dens / dmax, dens / dmax
        ])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        npix = 32
        extent = [box_size / 2 - 30, box_size / 2 + 30]
        bounds = [0, box_size]
        dat = pc.grid_projection(extent, npix, bounds, None,
                                 reduction='volume')

        assert dat.shape == (npix, npix, 3)
        # RGB values should be non-negative
        assert np.all(dat >= 0)

    def test_projection_shape(self, arepo_snap):
        """projection() with volume rendering returns (N, 3)."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        dmax = dens.max()
        fields = np.column_stack([
            dens / dmax, dens / dmax, dens / dmax, dens / dmax
        ])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        N = 16
        rng = np.random.default_rng(99)
        starts = np.column_stack([
            rng.uniform(box_size / 2 - 20, box_size / 2 + 20, N),
            rng.uniform(box_size / 2 - 20, box_size / 2 + 20, N),
            np.full(N, 0.1),
        ])
        ends = starts.copy()
        ends[:, 2] = box_size - 0.1

        dat = pc.projection(starts, ends, reduction='volume')
        assert dat.shape == (N, 3)
        assert np.all(dat >= 0)

    def test_traced_projection_returns_3(self, arepo_snap):
        """traced_projection with volume rendering returns 3-element array."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        dmax = dens.max()
        fields = np.column_stack([
            dens / dmax, dens / dmax, dens / dmax, dens / dmax
        ])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        start = np.array([box_size / 2, box_size / 2, 0.1])
        end = np.array([box_size / 2, box_size / 2, box_size - 0.1])

        rgb, cell_ids, smid, ds = pc.traced_projection(
            start, end, reduction='volume')
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (3,)
        assert np.all(rgb >= 0)
        assert len(cell_ids) > 0
