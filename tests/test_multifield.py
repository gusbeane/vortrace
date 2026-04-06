"""Tests for multi-field integration and reduction modes."""

import numpy as np
import pytest
from vortrace import vortrace as vt
from vortrace import io


class TestMultiField:
    """Test multi-field projection functionality."""

    # ------------------------------------------------------------------
    # Multi-field grid projection
    # ------------------------------------------------------------------

    def test_multifield_grid_projection_scaling(self, arepo_snap):
        """Pass [dens, 2*dens]; second field should be 2x first."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        length = 75.

        fields = np.column_stack([dens, 2 * dens])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        extent = [box_size / 2. - length / 2., box_size / 2. + length / 2.]
        bounds = [0., box_size]
        npix = 64
        dat = pc.grid_projection(extent, npix, bounds, None)

        assert dat.ndim == 3
        assert dat.shape == (npix, npix, 2)
        np.testing.assert_allclose(dat[:, :, 1], 2 * dat[:, :, 0], rtol=1e-12)

    def test_multifield_matches_single_field(self, arepo_snap):
        """Multi-field first column matches standalone single-field result."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        length = 75.

        # Single field
        pc1 = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        extent = [box_size / 2. - length / 2., box_size / 2. + length / 2.]
        bounds = [0., box_size]
        npix = 64
        dat1 = pc1.grid_projection(extent, npix, bounds, None)

        # Multi field
        fields = np.column_stack([dens, 2 * dens])
        pc2 = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        dat2 = pc2.grid_projection(extent, npix, bounds, None)

        assert dat1.shape == (npix, npix)
        assert dat2.shape == (npix, npix, 2)
        np.testing.assert_allclose(dat2[:, :, 0], dat1, rtol=1e-12)

    def test_multifield_mass_weighted_temperature(self, arepo_snap):
        """Pass [dens, dens*T]; ratio recovers constant T."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        length = 75.

        T_const = 42.0
        fields = np.column_stack([dens, dens * T_const])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        extent = [box_size / 2. - length / 2., box_size / 2. + length / 2.]
        bounds = [0., box_size]
        npix = 64
        dat = pc.grid_projection(extent, npix, bounds, None)

        # dens*T / dens should equal T everywhere
        ratio = dat[:, :, 1] / dat[:, :, 0]
        np.testing.assert_allclose(ratio, T_const, rtol=1e-6)

    # ------------------------------------------------------------------
    # Multi-field projection() method
    # ------------------------------------------------------------------

    def test_multifield_projection(self, arepo_snap):
        """projection() with multi-field returns (N, nfields)."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        length = 75.

        fields = np.column_stack([dens, 3 * dens])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        N = 16
        rng = np.random.default_rng(99)
        starts = np.column_stack([
            rng.uniform(box_size / 2 - length / 3, box_size / 2 + length / 3,
                        N),
            rng.uniform(box_size / 2 - length / 3, box_size / 2 + length / 3,
                        N),
            np.full(N, 0.1),
        ])
        ends = starts.copy()
        ends[:, 2] = box_size - 0.1

        dat = pc.projection(starts, ends)
        assert dat.shape == (N, 2)
        np.testing.assert_allclose(dat[:, 1], 3 * dat[:, 0], rtol=1e-6)

    # ------------------------------------------------------------------
    # traced_projection multi-field
    # ------------------------------------------------------------------

    def test_traced_projection_multifield(self, arepo_snap):
        """traced_projection with multi-field returns array dens."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]

        fields = np.column_stack([dens, 5 * dens])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])

        start = np.array([box_size / 2., box_size / 2., 0.1])
        end = np.array([box_size / 2., box_size / 2., box_size - 0.1])

        dens_out, cell_ids, smid, ds = pc.traced_projection(start, end)
        assert isinstance(dens_out, np.ndarray)
        assert dens_out.shape == (2,)
        np.testing.assert_allclose(dens_out[1], 5 * dens_out[0], rtol=1e-6)


class TestReduction:
    """Test max/min reduction modes."""

    def test_max_reduction(self, arepo_snap):
        """Max reduction should give peak density along sightline."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]

        pc = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        extent = [box_size / 2. - 30., box_size / 2. + 30.]
        bounds = [0., box_size]
        npix = 32

        dat_max = pc.grid_projection(extent, npix, bounds, None,
                                     reduction='max')
        dat_sum = pc.grid_projection(extent, npix, bounds, None,
                                     reduction='integrate')

        assert dat_max.shape == (npix, npix)
        assert dat_sum.shape == (npix, npix)
        # Max values should be positive (densities are positive)
        assert np.all(dat_max > 0)
        # Max of a density field should generally be less than the integral
        # over a long path (for most sightlines through dense regions)
        # Just check they're different and max > 0
        assert not np.allclose(dat_max, dat_sum)

    def test_min_reduction(self, arepo_snap):
        """Min reduction should give minimum density along sightline."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]

        pc = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        extent = [box_size / 2. - 30., box_size / 2. + 30.]
        bounds = [0., box_size]
        npix = 32

        dat_max = pc.grid_projection(extent, npix, bounds, None,
                                     reduction='max')
        dat_min = pc.grid_projection(extent, npix, bounds, None,
                                     reduction='min')

        assert dat_min.shape == (npix, npix)
        # Min should be <= max everywhere
        assert np.all(dat_min <= dat_max)

    def test_max_multifield(self, arepo_snap):
        """Max reduction with multiple fields."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]

        fields = np.column_stack([dens, 2 * dens])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        extent = [box_size / 2. - 30., box_size / 2. + 30.]
        bounds = [0., box_size]
        npix = 32

        dat = pc.grid_projection(extent, npix, bounds, None, reduction='max')
        assert dat.shape == (npix, npix, 2)
        # max(2*dens) == 2 * max(dens)
        np.testing.assert_allclose(dat[:, :, 1], 2 * dat[:, :, 0], rtol=1e-12)


class TestSingleProjectionReduction:
    """Test traced_projection with different reduction modes."""

    def test_traced_projection_max_reduction(self, arepo_snap):
        """traced_projection with max reduction."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        pc = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        start = np.array([box_size / 2., box_size / 2., 0.1])
        end = np.array([box_size / 2., box_size / 2., box_size - 0.1])

        max_val, cell_ids, _, _ = pc.traced_projection(
            start, end, reduction='max')
        # Max should equal max of field values in intersected cells
        np.testing.assert_allclose(max_val, np.max(dens[cell_ids]))

    def test_traced_projection_min_reduction(self, arepo_snap):
        """traced_projection with min reduction."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        pc = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        start = np.array([box_size / 2., box_size / 2., 0.1])
        end = np.array([box_size / 2., box_size / 2., box_size - 0.1])

        min_val, cell_ids, _, _ = pc.traced_projection(
            start, end, reduction='min')
        np.testing.assert_allclose(min_val, np.min(dens[cell_ids]))

    def test_traced_projection_max_multifield(self, arepo_snap):
        """traced_projection max with multi-field: 2*dens max = 2 * dens max."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        fields = np.column_stack([dens, 2 * dens])
        pc = vt.ProjectionCloud(
            pos, fields,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        start = np.array([box_size / 2., box_size / 2., 0.1])
        end = np.array([box_size / 2., box_size / 2., box_size - 0.1])

        max_vals, _, _, _ = pc.traced_projection(
            start, end, reduction='max')
        assert max_vals.shape == (2,)
        np.testing.assert_allclose(max_vals[1], 2 * max_vals[0], rtol=1e-12)

    def test_traced_projection_invalid_reduction(self, arepo_snap):
        """Invalid reduction string should raise ValueError."""
        pos = arepo_snap["pos"]
        dens = arepo_snap["dens"]
        box_size = arepo_snap["box_size"]
        pc = vt.ProjectionCloud(
            pos, dens,
            boundbox=[0., box_size, 0., box_size, 0., box_size])
        start = np.array([box_size / 2., box_size / 2., 0.1])
        end = np.array([box_size / 2., box_size / 2., box_size - 0.1])
        with pytest.raises(ValueError, match="Unknown reduction"):
            pc.traced_projection(start, end, reduction='bogus')


class TestMultiFieldIO:
    """Test IO roundtrip with multi-field clouds."""

    def test_npz_roundtrip(self, tmp_path):
        rng = np.random.default_rng(123)
        pos = rng.random((200, 3)) * 100.0
        dens = rng.random((200, 3)) + 0.1  # 3 fields
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        cloud = vt.ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud_multi.npz"
        io.save_cloud(path, cloud)
        loaded = io.load_cloud(path)

        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig, cloud.fields_orig)
        assert loaded._nfields == 3
        assert loaded.boundbox == cloud.boundbox

    def test_hdf5_roundtrip(self, tmp_path):
        pytest.importorskip("h5py")
        rng = np.random.default_rng(123)
        pos = rng.random((200, 3)) * 100.0
        dens = rng.random((200, 3)) + 0.1
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        cloud = vt.ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud_multi.hdf5"
        io.save_cloud(path, cloud, fmt="hdf5")
        loaded = io.load_cloud(path)

        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig, cloud.fields_orig)
        assert loaded._nfields == 3
