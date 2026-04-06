"""Roundtrip tests for vortrace.io save/load functions."""

import numpy as np
import pytest
from vortrace import io
from vortrace.vortrace import ProjectionCloud


# ---------------------------------------------------------------------------
# Grid roundtrip tests
# ---------------------------------------------------------------------------

class TestGridIO:
    """Test save_grid / load_grid roundtrips."""

    def _make_grid(self):
        rng = np.random.default_rng(42)
        return rng.random((32, 32))

    def test_npz_roundtrip(self, tmp_path):
        data = self._make_grid()
        path = tmp_path / "grid.npz"
        io.save_grid(path, data)
        loaded, meta = io.load_grid(path)
        np.testing.assert_array_equal(loaded, data)
        assert meta == {}

    def test_npz_with_extent_and_metadata(self, tmp_path):
        data = self._make_grid()
        extent = [0.0, 1.0, 0.0, 1.0]
        metadata = {"units": "Msun/kpc^2", "npix": 32}
        path = tmp_path / "grid.npz"
        io.save_grid(path, data, extent=extent, metadata=metadata, fmt="npz")
        loaded, meta = io.load_grid(path)
        np.testing.assert_array_equal(loaded, data)
        np.testing.assert_array_almost_equal(meta["extent"], extent)
        assert meta["units"] == "Msun/kpc^2"
        assert meta["npix"] == 32

    def test_hdf5_roundtrip(self, tmp_path):
        h5py = pytest.importorskip("h5py")  # noqa: F841
        data = self._make_grid()
        path = tmp_path / "grid.hdf5"
        io.save_grid(path, data, fmt="hdf5")
        loaded, meta = io.load_grid(path)
        np.testing.assert_array_equal(loaded, data)

    def test_hdf5_with_extent_and_metadata(self, tmp_path):
        h5py = pytest.importorskip("h5py")  # noqa: F841
        data = self._make_grid()
        extent = [-10.0, 10.0, -10.0, 10.0]
        metadata = {"projection": "xy"}
        path = tmp_path / "grid.hdf5"
        io.save_grid(path, data, extent=extent, metadata=metadata, fmt="hdf5")
        loaded, meta = io.load_grid(path)
        np.testing.assert_array_equal(loaded, data)
        np.testing.assert_array_almost_equal(meta["extent"], extent)
        assert meta["projection"] == "xy"

    def test_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            io.save_grid(tmp_path / "grid.fits", self._make_grid(), fmt="fits")


# ---------------------------------------------------------------------------
# Cloud roundtrip tests
# ---------------------------------------------------------------------------

class TestCloudIO:
    """Test save_cloud / load_cloud roundtrips."""

    def _make_cloud_data(self):
        rng = np.random.default_rng(123)
        pos = rng.random((200, 3)) * 100.0
        dens = rng.random(200) + 0.1
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
        return pos, dens, boundbox

    def test_npz_roundtrip(self, tmp_path):
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud.npz"
        io.save_cloud(path, cloud)
        loaded = io.load_cloud(path)
        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig, cloud.fields_orig)
        assert loaded.boundbox == cloud.boundbox

    def test_hdf5_roundtrip(self, tmp_path):
        h5py = pytest.importorskip("h5py")  # noqa: F841
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud.hdf5"
        io.save_cloud(path, cloud, fmt="hdf5")
        loaded = io.load_cloud(path)
        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig, cloud.fields_orig)
        np.testing.assert_array_almost_equal(loaded.boundbox, cloud.boundbox)

    def test_convenience_methods(self, tmp_path):
        """Test ProjectionCloud.save() and ProjectionCloud.load()."""
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud.npz"
        cloud.save(path)
        loaded = ProjectionCloud.load(path)
        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig, cloud.fields_orig)

    def test_unknown_format_raises(self, tmp_path):
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        with pytest.raises(ValueError, match="Unknown format"):
            io.save_cloud(tmp_path / "cloud.fits", cloud, fmt="fits")

    def test_npz_roundtrip_with_tree(self, tmp_path):
        """Cloud saved with tree loads without rebuilding."""
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud_tree.npz"
        io.save_cloud(path, cloud, save_tree=True)
        loaded = io.load_cloud(path)
        assert loaded._cloud.get_tree_built()
        result = loaded.grid_projection([10., 90.], 8, [0., 100.], None)
        assert result.shape == (8, 8)

    def test_npz_roundtrip_without_tree(self, tmp_path):
        """Cloud saved without tree still loads and rebuilds."""
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud_notree.npz"
        io.save_cloud(path, cloud, save_tree=False)
        loaded = io.load_cloud(path)
        assert loaded._cloud.get_tree_built()
        result = loaded.grid_projection([10., 90.], 8, [0., 100.], None)
        assert result.shape == (8, 8)

    def test_hdf5_roundtrip_with_tree(self, tmp_path):
        h5py = pytest.importorskip("h5py")  # noqa: F841
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        path = tmp_path / "cloud_tree.hdf5"
        io.save_cloud(path, cloud, fmt="hdf5", save_tree=True)
        loaded = io.load_cloud(path)
        assert loaded._cloud.get_tree_built()
        result = loaded.grid_projection([10., 90.], 8, [0., 100.], None)
        assert result.shape == (8, 8)

    def test_tree_produces_same_results(self, tmp_path):
        """Loaded tree gives identical query results to rebuilt tree."""
        pos, dens, boundbox = self._make_cloud_data()
        cloud = ProjectionCloud(pos, dens, boundbox=boundbox)
        result_orig = cloud.grid_projection([10., 90.], 16, [0., 100.], None)

        path = tmp_path / "cloud.npz"
        io.save_cloud(path, cloud, save_tree=True)
        loaded = io.load_cloud(path)
        result_loaded = loaded.grid_projection([10., 90.], 16, [0., 100.],
                                               None)

        np.testing.assert_array_equal(result_orig, result_loaded)
