"""Tests for adaptive bounding box padding (issues #19 and #20)."""

import numpy as np
import pytest
import warnings
from vortrace.vortrace import ProjectionCloud
from vortrace import io


class TestVolBasedPadding:
    """Test that vol-based adaptive padding includes nearby cells."""

    def test_vol_padding_includes_boundary_cells(self):
        """Points within 3*max_radius of box edge should be included."""
        rng = np.random.default_rng(42)
        # Box is [10, 90] in all dimensions
        boundbox = [10.0, 90.0, 10.0, 90.0, 10.0, 90.0]

        # Interior points
        pos_interior = rng.uniform(20, 80, size=(100, 3))

        # Point just outside the box but within adaptive padding
        # With vol giving max_radius ~ 5, pad = 3*5 = 15
        pos_boundary = np.array([[5.0, 50.0, 50.0]])  # 5 units outside

        pos = np.vstack([pos_interior, pos_boundary])
        fields = np.ones(len(pos))

        # Cell volumes: sphere of radius 5 -> V = 4/3 * pi * 5^3
        max_r = 5.0
        vol = np.full(len(pos), (4.0 / 3.0) * np.pi * max_r**3)

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox, vol=vol)

        # The boundary point at x=5 is within pad=15 of box edge x=10
        # so it should be included
        assert len(cloud.orig_ids) == len(pos)

    def test_vol_padding_excludes_far_cells(self):
        """Points far outside pad should be excluded."""
        rng = np.random.default_rng(42)
        boundbox = [10.0, 90.0, 10.0, 90.0, 10.0, 90.0]

        pos_interior = rng.uniform(20, 80, size=(100, 3))
        # Point very far outside box
        pos_far = np.array([[-100.0, 50.0, 50.0]])

        pos = np.vstack([pos_interior, pos_far])
        fields = np.ones(len(pos))

        max_r = 1.0
        vol = np.full(len(pos), (4.0 / 3.0) * np.pi * max_r**3)

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox, vol=vol)

        # The far point should be excluded
        assert len(cloud.orig_ids) == 100
        # orig_ids should not contain the last index
        assert 100 not in cloud.orig_ids


class TestUniformPadding:
    """Test uniform padding for thin bounding boxes."""

    def test_thin_box_uses_longest_side(self):
        """Thin box (100x100x1) should pad by 0.15*100=15 in all dims."""
        rng = np.random.default_rng(42)
        # Thin box: wide in x,y but thin in z
        boundbox = [0.0, 100.0, 0.0, 100.0, 50.0, 51.0]

        # Interior points
        pos_interior = rng.uniform(10, 90, size=(50, 3))
        pos_interior[:, 2] = 50.5  # all inside z range

        # Point outside z bounds but within uniform padding
        # Old per-dimension: pad_z = 0.15 * 1 = 0.15
        # New uniform: pad = 0.15 * 100 = 15
        pos_outside_z = np.array([[50.0, 50.0, 40.0]])  # 10 units below

        pos = np.vstack([pos_interior, pos_outside_z])
        fields = np.ones(len(pos))

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox)

        # With uniform padding = 15, the point at z=40 (10 below z=50)
        # should be included
        assert len(pos) == len(cloud.orig_ids)

    def test_thin_box_old_padding_would_exclude(self):
        """Verify that old per-dimension padding would have excluded the
        point, confirming the new uniform padding is necessary."""
        boundbox = [0.0, 100.0, 0.0, 100.0, 50.0, 51.0]

        # With old padding: pad_z = 0.15 * 1 = 0.15
        # Point at z=40 is 10 units outside, >> 0.15
        # With new uniform padding: pad = 0.15 * 100 = 15
        # Point at z=40 is 10 units outside, < 15
        pos = np.array([[50.0, 50.0, 50.5],
                        [50.0, 50.0, 40.0]])
        fields = np.ones(2)

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox)
        # Both points should be included with uniform padding
        assert len(cloud.orig_ids) == 2


class TestIOWithVol:
    """Test IO roundtrip preserves vol."""

    def test_npz_roundtrip_with_vol(self, tmp_path):
        rng = np.random.default_rng(123)
        pos = rng.random((200, 3)) * 100.0
        fields = rng.random(200) + 0.1
        vol = rng.random(200) * 10.0
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox,
                                vol=vol)
        path = tmp_path / "cloud_vol.npz"
        io.save_cloud(path, cloud)
        loaded = io.load_cloud(path)

        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig,
                                      cloud.fields_orig)
        np.testing.assert_array_equal(loaded.vol_orig, cloud.vol_orig)
        assert loaded.boundbox == cloud.boundbox

    def test_hdf5_roundtrip_with_vol(self, tmp_path):
        pytest.importorskip("h5py")
        rng = np.random.default_rng(123)
        pos = rng.random((200, 3)) * 100.0
        fields = rng.random(200) + 0.1
        vol = rng.random(200) * 10.0
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox,
                                vol=vol)
        path = tmp_path / "cloud_vol.hdf5"
        io.save_cloud(path, cloud, fmt="hdf5")
        loaded = io.load_cloud(path)

        np.testing.assert_array_equal(loaded.pos_orig, cloud.pos_orig)
        np.testing.assert_array_equal(loaded.fields_orig,
                                      cloud.fields_orig)
        np.testing.assert_array_equal(loaded.vol_orig, cloud.vol_orig)
        np.testing.assert_array_almost_equal(loaded.boundbox,
                                             cloud.boundbox)

    def test_npz_roundtrip_without_vol(self, tmp_path):
        """Clouds without vol should roundtrip with vol_orig=None."""
        rng = np.random.default_rng(123)
        pos = rng.random((200, 3)) * 100.0
        fields = rng.random(200) + 0.1
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        cloud = ProjectionCloud(pos, fields, boundbox=boundbox)
        path = tmp_path / "cloud_novol.npz"
        io.save_cloud(path, cloud)
        loaded = io.load_cloud(path)

        assert loaded.vol_orig is None


class TestWarning:
    """Test that a warning is emitted when vol is not provided."""

    def test_warning_without_vol(self):
        rng = np.random.default_rng(42)
        pos = rng.random((50, 3)) * 100.0
        fields = np.ones(50)
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProjectionCloud(pos, fields, boundbox=boundbox)
            user_warnings = [x for x in w
                             if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "vol" in str(user_warnings[0].message).lower()

    def test_no_warning_with_vol(self):
        rng = np.random.default_rng(42)
        pos = rng.random((50, 3)) * 100.0
        fields = np.ones(50)
        vol = np.ones(50) * 10.0
        boundbox = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProjectionCloud(pos, fields, boundbox=boundbox, vol=vol)
            user_warnings = [x for x in w
                             if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0
