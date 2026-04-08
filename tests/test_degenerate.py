"""Tests for degenerate split point handling in ray integration.

Uses a cubic lattice of Voronoi generators with uniform field=1.0.
For Sum mode, the integral should equal the ray length.
"""
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


def integrate_ray(pos, fields, vol, boundbox, start, end):
    """Set up point cloud and integrate a single ray. Returns scalar column density."""
    pc = vt.ProjectionCloud(pos, fields, boundbox=boundbox, vol=vol)
    dens, _cell_ids, _s_vals, _ds_vals = pc.traced_projection(
        np.array(start), np.array(end))
    return dens


def ray_length(start, end):
    """Euclidean distance between start and end."""
    return np.linalg.norm(np.array(end) - np.array(start))


class TestDegenerateSplitPoints:
    """Tests for rays that hit degenerate Voronoi geometry."""

    def test_diagonal_through_vertices(self):
        """Ray along body diagonal, passing through vertices where 8 cells meet."""
        pos, fields, vol, bb = make_cubic_lattice()
        start = [-0.1, -0.1, -0.1]
        end = [4.1, 4.1, 4.1]
        result = integrate_ray(pos, fields, vol, bb, start, end)
        expected = ray_length(start, end)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_ray_along_face(self):
        """Ray along face plane x=0.5; bisector normal is perpendicular to ray."""
        pos, fields, vol, bb = make_cubic_lattice()
        start = [0.5, 0.2, -0.1]
        end = [0.5, 0.2, 4.1]
        result = integrate_ray(pos, fields, vol, bb, start, end)
        expected = ray_length(start, end)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_ray_along_edge(self):
        """Ray along edge where 4 cells meet; combines parallel bisector + vertex."""
        pos, fields, vol, bb = make_cubic_lattice()
        start = [0.5, 0.5, -0.1]
        end = [0.5, 0.5, 4.1]
        result = integrate_ray(pos, fields, vol, bb, start, end)
        expected = ray_length(start, end)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_ray_through_single_vertex(self):
        """Ray through one vertex at (0.5, 0.5, 0.5)."""
        pos, fields, vol, bb = make_cubic_lattice()
        start = [0.1, 0.1, 0.1]
        end = [0.9, 0.9, 0.9]
        result = integrate_ray(pos, fields, vol, bb, start, end)
        expected = ray_length(start, end)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_near_degenerate_sanity(self):
        """Ray near but not on degenerate geometry — regression check."""
        pos, fields, vol, bb = make_cubic_lattice()
        start = [0.50001, 0.50001, -0.1]
        end = [0.50001, 0.50001, 4.1]
        result = integrate_ray(pos, fields, vol, bb, start, end)
        expected = ray_length(start, end)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_smoke_selective_field(self):
        """Only one cell has nonzero field; integral should be positive."""
        pos, fields, vol, bb = make_cubic_lattice()
        fields = np.zeros(len(pos), dtype=np.float64)
        # Set field=1 for the cell at (2,2,2)
        center_idx = np.argmin(np.linalg.norm(pos - [2.0, 2.0, 2.0], axis=1))
        fields[center_idx] = 1.0
        result = integrate_ray(pos, fields, vol, bb,
                               [2.0, 2.0, -0.1], [2.0, 2.0, 4.1])
        assert result > 0.0, f"Expected positive integral, got {result}"

    def test_float32_precision_positions(self):
        """Positions centered in float32 then converted to float64.

        Regression test for split-point-outside-segment-bounds error
        when positions have float32 quantization noise (e.g. from the
        arepo snapshot loader).
        """
        n = 10
        pos, fields, vol, bb = make_cubic_lattice(n)

        # Shift to a large offset, round-trip through float32, shift back.
        # This mimics loading coordinates as float32 and recentering.
        offset = np.array([500.0, 500.0, 500.1])
        pos_f32 = (pos + offset).astype(np.float32)
        pos_f32 -= offset.astype(np.float32)
        pos_degraded = pos_f32.astype(np.float64)

        # Diagonal ray through the lattice
        start = [-0.1, -0.1, -0.1]
        end = [n - 0.9, n - 0.9, n - 0.9]
        result = integrate_ray(pos_degraded, fields, vol, bb, start, end)
        expected = ray_length(start, end)
        np.testing.assert_allclose(result, expected, rtol=1e-3)
