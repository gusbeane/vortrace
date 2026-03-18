"""Smoke tests for vortrace.plot functions."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest  # noqa: E402
from vortrace import plot  # noqa: E402


class TestPlotGrid:
    """Smoke tests for plot_grid."""

    def _make_grid(self):
        rng = np.random.default_rng(0)
        return rng.random((64, 64)) + 1e-3  # avoid log(0)

    def test_basic(self):
        data = self._make_grid()
        fig, ax, im = plot.plot_grid(data)
        assert fig is not None
        assert ax is not None
        assert im is not None

    def test_with_extent(self):
        data = self._make_grid()
        fig, ax, im = plot.plot_grid(data, extent=[-1, 1, -1, 1])
        assert fig is not None

    def test_linear_scale(self):
        data = self._make_grid()
        fig, ax, im = plot.plot_grid(data, log=False)
        assert im.norm is not None

    def test_no_colorbar(self):
        data = self._make_grid()
        fig, ax, im = plot.plot_grid(data, colorbar=False)
        assert fig is not None

    def test_custom_ax(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig2, ax2, im = plot.plot_grid(self._make_grid(), ax=ax)
        assert fig2 is fig
        assert ax2 is ax


class TestPlotRay:
    """Smoke tests for plot_ray."""

    def test_basic(self):
        s = np.linspace(0, 10, 50)
        d = np.exp(-s) + 1e-3
        fig, ax = plot.plot_ray(s, d)
        assert fig is not None
        assert ax is not None

    def test_linear_scale(self):
        s = np.linspace(0, 10, 50)
        d = np.exp(-s) + 1e-3
        fig, ax = plot.plot_ray(s, d, log=False)
        assert ax.get_yscale() == "linear"

    def test_custom_ax(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        s = np.linspace(0, 5, 20)
        d = np.ones_like(s)
        fig2, ax2 = plot.plot_ray(s, d, ax=ax)
        assert fig2 is fig
        assert ax2 is ax
