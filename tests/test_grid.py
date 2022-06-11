from vortrace import grid as gr
import numpy as np


class TestBaseGrid:

    def test_simple(self):
        grid0 = gr.generate_base_grid([0, 1], 64)
        grid1 = gr.generate_base_grid([[0, 1], [0, 1]], (64, 64))
        np.testing.assert_equal(grid0, grid1)
        assert grid0.shape == (64 * 64, 3)

        # Check some specific values of the grid.
        assert grid0[0][0] == 0.5 / 64.
        assert grid0[-1][0] == 63.5 / 64.
        assert grid0[0][1] == 0.5 / 64.
        assert grid0[-1][1] == 63.5 / 64.

    def test_rectangle(self):
        grid = gr.generate_base_grid([[0, 1], [0, 3]], (64, 128))
        assert grid.shape == (64 * 128, 3)

        # Check some specific values of the grid.
        assert grid[0][0] == 0.5 / 64.
        assert grid[-1][0] == 63.5 / 64.
        assert grid[0][1] == 0.5 / (128. / 3.)
        assert grid[-1][1] == 127.5 / (128 / 3.)
