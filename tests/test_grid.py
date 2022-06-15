from vortrace import grid as gr
import numpy as np


class TestBaseGrid:

    def test_simple(self):
        grid0 = gr.generate_base_grid([0, 1], 64)
        grid1 = gr.generate_base_grid([[0, 1], [0, 1]], (64, 64))
        np.testing.assert_equal(grid0, grid1)
        assert grid0.shape == (64, 64, 3)

        # Check some specific values of the grid.
        assert grid0[0][0][0] == 0.5 / 64.
        assert grid0[-1][-1][0] == 63.5 / 64.
        assert grid0[0][0][1] == 0.5 / 64.
        assert grid0[-1][-1][1] == 63.5 / 64.

    def test_rectangle(self):
        grid = gr.generate_base_grid([[0, 1], [0, 3]], (64, 128))
        assert grid.shape == (64, 128, 3)

        # Check some specific values of the grid.
        assert grid[0][0][0] == 0.5 / 64.
        assert grid[-1][-1][0] == 63.5 / 64.
        assert grid[0][0][1] == 0.5 / (128. / 3.)
        assert grid[-1][-1][1] == 127.5 / (128 / 3.)
    
    def test_offset(self):
        grid0 = gr.generate_base_grid([3, 4], 64)
        assert grid0.shape == (64, 64, 3)

        # Check some specific values of the grid.
        assert grid0[0][0][0] == 3 + 0.5 / 64.
        assert grid0[-1][-1][0] == 3 + 63.5 / 64.
        assert grid0[0][0][1] == 3 + 0.5 / 64.
        assert grid0[-1][-1][1] == 3 + 63.5 / 64.


class TestProjectionGrid:

    def test_xy(self):
        grid_s, _ = gr.generate_projection_grid([0, 1], 2, [0, 1],
                                                [0.5, 0.5, 0.5])

        grid_s_xy = np.array([
            [[0.25, 0.25, 0.], [0.25, 0.75, 0.]],
            [[0.75, 0.25, 0.], [0.75, 0.75, 0.]],
        ])

        np.testing.assert_almost_equal(grid_s, grid_s_xy)

    def test_yz(self):
        grid_s, _ = gr.generate_projection_grid([0, 1],
                                                2, [0, 1], [0.5, 0.5, 0.5],
                                                proj='yz')

        grid_s_yz = np.array([
            [[0., 0.25, 0.25], [0., 0.25, 0.75]],
            [[0., 0.75, 0.25], [0., 0.75, 0.75]],
        ])

        np.testing.assert_almost_equal(grid_s, grid_s_yz)

    def test_zx(self):
        grid_s, _ = gr.generate_projection_grid([0, 1],
                                                2, [0, 1], [0.5, 0.5, 0.5],
                                                proj='zx')

        grid_s_zx = np.array([
            [[0.25, 0., 0.25], [0.75, 0., 0.25]],
            [[0.25, 0., 0.75], [0.75, 0., 0.75]],
        ])

        np.testing.assert_almost_equal(grid_s, grid_s_zx)

    def test_yx(self):
        grid_s, _ = gr.generate_projection_grid([0, 1],
                                                2, [0, 1], [0.5, 0.5, 0.5],
                                                proj='yx')

        grid_s_yx = np.array([
            [[0.25, 0.25, 0.], [0.75, 0.25, 0.]],
            [[0.25, 0.75, 0.], [0.75, 0.75, 0.]],
        ])

        np.testing.assert_almost_equal(grid_s, grid_s_yx)

    def test_zy(self):
        grid_s, _ = gr.generate_projection_grid([0, 1],
                                                2, [0, 1], [0.5, 0.5, 0.5],
                                                proj='zy')

        grid_s_zy = np.array([
            [[0., 0.25, 0.25], [0., 0.75, 0.25]],
            [[0., 0.25, 0.75], [0., 0.75, 0.75]],
        ])

        np.testing.assert_almost_equal(grid_s, grid_s_zy)

    def test_xz(self):
        grid_s, _ = gr.generate_projection_grid([0, 1],
                                                2, [0, 1], [0.5, 0.5, 0.5],
                                                proj='xz')

        grid_s_xz = np.array([
            [[0.25, 0., 0.25], [0.25, 0., 0.75]],
            [[0.75, 0., 0.25], [0.75, 0., 0.75]],
        ])

        np.testing.assert_almost_equal(grid_s, grid_s_xz)
