import numpy as np
from numba import njit


@njit
def _generate_base_grid(extent, nres):
    grid = np.zeros((nres[0] * nres[1], 3))

    deltax = (extent[0][1] - extent[0][0]) / (nres[0])
    deltay = (extent[1][1] - extent[1][0]) / (nres[1])

    # Now create grid.
    for i in range(nres[0]):
        for j in range(nres[1]):
            k = i * nres[1] + j

            x = deltax * (i + 0.5)
            y = deltay * (j + 0.5)

            grid[k][0] = x
            grid[k][1] = y

    return grid


def generate_base_grid(extent, nres):
    """Generates a 2D primitive grid.

    Generates a 2D primitive grid spanning a given extent with a certain number of
    resolution elements. The grid is 2D but the output is given as a 3D array with
    the z-values of all grid points set to zero.

    Args:
        extent (array of float):  A (2,2) or (2,) array specifying the extent of the base grid.
            If a (2,) array is given, assumes the same extent in both directions.
        nres (int or array of int): Number of resolution elements. Either an int or a (2,) array
            of ints, specifying in each direction.
    
    Returns:
        grid (array of float): A (N,3) array of grid points, where N is the total number of resolution
            elements in the grid.
    """

    # Preprocess arguemnts.
    extent = np.array(extent)

    if extent.ravel().size == 2:
        extent = np.array([extent, extent])
    elif extent.ravel().size == 4:
        extent = extent
    else:
        raise ValueError("extent must be either (2,2) or (2,) array")

    if isinstance(nres, int):
        nres = (nres, nres)

    # Prepare output and for loop.
    grid = _generate_base_grid(extent, nres)

    return grid


if __name__ == '__main__':
    grid0 = generate_base_grid([0, 1], 64)
    grid1 = generate_base_grid([[0, 1], [0, 1]], (64, 64))
