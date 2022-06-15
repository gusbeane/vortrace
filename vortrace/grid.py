"""Basic grid module.

Collection of basic grid generation and manipulation routines.

Example:
    Example placeholders.

Todo:
    * Add examples.

"""

import numpy as np
from numba import njit


def _convert_proj_to_tait_bryan_angles_hand(proj):
    return {
        # Right handed projections.
        'xy': (0., 0., 0., 1),
        'yz': (np.pi / 2., 0., np.pi / 2., 1),
        'zx': (3. * np.pi / 2., 3. * np.pi / 2., 0., 1),

        # Left handed projections.
        'yx': (np.pi / 2., 0., np.pi, -1),
        'xz': (0., 0., np.pi / 2., -1),
        'zy': (0., 3. * np.pi / 2., 0., -1),
    }[proj]


@njit
def _generate_base_grid(extent, nres):
    grid = np.zeros((nres[0], nres[1], 3))

    deltax = (extent[0][1] - extent[0][0]) / (nres[0])
    deltay = (extent[1][1] - extent[1][0]) / (nres[1])

    # Now create grid.
    for i in range(nres[0]):
        for j in range(nres[1]):
            x = deltax * (i + 0.5)
            y = deltay * (j + 0.5)

            grid[i][j][0] = x
            grid[i][j][1] = y

    return grid


def generate_base_grid(extent, nres):
    """Generates a 2D primitive grid.

    Generates a 2D primitive grid spanning a given extent with a certain number
    of resolution elements. The grid is 2D but the output is given as a 3D
    array with the z-values of all grid points set to zero.

    Args:
        extent (array of float):  A (2,2) or (2,) array specifying the extent
            of the base grid. If a (2,) array is given, assumes the same extent
            in both directions.
        nres (int or array of int): Number of resolution elements. Either an
            int or a (2,) array of ints, specifying in each direction.

    Returns:
        grid (array of float): A (N,3) array of grid points, where N is the
            total number of resolution elements in the grid.
    """

    # Preprocess arguemnts.
    extent = np.array(extent)

    if extent.ravel().size == 2:
        extent = np.array([extent, extent])
    elif extent.ravel().size == 4:
        pass
    else:
        raise ValueError('extent must be either (2,2) or (2,) array')

    if isinstance(nres, int):
        nres = (nres, nres)

    # Prepare output and for loop.
    grid = _generate_base_grid(extent, nres)

    return grid


def _rotation_matrix_from_euler(phi, theta, psi):
    """Generates a rotation matrix from Euler angles.

    Generates a rotation matrix from a given set of Euler angles.

    The Euler angle convention used here is that given by the 'z-x-z'
    convention, described here: https://mathworld.wolfram.com/EulerAngles.html

    Args:
        phi (float): Angle phi, describing the rotation about the original
            z-axis
        theta (float): Angle theta, describing the rotation about the new
            xprime-axis.
        psi (float): Angle psi, describing the rotation about the new
            zprime-axis.

    Returns:
        grid (array of float): A (N,3) array of grid points, where N is the
            total number of resolution elements in the grid.
    """

    matd = np.array([
        [np.cos(phi), np.sin(phi), 0.],
        [-np.sin(phi), np.cos(phi), 0.],
        [0., 0., 1.],
    ])

    matc = np.array([
        [1., 0., 0.],
        [0., np.cos(theta), np.sin(theta)],
        [0., -np.sin(theta), np.cos(theta)],
    ])

    matb = np.array([
        [np.cos(psi), np.sin(psi), 0.],
        [-np.sin(psi), np.cos(psi), 0.],
        [0., 0., 1.],
    ])

    mata = np.matmul(np.matmul(matb, matc), matd)

    return mata


def _rotation_matrix_from_yaw_pitch_roll(yaw, pitch, roll):
    ca, sa = np.cos(yaw), np.sin(yaw)
    cb, sb = np.cos(pitch), np.sin(pitch)
    cg, sg = np.cos(roll), np.sin(roll)

    rot_mat = np.array([
        [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
        [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
        [-sb, cb * sg, cb * cg],
    ])

    return rot_mat


def generate_projection_grid(extent,
                             nres,
                             bounds,
                             center,
                             proj=None,
                             yaw=0.,
                             pitch=0.,
                             roll=0.):
    """Generates a projection grid.

    This generates a projection grid which can be arbitrarily rotated about a
    center. A grid_start and grid_end grids are returned, which give the
    starting and ending integration positions to be integrated.

    The grid is constructed by first laying down a grid in the x-y plane
    spanning extent with nres elements. We then construct grid_start and
    grid_end by assigning the z-coordinates of the base grid either bounds[0]
    or bounds[1], respectively. Both grids are then optionally rotated about
    the point center.

    You can use grid_start and grid_end as starting and ending positions for
    projection integrations.

    The Euler angle convention used here is that given by the 'x-z-x'
    convention, described here: https://mathworld.wolfram.com/EulerAngles.html

    Args:
        extent (array of float):  A (2,2) or (2,) array specifying the extent
            of the base grid. If a (2,) array is given, assumes the same extent
            in both directions.
        nres (int or array of int): Number of resolution elements. Either an
            int or a (2,) array of ints, specifying in each direction.
        bounds (array of float): A (2,) array indicating the start and end
            bounds for the integration. The integration will start at bounds[0]
            and end at bounds[1] along the z-axis before any rotation.
        center (array of float or None): A (3,) array indicating the center
            about which the Euler rotations should be performed. If center is
            set to None, no rotation is done and the projection is by default
            'xy'.
        proj (string): A two letter string indicating the desired Cartesian
            projection
        yaw (float, optional): Yaw angle, describing the rotation about the
            original z-axis
        pitch (float, optional): Pitch angle, describing the rotation about the
            new xprime-axis.
        roll (float, optional): Roll angle, describing the rotation about the
            new zprime-axis.

    Returns:
        grid_start (array of float): A (N0,N1,3) array of grid points, where N0
            and N1 are nres[0] and nres[1], respectively (or nres if nres is an
            int). Starting points for integration specified by bounds.
        grid_end (array of float): A (N0,N1,3) array of grid poitns, where N0
            and N1 are nres[0] and nres[1], respectively (or nres if nres is an
            int). Ending points for integration specified by bounds.
    """

    # First, we make a grid that is aligned in the x-y plane and goes from
    # bounds[0] to bounds[1]
    grid_start = generate_base_grid(extent, nres)
    grid_end = np.copy(grid_start)

    if proj is not None:
        yaw, pitch, roll, hand = _convert_proj_to_tait_bryan_angles_hand(proj)
    else:
        hand = np.sign(bounds[1] - bounds[0])

    # If user specifies a projection and the handedness of the provided bounds
    # does not match the handedness of the projection, we will kindly fix it
    # for the user.
    if hand != np.sign(bounds[1] - bounds[0]):
        bounds = np.flip(bounds)

    grid_start[..., 2] = bounds[0]
    grid_end[..., 2] = bounds[1]

    # Subtract the center and do the euler rotation.
    if center is not None:
        grid_start -= center
        grid_end -= center

        # Do the Euler rotation.
        rot_mat = _rotation_matrix_from_yaw_pitch_roll(yaw, pitch, roll)

        grid_start = np.dot(grid_start, rot_mat.T)
        grid_end = np.dot(grid_end, rot_mat.T)

        # Add back in the center.
        grid_start += center
        grid_end += center

    return grid_start, grid_end
