"""High-level interface for projections through Voronoi meshes.

Provides :class:`ProjectionCloud`, the main user-facing class that wraps
the C++ backend for grid projections, direct ray projections, and
single-ray segment queries.
"""

import logging

from .Cvortrace import (  # type: ignore
    PointCloud, Projection, Ray, ReductionMode,
)

from vortrace import grid as gr
import numpy as np

_log = logging.getLogger("vortrace")


class ProjectionCloud:
    """Main interface for projections through an unstructured Voronoi mesh.

    Wraps the C++ ``Cvortrace`` backend to provide grid-based projections,
    arbitrary ray projections, and single-ray segment queries.  Supports
    multiple fields and reduction modes (sum/integrate, max, min).
    """

    _REDUCTION_MAP = {
        'integrate': ReductionMode.Sum,
        'sum': ReductionMode.Sum,
        'max': ReductionMode.Max,
        'min': ReductionMode.Min,
    }

    def __init__(self, pos, fields, boundbox=None, vol=None):
        """Create a ProjectionCloud from particle data.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Particle positions.
        fields : array_like, shape (N,) or (N, nfields)
            Scalar field(s) to integrate (e.g. density).
        boundbox : list of float, optional
            Bounding box ``[xmin, xmax, ymin, ymax, zmin, zmax]``.
            If *None*, derived from *pos*.
        vol : array_like of shape (N,), optional
            Cell volumes.  When provided, the C++ backend uses them
            for adaptive padding of the kDTree search radius.
        """
        # Store original data
        self.pos_orig = np.array(pos)
        self.fields_orig = np.array(fields)
        self.vol_orig = np.array(vol) if vol is not None else None

        fields_array = self.fields_orig
        # Determine number of fields
        if fields_array.ndim == 1:
            self._nfields = 1
        elif fields_array.ndim == 2:
            self._nfields = fields_array.shape[1]
        else:
            raise ValueError(
                "fields must be 1D (npart,) or 2D (npart, nfields)")

        if boundbox is None:
            boundbox = [
                pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(),
                pos[:, 1].max(), pos[:, 2].min(), pos[:, 2].max()
            ]

        self.boundbox = boundbox

        # C++ handles filtering internally
        self._cloud = PointCloud()
        if self.vol_orig is not None:
            self._cloud.loadPoints(self.pos_orig, self.fields_orig,
                                   boundbox, self.vol_orig)
        else:
            self._cloud.loadPoints(self.pos_orig, self.fields_orig,
                                   boundbox)
        self._cloud.buildTree()

        # Get filtered index mapping from C++
        self.orig_ids = np.array(self._cloud.get_orig_ids())

        _log.info("Applied bounding box filter: %d -> %d particles",
                  len(self.pos_orig), len(self.orig_ids))

    def _prepare_array_for_backend(self, arr):
        """Converts input to a C-contiguous float64 numpy array if not already.

        Args:
            arr: The input array-like object.

        Returns:
            np.ndarray: A C-contiguous, float64 numpy array.
        """
        np_arr = np.asarray(arr)
        if np_arr.dtype != np.float64 or not np_arr.flags['C_CONTIGUOUS']:
            np_arr = np.ascontiguousarray(np_arr, dtype=np.float64)
        return np_arr

    def grid_projection(self, extent, nres, bounds, center, *, proj=None,
                        yaw=0., pitch=0., roll=0., reduction='integrate'):
        """Make a grid projection through the point cloud.

        Args:
            extent: Spatial extent ``[min, max]`` or ``[[xmin,xmax],
                [ymin,ymax]]``.
            nres: Number of pixels (int for square, or ``(nx, ny)``).
            bounds: Integration bounds ``[z_start, z_end]``.
            center: Rotation center ``(x, y, z)`` or None.
            proj: Cartesian projection string (e.g. ``'xy'``).
            yaw: Yaw angle in radians.
            pitch: Pitch angle in radians.
            roll: Roll angle in radians.
            reduction: ``'integrate'``/``'sum'``, ``'max'``,
                or ``'min'``.

        Returns:
            ndarray: Shape ``(nres, nres)`` for single field, or
                ``(nres, nres, nfields)`` for multi-field.
        """
        reduction_mode = self._REDUCTION_MAP.get(reduction)
        if reduction_mode is None:
            raise ValueError(
                f"Unknown reduction {reduction!r}. "
                f"Use one of {list(self._REDUCTION_MAP)}")

        pos_start, pos_end = gr.generate_projection_grid(extent, nres, bounds,
                                                         center, proj=proj,
                                                         yaw=yaw, pitch=pitch,
                                                         roll=roll)

        # Flatten before feeding into backend.
        orig_shape = pos_start.shape
        pos_start = pos_start.reshape(-1, pos_start.shape[-1])
        pos_end = pos_end.reshape(-1, pos_end.shape[-1])

        # Actually do the projection using the Cvortrace backend.
        proj_obj = Projection(pos_start, pos_end)
        proj_obj.makeProjection(self._cloud, reduction_mode)
        dat = proj_obj.returnProjection()

        # Reshape before returning.
        if self._nfields == 1:
            dat = np.reshape(dat, orig_shape[:-1])
        else:
            dat = np.reshape(dat, (*orig_shape[:-1], self._nfields))
        return dat

    def projection(self, pos_start, pos_end, *, reduction='integrate'):
        """Make a projection through the point cloud.

        Args:
            pos_start (array of float): Starting points of the projection.
            pos_end (array of float): Ending points of the projection.
            reduction (str): Reduction mode: 'integrate'/'sum', 'max', 'min'.

        Returns:
            dat (array of float): The projection data. Shape ``(N,)`` when
                a single field was loaded, ``(N, nfields)`` otherwise.
        """
        reduction_mode = self._REDUCTION_MAP.get(reduction)
        if reduction_mode is None:
            raise ValueError(
                f"Unknown reduction {reduction!r}. "
                f"Use one of {list(self._REDUCTION_MAP)}")

        pos_start = self._prepare_array_for_backend(pos_start)
        pos_end = self._prepare_array_for_backend(pos_end)

        # ——— sanity‐check shape ———
        if (pos_start.ndim != 2 or pos_end.ndim != 2 or
                pos_start.shape != pos_end.shape or
                pos_start.shape[1] != 3):  # Ensure 3 columns for x,y,z
            raise ValueError('pos_start / pos_end must be 2D arrays of '
                             'identical shape (N,3)')

        # now safe to call into C++
        proj_obj = Projection(pos_start, pos_end)
        proj_obj.makeProjection(self._cloud, reduction_mode)
        return proj_obj.returnProjection()

    def single_projection(self, pos_start, pos_end, return_midpoint=True):
        """Perform projection for a single ray and return column density and
        per-segment info.

        Args:
            pos_start (array): shape (3,) or (1,3) start point.
            pos_end (array): shape (3,) or (1,3) end point.
            return_midpoint (bool, optional): if True, return midpoint of
                each segment.

        Returns:
            tuple: ``(dens, cell_ids, s_vals, ds_vals)``.
            When ``nfields == 1``, *dens* is a scalar (backward compatible).
            When ``nfields > 1``, *dens* is a 1-D array of length *nfields*.
        """
        # Allow lists/tuples as input, convert to numpy arrays for shape checks
        pos_start_np = np.asarray(pos_start)
        pos_end_np = np.asarray(pos_end)

        # Allow either 1D (3,) or 2D (1,3) inputs, and reshape 1D to 2D (1,3)
        if pos_start_np.ndim == 1 and pos_end_np.ndim == 1:
            if pos_start_np.shape != (3,) or pos_end_np.shape != (3,):
                raise ValueError('If 1D, pos_start and pos_end must both '
                                 'have shape (3,)')
            # Reshape to (1,3) for consistent processing
            pos_start_np = pos_start_np[np.newaxis, :]
            pos_end_np = pos_end_np[np.newaxis, :]
        elif pos_start_np.ndim == 2 and pos_end_np.ndim == 2:
            if pos_start_np.shape != (1, 3) or pos_end_np.shape != (1, 3):
                raise ValueError('If 2D, pos_start and pos_end must both '
                                 'have shape (1,3)')
        else:
            # Handles cases like one is 1D and other is 2D, or incorrect
            # ndims/shapes
            raise ValueError('pos_start and pos_end must both be 1D shape '
                             '(3,) or both 2D shape (1,3)')

        # Now that shape is (1,3), prepare for backend
        pos_start_c = self._prepare_array_for_backend(pos_start_np)
        pos_end_c = self._prepare_array_for_backend(pos_end_np)

        # extract single vectors for Ray constructor (expects 1D (3,) like
        # arrays)
        start_vec = pos_start_c[0]
        end_vec = pos_end_c[0]

        # compute using Ray
        ray = Ray(start_vec, end_vec)
        ray.integrate(self._cloud)

        col_vals = np.array(ray.get_col())  # length nfields
        segments = ray.get_segments()

        # unpack segment info into arrays
        cell_ids_raw = np.array([seg[0] for seg in segments], dtype=int)
        s_raw = np.array([seg[1] for seg in segments], dtype=np.float64)
        sedge_raw = np.array([seg[2] for seg in segments], dtype=np.float64)
        ds_raw = np.array([seg[3] for seg in segments], dtype=np.float64)
        # vectorized merge of duplicate cell_ids: first and last unique,
        # interior appear in consecutive pairs
        length = cell_ids_raw.size
        if length == 2:
            cell_ids = cell_ids_raw
            s_vals = s_raw
            smid_vals = sedge_raw
            ds_vals = ds_raw
        elif length > 2:
            mids = np.arange(1, length-1, 2)
            # assert matching start s for each duplicate pair
            if not np.allclose(s_raw[mids], s_raw[mids+1]):
                raise ValueError('mismatched s values in duplicate segments')

            # pick unique cell_ids
            cell_ids = np.concatenate((
                [cell_ids_raw[0]],
                cell_ids_raw[mids],
                [cell_ids_raw[-1]]
            ))

            # pick s values: first, one per pair, last
            s_vals = np.concatenate((
                [s_raw[0]],
                s_raw[mids],
                [s_raw[-1]]
            ))

            # compute smid values for each pair
            smid_vals = np.concatenate((
                [sedge_raw[0]/2.],
                (sedge_raw[mids] + sedge_raw[mids+1]) / 2.,
                [(sedge_raw[-1]+np.linalg.norm(pos_end-pos_start)) / 2.]
            ))

            # sum ds values across each duplicate pair
            ds_vals = np.concatenate((
                [ds_raw[0]],
                ds_raw[mids] + ds_raw[mids+1],
                [ds_raw[-1]]
            ))
        else:
            raise ValueError('pos_start and pos_end are in the same cell')

        # Validation: check that column values are consistent with segments
        # Map filtered cell_ids back to original indices for field lookup
        orig_cell_ids = self.orig_ids[cell_ids]
        if self._nfields == 1:
            field_vals = self.fields_orig[orig_cell_ids]
        else:
            field_vals = self.fields_orig[orig_cell_ids, 0]

        if not np.isclose(col_vals[0],
                          np.sum(field_vals * ds_vals)):
            raise ValueError(f"extracted ray cells and ds does not give "
                             f"consistent density: {col_vals[0]} != "
                             f"{np.sum(field_vals * ds_vals)}")

        # Return scalar dens for backward compat when nfields == 1
        dens_out = col_vals[0] if self._nfields == 1 else col_vals

        if return_midpoint:
            return dens_out, self.orig_ids[cell_ids], smid_vals, ds_vals
        else:
            return dens_out, self.orig_ids[cell_ids], s_vals, ds_vals

    # ------------------------------------------------------------------
    # Convenience I/O wrappers
    # ------------------------------------------------------------------

    def save(self, filename, *, fmt="npz"):
        """Save this cloud to disk.

        See :func:`vortrace.io.save_cloud` for details.
        """
        from vortrace.io import save_cloud
        save_cloud(filename, self, fmt=fmt)

    @classmethod
    def load(cls, filename):
        """Load a cloud from disk.

        See :func:`vortrace.io.load_cloud` for details.
        """
        from vortrace.io import load_cloud
        return load_cloud(filename)
