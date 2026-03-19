"""Main vortrace functions.

Main entry point into vortrace functions.

Example:
    Example placeholders.

Todo:
    * Add examples.

"""

import logging

try:
    # Preferred: C extension built into vortrace package
    from .Cvortrace import PointCloud, Projection, Ray, ReductionMode  # type: ignore
except ModuleNotFoundError:
    # Fallback: the extension was built at the top-level
    # (e.g. via scikit-build-core default)
    # In that case, try importing it as a top-level module and expose
    # the same names.
    from importlib import import_module

    _cmod = import_module("Cvortrace")
    PointCloud = _cmod.PointCloud  # type: ignore
    Projection = _cmod.Projection  # type: ignore
    Ray = _cmod.Ray  # type: ignore
    ReductionMode = _cmod.ReductionMode  # type: ignore

from vortrace import grid as gr
import numpy as np

_log = logging.getLogger("vortrace")


class ProjectionCloud:
    """Object for making projections through Voronoi mesh.

    Organizes simple wrappers around the underlying Cvortrace package, which
    does all the heavy lifting.

    Example:
        Example placeholders.

    Todo:
        * Add examples.

    """

    _REDUCTION_MAP = {
        'integrate': ReductionMode.Sum,
        'sum': ReductionMode.Sum,
        'max': ReductionMode.Max,
        'min': ReductionMode.Min,
    }

    def __init__(self, pos, fields, boundbox=None):
        # Store original data
        self.pos_orig = np.array(pos)
        self.fields_orig = np.array(fields)

        fields_array = np.array(fields)
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

        # Apply bounding box filtering in Python (matching C++ logic)
        # Pad by 15% of the box size in each dimension
        dx = boundbox[1] - boundbox[0]  # box size in x
        dy = boundbox[3] - boundbox[2]  # box size in y
        dz = boundbox[5] - boundbox[4]  # box size in z

        pad_frac = 0.15
        pad_x = pad_frac * dx
        pad_y = pad_frac * dy
        pad_z = pad_frac * dz

        xmin = boundbox[0] - pad_x
        xmax = boundbox[1] + pad_x
        ymin = boundbox[2] - pad_y
        ymax = boundbox[3] + pad_y
        zmin = boundbox[4] - pad_z
        zmax = boundbox[5] + pad_z

        # Find particles within the padded bounding box
        pos_array = np.array(pos)

        mask = ((pos_array[:, 0] >= xmin) & (pos_array[:, 0] <= xmax) &
                (pos_array[:, 1] >= ymin) & (pos_array[:, 1] <= ymax) &
                (pos_array[:, 2] >= zmin) & (pos_array[:, 2] <= zmax))

        npart_orig = fields_array.shape[0]

        # Apply the mask to filter particles
        self.pos = pos_array[mask]
        self.fields = fields_array[mask]
        self.orig_ids = np.arange(npart_orig)[mask]

        _log.info("Applied bounding box filter: %d -> %d particles",
                  npart_orig, self.fields.shape[0])

        self._cloud = PointCloud()
        self._cloud.loadPoints(self.pos, self.fields, boundbox)
        self._cloud.buildTree()

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
            pos_start        (array): shape (3,) or (1,3) start point
            pos_end          (array): shape (3,) or (1,3) end point
            return_midpoint (bool, optional): if True, return midpoint of each
                segment
        Returns:
            dens (float or ndarray), cell_ids (ndarray),
            s_vals (ndarray), ds_vals (ndarray)

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
        if self._nfields == 1:
            field_vals = self.fields[cell_ids]
        else:
            # fields is 2D: (npart, nfields); index first field for validation
            field_vals = self.fields[cell_ids, 0]

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
