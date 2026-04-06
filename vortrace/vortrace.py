"""High-level interface for projections through Voronoi meshes.

Provides :class:`ProjectionCloud`, the main user-facing class that wraps
the C++ backend for grid projections, direct ray projections, and
single-ray segment queries.
"""

from __future__ import annotations

import logging

from numpy.typing import ArrayLike

from .Cvortrace import (  # type: ignore
    PointCloud, Projection, Ray, ReductionMode, Slice,
)

from vortrace import grid as gr
import numpy as np

_log = logging.getLogger("vortrace")


class ProjectionCloud:
    """Main interface for projections through an unstructured Voronoi mesh.

    Wraps the C++ ``Cvortrace`` backend to provide grid-based projections,
    arbitrary ray projections, and single-ray segment queries.  Supports
    multiple fields and reduction modes (sum/integrate, max, min, volume).
    """

    _REDUCTION_MAP = {
        'integrate': ReductionMode.Sum,
        'sum': ReductionMode.Sum,
        'max': ReductionMode.Max,
        'min': ReductionMode.Min,
        'volume': ReductionMode.VolumeRender,
    }

    def __init__(self, pos: ArrayLike, fields: ArrayLike,
                 boundbox: list[float] | None = None,
                 vol: ArrayLike | None = None, *,
                 periodic: bool = False,
                 _skip_build_tree: bool = False) -> None:
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
        periodic : bool, optional
            If *True*, enable periodic boundary conditions.  The
            bounding box defines the periodic domain and no spatial
            filtering is applied (all particles are loaded).
        """
        # Store original data
        self.pos_orig = np.array(pos)
        self.fields_orig = np.array(fields)
        self.vol_orig = np.array(vol) if vol is not None else None
        self.periodic = periodic

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
                                   boundbox, self.vol_orig, periodic)
        else:
            self._cloud.loadPoints(self.pos_orig, self.fields_orig,
                                   boundbox, periodic=periodic)
        if not _skip_build_tree:
            self._cloud.buildTree()

        # Get filtered index mapping from C++
        self.orig_ids = np.array(self._cloud.get_orig_ids())

        _log.info("Applied bounding box filter: %d -> %d particles",
                  len(self.pos_orig), len(self.orig_ids))

    def _validate_reduction(self, reduction: str) -> ReductionMode:
        """Validate reduction mode and return the corresponding enum.

        Raises ValueError for unknown modes or invalid field configuration.
        """
        mode = self._REDUCTION_MAP.get(reduction)
        if mode is None:
            raise ValueError(
                f"Unknown reduction {reduction!r}. "
                f"Use one of {list(self._REDUCTION_MAP)}")
        if mode == ReductionMode.VolumeRender:
            if self._nfields != 4:
                raise ValueError(
                    "Volume rendering requires exactly 4 fields "
                    f"(R, G, B, alpha), got {self._nfields}")
            if not self._cloud.get_valid_rgba():
                raise ValueError(
                    "Volume rendering requires R, G, B values in [0, 1] "
                    "and alpha >= 0")
        return mode

    def _prepare_array_for_backend(self, arr: ArrayLike) -> np.ndarray:
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

    def grid_projection(self, extent: ArrayLike,
                        nres: int | tuple[int, int],
                        bounds: ArrayLike,
                        center: ArrayLike | None, *,
                        proj: str | None = None,
                        yaw: float = 0., pitch: float = 0.,
                        roll: float = 0.,
                        reduction: str = 'integrate') -> np.ndarray:
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
                ``'min'``, or ``'volume'``.  Volume rendering
                requires exactly 4 fields (R, G, B, alpha) and
                returns 3 output channels (RGB).

        Returns:
            ndarray: Shape ``(nres, nres)`` for single field,
                ``(nres, nres, nfields)`` for multi-field, or
                ``(nres, nres, 3)`` for volume rendering.
        """
        reduction_mode = self._validate_reduction(reduction)

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
        out_nfields = (3 if reduction_mode == ReductionMode.VolumeRender
                       else self._nfields)
        if out_nfields == 1:
            dat = np.reshape(dat, orig_shape[:-1])
        else:
            dat = np.reshape(dat, (*orig_shape[:-1], out_nfields))
        return dat

    def projection(self, pos_start: ArrayLike, pos_end: ArrayLike, *,
                   reduction: str = 'integrate') -> np.ndarray:
        """Make a projection through the point cloud.

        Args:
            pos_start (array of float): Starting points of the projection.
            pos_end (array of float): Ending points of the projection.
            reduction (str): Reduction mode: 'integrate'/'sum', 'max', 'min',
                or 'volume'. Volume rendering requires 4 fields (R, G, B,
                alpha) and returns 3 channels (RGB).

        Returns:
            dat (array of float): The projection data. Shape ``(N,)`` when
                a single field was loaded, ``(N, nfields)`` otherwise.
        """
        reduction_mode = self._validate_reduction(reduction)

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

    def single_projection(self, pos_start: ArrayLike, pos_end: ArrayLike,
                           return_midpoint: bool = True, *,
                           reduction: str = 'integrate') -> tuple:
        """Perform projection for a single ray and return column density and
        per-segment info.

        Args:
            pos_start (array): shape (3,) or (1,3) start point.
            pos_end (array): shape (3,) or (1,3) end point.
            return_midpoint (bool, optional): if True, return midpoint of
                each segment.
            reduction (str): Reduction mode: ``'integrate'``/``'sum'``,
                ``'max'``, or ``'min'``.

        Returns:
            tuple: ``(value, cell_ids, s_vals, ds_vals)``.
            When ``nfields == 1``, *value* is a scalar (backward compatible).
            When ``nfields > 1``, *value* is a 1-D array of length *nfields*.
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

        reduction_mode = self._validate_reduction(reduction)

        # compute using Ray — walk then reduce
        ray = Ray(start_vec, end_vec)
        ray.integrate(self._cloud, reduction_mode)

        if reduction_mode == ReductionMode.Max:
            col_vals = np.array(ray.get_max_val())
        elif reduction_mode == ReductionMode.Min:
            col_vals = np.array(ray.get_min_val())
        elif reduction_mode == ReductionMode.VolumeRender:
            col_vals = np.array(ray.get_vol_render_val())
        else:
            col_vals = np.array(ray.get_col())  # length nfields

        # Segments are already merged (one per cell, in order)
        segments = ray.get_segments()
        cell_ids = np.array([seg[0] for seg in segments], dtype=int)
        s_enter = np.array([seg[1] for seg in segments], dtype=np.float64)
        s_exit = np.array([seg[2] for seg in segments], dtype=np.float64)
        ds_vals = np.array([seg[3] for seg in segments], dtype=np.float64)
        smid_vals = (s_enter + s_exit) / 2.0

        # Validation: check that column values are consistent with segments
        # (only meaningful for Sum/integrate mode)
        orig_cell_ids = self.orig_ids[cell_ids]
        if reduction_mode == ReductionMode.Sum:
            for f_idx in range(self._nfields):
                if self._nfields == 1:
                    field_vals = self.fields_orig[orig_cell_ids]
                else:
                    field_vals = self.fields_orig[orig_cell_ids, f_idx]
                computed = np.sum(field_vals * ds_vals)
                if not np.isclose(col_vals[f_idx], computed):
                    raise ValueError(
                        f"Field {f_idx}: extracted ray cells and ds does "
                        f"not give consistent density: "
                        f"{col_vals[f_idx]} != {computed}")

        # VolumeRender always returns a 3-element RGB array.
        # Scalar dens for backward compat when nfields == 1.
        if reduction_mode == ReductionMode.VolumeRender:
            dens_out = col_vals
        elif self._nfields == 1:
            dens_out = col_vals[0]
        else:
            dens_out = col_vals

        if return_midpoint:
            return dens_out, self.orig_ids[cell_ids], smid_vals, ds_vals
        else:
            return dens_out, self.orig_ids[cell_ids], s_enter, ds_vals

    def slice(self, extent: ArrayLike,
              nres: int | tuple[int, int],
              depth: float) -> np.ndarray:
        """Extract a 2D slice at constant depth through the point cloud.

        Unlike a projection (which integrates along the line of sight), a
        slice returns the nearest-cell field value at each pixel position.

        Args:
            extent: Spatial extent ``[xmin, xmax, ymin, ymax]``.
            nres: Number of pixels (int for square, or ``(nx, ny)``).
            depth: The z-coordinate of the slicing plane.

        Returns:
            ndarray: Shape ``(nx, ny)`` for single field,
                ``(nx, ny, nfields)`` for multi-field.
        """
        if isinstance(nres, int):
            npix = (nres, nres)
        else:
            npix = tuple(nres)

        extent_arr = np.asarray(extent, dtype=np.float64).ravel()
        if extent_arr.size != 4:
            raise ValueError(
                "extent must have 4 elements [xmin, xmax, ymin, ymax]")

        sl = Slice(list(npix), list(extent_arr), float(depth))
        sl.makeSlice(self._cloud)
        dat = sl.returnSlice()

        if self._nfields == 1:
            dat = dat.reshape(npix[0], npix[1])
        else:
            dat = dat.reshape(npix[0], npix[1], self._nfields)

        return dat

    # ------------------------------------------------------------------
    # Convenience I/O wrappers
    # ------------------------------------------------------------------

    def save(self, filename: str, *, fmt: str = "npz",
             save_tree: bool = True) -> None:
        """Save this cloud to disk.

        See :func:`vortrace.io.save_cloud` for details.
        """
        from vortrace.io import save_cloud
        save_cloud(filename, self, fmt=fmt, save_tree=save_tree)

    @classmethod
    def load(cls, filename: str) -> ProjectionCloud:
        """Load a cloud from disk.

        See :func:`vortrace.io.load_cloud` for details.
        """
        from vortrace.io import load_cloud
        return load_cloud(filename)
