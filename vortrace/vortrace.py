"""Main vortrace functions.

Main entry point into vortrace functions.

Example:
    Example placeholders.

Todo:
    * Add examples.

"""

from .Cvortrace import PointCloud, Projection, Ray
from vortrace import grid as gr
import numpy as np


class ProjectionCloud:
    """Object for making projections through Voronoi mesh.

    Organizes simple wrappers around the underlying Cvortrace package, which
    does all the heavy lifting.

    Example:
        Example placeholders.

    Todo:
        * Add examples.

"""

    def __init__(self, pos, dens, boundbox=None):
        self.pos = np.array(pos)
        self.dens = np.array(dens)

        if boundbox is None:
            boundbox = [
                pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(),
                pos[:, 1].max(), pos[:, 2].min(), pos[:, 2].max()
            ]

        self.boundbox = boundbox

        self._cloud = PointCloud()
        self._cloud.loadPoints(pos, dens, boundbox)
        self._cloud.buildTree()

    def grid_projection(self, extent, nres, bounds, center, *, proj=None,
                        yaw=0., pitch=0., roll=0.):

        pos_start, pos_end = gr.generate_projection_grid(extent, nres, bounds,
                                                         center, proj=proj,
                                                         yaw=yaw, pitch=pitch,
                                                         roll=roll)

        # Flatten before feeding into backend.
        # TODO: make C backend accept arbitrary shape?
        orig_shape = pos_start.shape
        pos_start = pos_start.reshape(-1, pos_start.shape[-1])
        pos_end = pos_end.reshape(-1, pos_end.shape[-1])

        # Actually do the projection using the Cvortrace bakend.
        proj = Projection(pos_start, pos_end)
        proj.makeProjection(self._cloud)
        dat = proj.returnProjection()

        # Reshape before returning.
        dat = np.reshape(dat, orig_shape[:-1])
        return dat

    def projection(self, pos_start, pos_end):
        """Make a projection through the point cloud.

        Args:
            pos_start (array of float): Starting points of the projection.
            pos_end (array of float): Ending points of the projection.

        Returns:
            dat (array of float): The projection data.
        """
        # ——— enforce numpy arrays ———
        pos_start = np.asarray(pos_start)
        pos_end   = np.asarray(pos_end)

        # ——— enforce correct dtype and C‑contiguity ———
        if pos_start.dtype != np.float64 or not pos_start.flags['C_CONTIGUOUS']:
            pos_start = np.ascontiguousarray(pos_start, dtype=np.float64)
        if pos_end.dtype != np.float64 or not pos_end.flags['C_CONTIGUOUS']:
            pos_end = np.ascontiguousarray(pos_end,   dtype=np.float64)

        # ——— sanity‐check shape ———
        if pos_start.ndim != 2 or pos_end.ndim != 2 \
            or pos_start.shape != pos_end.shape:
            raise ValueError('pos_start / pos_end must be 2D arrays\
                              of identical shape')

        # now safe to call into C++
        proj = Projection(pos_start, pos_end)
        proj.makeProjection(self._cloud)
        return proj.returnProjection()
    
    def single_projection(self, pos_start, pos_end):
        """Perform projection for a single ray and return column density and per-segment info.
        Args:
            pos_start (array): shape (1,3) start point
            pos_end   (array): shape (1,3) end point
        Returns:
            dens (float), cell_ids (ndarray), s_vals (ndarray), ds_vals (ndarray)
        """
        # enforce numpy arrays
        pos_start = np.asarray(pos_start)
        pos_end = np.asarray(pos_end)
        # allow either 1D (3,) or 2D (1,3) inputs
        if pos_start.ndim == 1 and pos_end.ndim == 1:
            if pos_start.shape != (3,) or pos_end.shape != (3,):
                raise ValueError('pos_start and pos_end must have shape (3,) or (1,3)')
            pos_start = pos_start[np.newaxis, :]
            pos_end = pos_end[np.newaxis, :]
        elif pos_start.ndim == 2 and pos_end.ndim == 2:
            if pos_start.shape != (1,3) or pos_end.shape != (1,3):
                raise ValueError('pos_start and pos_end must have shape (3,) or (1,3)')
        else:
            raise ValueError('pos_start and pos_end must have shape (3,) or (1,3)')

        # ——— enforce dtype and contiguity ———
        if pos_start.dtype != np.float64 or not pos_start.flags['C_CONTIGUOUS']:
            pos_start = np.ascontiguousarray(pos_start, dtype=np.float64)
        if pos_end.dtype != np.float64 or not pos_end.flags['C_CONTIGUOUS']:
            pos_end = np.ascontiguousarray(pos_end, dtype=np.float64)
        
        # extract single vectors
        start = pos_start[0]
        end = pos_end[0]
        
        # compute using Ray
        ray = Ray(start, end)
        ray.integrate(self._cloud)
        dens = ray.get_dens_col()
        segments = ray.get_segments()
        
        # unpack segment info into arrays
        cell_ids_raw = np.array([seg[0] for seg in segments], dtype=int)
        s_raw = np.array([seg[1] for seg in segments], dtype=np.float64)
        ds_raw = np.array([seg[2] for seg in segments], dtype=np.float64)
        # vectorized merge of duplicate cell_ids: first and last unique, interior appear in consecutive pairs
        L = cell_ids_raw.size
        if L <= 2:
            cell_ids = cell_ids_raw
            s_vals = s_raw
            ds_vals = ds_raw
        else:
            mids = np.arange(1, L-1, 2)
            # assert matching start s for each duplicate pair
            if not np.allclose(s_raw[mids], s_raw[mids+1]):
                raise ValueError("mismatched s values in duplicate segments")
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
            # sum ds values across each duplicate pair
            ds_vals = np.concatenate((
                [ds_raw[0]],
                ds_raw[mids] + ds_raw[mids+1],
                [ds_raw[-1]]
            ))

        if not np.isclose(dens, np.sum(self.dens[cell_ids]*ds_vals)):
            raise ValueError("extracted ray cells and ds does not give consistent density: {} != {}".format(
                dens, np.sum(self.dens[cell_ids]*ds_vals)))

        return dens, cell_ids, s_vals, ds_vals

