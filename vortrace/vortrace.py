"""Main vortrace functions.

Main entry point into vortrace functions.

Example:
    Example placeholders.

Todo:
    * Add examples.

"""

import Cvortrace
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
        pos = np.array(pos)
        dens = np.array(dens)

        if boundbox is None:
            boundbox = [
                pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(),
                pos[:, 1].max(), pos[:, 2].min(), pos[:, 2].max()
            ]

        self.boundbox = boundbox

        self._cloud = Cvortrace.PointCloud()
        self._cloud.loadPoints(pos, dens, boundbox)
        self._cloud.buildTree()

    def projection(self,
                   extent,
                   nres,
                   bounds,
                   center,
                   proj=None,
                   yaw=0.,
                   pitch=0.,
                   roll=0.):

        pos_start, pos_end = gr.generate_projection_grid(extent,
                                                         nres,
                                                         bounds,
                                                         center,
                                                         proj=proj,
                                                         yaw=yaw,
                                                         pitch=pitch,
                                                         roll=roll)

        # Flatten before feeding into backend.
        # TODO: make C backend accept arbitrary shape?
        orig_shape = pos_start.shape
        pos_start = pos_start.reshape(-1, pos_start.shape[-1])
        pos_end = pos_end.reshape(-1, pos_end.shape[-1])

        # Actually do the projection using the Cvortrace bakend.
        proj = Cvortrace.Projection(pos_start, pos_end)
        proj.makeProjection(self._cloud)
        dat = proj.returnProjection()

        # Reshape before returning.
        dat = np.reshape(dat, orig_shape[:-1])
        return dat
