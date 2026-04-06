Custom Rotations
================

``vortrace`` supports arbitrary viewing angles using Tait-Bryan Euler angles
(yaw, pitch, roll) for grid projections.

Projection planes
-----------------

The ``proj`` parameter in Python provides shortcuts for the six Cartesian
projection planes:

.. list-table::
   :header-rows: 1

   * - ``proj``
     - Horizontal axis
     - Vertical axis
     - Line of sight
   * - ``'xy'`` (default)
     - x
     - y
     - z
   * - ``'xz'``
     - x
     - z
     - y
   * - ``'yz'``
     - y
     - z
     - x
   * - ``'yx'``
     - y
     - x
     - z
   * - ``'zx'``
     - z
     - x
     - y
   * - ``'zy'``
     - z
     - y
     - x

Tait-Bryan angles
-----------------

For viewing angles that don't align with the coordinate axes, use the
``yaw``, ``pitch``, and ``roll`` parameters. These are applied as
extrinsic rotations about the z, y, and x axes respectively, centered on
the ``center`` point.

.. tab:: Python

   .. code-block:: python

      import vortrace as vt

      center = [BoxSize / 2, BoxSize / 2, BoxSize / 2]

      # 30-degree yaw rotation
      image = pc.grid_projection(
          extent, npix, bounds, center,
          yaw=0.5236,  # radians
      )

      # Combined rotation
      image = pc.grid_projection(
          extent, npix, bounds, center,
          yaw=0.3, pitch=0.1, roll=0.0,
      )

.. tab:: C++

   In C++, rotations are handled at the ray-grid level. Build your own
   rotated start/end arrays and pass them to ``Projection``.

   .. code-block:: cpp

      #include <vortrace/vortrace.hpp>
      #include <cmath>

      // Rotate a point (x, y, z) by yaw angle around the z-axis
      // centered on (cx, cy, cz)
      void rotate_yaw(double& x, double& y, double z,
                      double cx, double cy, double yaw) {
          double dx = x - cx, dy = y - cy;
          x = cx + dx * std::cos(yaw) - dy * std::sin(yaw);
          y = cy + dx * std::sin(yaw) + dy * std::cos(yaw);
      }

      // Build rotated ray grids and pass to Projection as usual
      // (see the grid_projection tutorial for the grid setup loop)

   For the full Tait-Bryan rotation implementation, see the Python source
   in ``vortrace/grid.py``.

.. note::

   Angles are in **radians**. The ``center`` parameter must be set when
   using rotations -- it defines the point around which the grid is rotated.
