Periodic Boundary Conditions
============================

``vortrace`` supports periodic boundary conditions for cosmological simulation
boxes.  When enabled, the nearest-neighbor search checks periodic images
of each query point, ensuring correct projections across box boundaries.

.. note::

   Periodic integrals are assumed to lie entirely within a single periodic
   image of the box.  If your ray spans multiple periodic images, you must
   split it into segments that each fit within one box image.

Usage
-----

.. tab:: Python

   .. code-block:: python

      import numpy as np
      import vortrace as vt

      pc = vt.ProjectionCloud(
          pos, rho,
          boundbox=[0, BoxSize, 0, BoxSize, 0, BoxSize],
          periodic=True,
      )

      image = pc.grid_projection(
          extent=[0, BoxSize],
          nres=256,
          bounds=[0, BoxSize],
          center=None,
      )

      fig, ax, im = vt.plot.plot_grid(
          image,
          extent=[0, BoxSize, 0, BoxSize],
          label=r"$\Sigma$",
      )

.. tab:: C++

   .. code-block:: cpp

      #include <vortrace/vortrace.hpp>

      std::array<double, 6> subbox = {0, BoxSize, 0, BoxSize, 0, BoxSize};

      PointCloud cloud;
      cloud.loadPoints(
          pos, npart, fields, npart, 1, subbox,
          nullptr, 0,   // no cell volumes
          true          // periodic = true
      );
      cloud.buildTree();

      // Projections work as usual -- periodicity is handled internally
      Projection proj(starts, ends, ngrid);
      proj.makeProjection(cloud, ReductionMode::Sum);

.. image:: /images/periodic.png
   :width: 80%
   :align: center
   :alt: Cosmological box projection with periodic boundaries

Behavior differences
---------------------

When ``periodic=True``:

- The bounding box defines the periodic domain. The kD-tree wraps queries across boundaries.
- **No spatial filtering** is applied -- all particles are used regardless
  of the bounding box.  (In non-periodic mode, particles outside the padded
  subbox are discarded to reduce computation.)  If you need to reduce the
  number of particles for performance, filter them yourself before passing
  them to ``vortrace``.

.. note::

   The bounding box must match the simulation box for periodic mode to
   produce correct results.
