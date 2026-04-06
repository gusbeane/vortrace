Slicing
=======

A slice extracts a 2D plane of interpolated field values at a constant depth
through the Voronoi mesh. Unlike a projection (which integrates along the
line of sight), a slice returns the field value of the cell at each pixel.

.. tab:: Python

   The ``Slice`` class is available through the low-level C++ bindings. It
   is not yet wrapped in ``ProjectionCloud``.

   .. code-block:: python

      from vortrace.Cvortrace import PointCloud, Slice
      import numpy as np

      # Build the point cloud
      cloud = PointCloud()
      cloud.loadPoints(pos, rho)
      cloud.buildTree()

      # Define the slice: 256x256 pixels over [xmin, xmax, ymin, ymax] at z = depth
      npix = (256, 256)
      extent = (0.1, 99.9, 0.1, 99.9)
      depth = 50.0

      sl = Slice(npix, extent, depth)
      sl.makeSlice(cloud)

      data = sl.getSliceData()  # flat array of size 256*256*nfields

.. tab:: C++

   .. code-block:: cpp

      #include <vortrace/vortrace.hpp>

      PointCloud cloud;
      cloud.loadPoints(pos, npart, fields, npart, 1, subbox);
      cloud.buildTree();

      std::array<size_t, 2> npix = {256, 256};
      std::array<Float, 4> extent = {0.1, 99.9, 0.1, 99.9};
      Float depth = 50.0;

      Slice slice(npix, extent, depth);
      slice.makeSlice(cloud);

      const auto& data = slice.getSliceData();
      // data[iy * npix_x * nfields + ix * nfields + f]

Saving a slice to file
----------------------

The C++ ``Slice`` class provides a built-in method to save the slice data:

.. code-block:: cpp

   slice.saveSlice("slice_output.dat");

For Python, save the result using NumPy or the :mod:`vortrace.io` module.

.. seealso::

   :doc:`grid_projection` for projections that integrate along the line
   of sight.
