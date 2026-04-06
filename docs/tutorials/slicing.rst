Slicing
=======

A slice extracts a 2D plane of interpolated field values at a constant depth
through the Voronoi mesh. Unlike a projection (which integrates along the
line of sight), a slice returns the field value of the cell at each pixel.

.. tab:: Python

   .. code-block:: python

      import vortrace as vt

      pc = vt.ProjectionCloud(pos, rho, boundbox=boundbox, vol=vol)

      # Slice: 256x256 pixels over [xmin, xmax, ymin, ymax] at z = depth
      data = pc.slice([0.1, 99.9, 0.1, 99.9], 256, depth=50.0)
      # data.shape is (256, 256) for a single field

   The low-level ``Cvortrace.Slice`` class is also available for direct use.

.. image:: /images/slicing.png
   :width: 80%
   :align: center
   :alt: Density slice at the box midplane

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
