Quick Start
===========

This page shows the simplest possible projection in both Python and C++.  For
more detailed examples, see the :doc:`tutorials/index`.

Load data and project
---------------------

.. tab:: Python

   .. code-block:: python

      import numpy as np
      import h5py as h5
      import vortrace as vt

      # Load an Arepo snapshot
      f = h5.File("snapshot.hdf5", "r")
      pos = np.array(f["PartType0/Coordinates"])
      rho = np.array(f["PartType0/Density"])
      mass = np.array(f["PartType0/Masses"])
      vol = mass / rho
      BoxSize = f["Parameters"].attrs["BoxSize"]
      f.close()

      # Create a projection cloud
      pc = vt.ProjectionCloud(
          pos, rho, vol=vol,
          boundbox=[0, BoxSize, 0, BoxSize, 0, BoxSize],
      )

      # Make a grid projection
      L = 75.0
      extent = [BoxSize / 2 - L / 2, BoxSize / 2 + L / 2]
      bounds = [0, BoxSize]
      npix = 256

      image = pc.grid_projection(extent, npix, bounds, center=None)

      # Plot the result
      fig, ax, im = vt.plot.plot_grid(
          image,
          extent=[-L / 2, L / 2, -L / 2, L / 2],
          label="Column density",
      )

.. tab:: C++

   .. code-block:: cpp

      #include <vortrace/vortrace.hpp>
      #include <vector>
      #include <cmath>

      int main() {
          // --- Load your particle data into flat arrays ---
          // pos:    [x0, y0, z0, x1, y1, z1, ...]
          // fields: [rho0, rho1, ...]
          size_t npart = /* number of particles */;
          std::vector<double> pos(3 * npart);
          std::vector<double> fields(npart);
          // ... fill pos and fields from your data source ...

          double BoxSize = /* your box size */;

          // Build a point cloud and kD-tree
          std::array<double, 6> subbox = {0, BoxSize, 0, BoxSize, 0, BoxSize};
          PointCloud cloud;
          cloud.loadPoints(pos.data(), npart, fields.data(), npart, 1, subbox);
          cloud.buildTree();

          // Set up a grid of rays
          double L = 75.0;
          double cx = BoxSize / 2, cy = BoxSize / 2;
          size_t npix = 256;
          double dx = L / npix;

          std::vector<Float> starts(3 * npix * npix);
          std::vector<Float> ends(3 * npix * npix);

          for (size_t iy = 0; iy < npix; iy++) {
              for (size_t ix = 0; ix < npix; ix++) {
                  size_t idx = 3 * (iy * npix + ix);
                  double x = cx - L / 2 + (ix + 0.5) * dx;
                  double y = cy - L / 2 + (iy + 0.5) * dx;
                  starts[idx]     = x;
                  starts[idx + 1] = y;
                  starts[idx + 2] = 0.0;
                  ends[idx]       = x;
                  ends[idx + 1]   = y;
                  ends[idx + 2]   = BoxSize;
              }
          }

          // Run the projection
          Projection proj(starts.data(), ends.data(), npix * npix);
          proj.makeProjection(cloud, ReductionMode::Sum);

          // Access the result
          const auto& data = proj.getProjectionData();
          // data[i] is the integrated column density for ray i
      }

Result
------

.. image:: /images/quickstart.png
   :width: 80%
   :align: center
   :alt: Column density projection

