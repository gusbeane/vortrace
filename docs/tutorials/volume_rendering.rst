Volume Rendering
================

Volume rendering produces an RGB image by compositing color and opacity
along each ray using emission-absorption integration.  Unlike a density
projection (which sums a quantity), volume rendering maps cell quantities
through a **transfer function** to ``(R, G, B, alpha)`` and applies
front-to-back compositing:

.. math::

   c(s) = \int_0^s c(s') \, \alpha(s') \, e^{-\tau(s')} \, ds',
   \qquad \tau(s) = \int_0^s \alpha(s') \, ds'

where :math:`\alpha` is the absorption coefficient per unit length and
:math:`\tau` is the optical depth.  The final color :math:`c(s)` is an
integral over the full ray length from `0` to `s`.

.. note::

   Unlike simple integrals (Sum, Max, Min), the order of the ray
   **start and end positions matters** for volume rendering.  The ray is
   composited front-to-back, so swapping start and end will produce a
   different image.

Defining a transfer function
-----------------------------

The transfer function maps a cell quantity (e.g. density) to four values:
red, green, blue, and absorption.  You define this yourself.

.. tab:: Python

   .. code-block:: python

      import numpy as np
      import matplotlib as mpl

      def gaussian(x, mean, sigma):
          return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
              -0.5 * ((x - mean) / sigma) ** 2
          )

      def transfer_function(q):
          """Map density to (R, G, B, alpha)."""
          logq = np.log10(q)
          logq_min = np.percentile(logq, 1)
          logq_max = np.percentile(logq, 99)

          # Map to a colormap
          norm_val = np.clip((logq - logq_min) / (logq_max - logq_min), 0, 1)
          rgba = mpl.colormaps["rainbow"](norm_val)

          # Gaussians for opacity
          sigma = 0.15 * (logq_max - logq_min)
          peaks = np.linspace(logq_min, logq_max, 4)
          logq_clipped = np.clip(logq, logq_min, logq_max)

          a = np.zeros(q.shape)
          for i, peak in enumerate(peaks):
              lognorm = -4.6 + (-1.2 + 4.6) / (peaks[-1] - peaks[0]) * (peak - peaks[0])
              a += 10 ** lognorm * gaussian(logq_clipped, peak, sigma)

          return rgba[:, 0], rgba[:, 1], rgba[:, 2], a

.. tab:: C++

   .. code-block:: cpp

      #include <cmath>
      #include <vector>

      // Apply a transfer function to density values.
      // Writes R, G, B, alpha into a flat fields array (4 fields per particle).
      void apply_transfer(const double* rho, size_t n, double* fields) {
          // Compute log-density range
          std::vector<double> logq(n);
          for (size_t i = 0; i < n; i++) logq[i] = std::log10(rho[i]);
          // ... determine logq_min, logq_max from percentiles ...

          for (size_t i = 0; i < n; i++) {
              double norm = (logq[i] - logq_min) / (logq_max - logq_min);
              norm = std::max(0.0, std::min(1.0, norm));

              // Map to RGB (implement your own colormap)
              fields[i * 4 + 0] = /* R */ ;
              fields[i * 4 + 1] = /* G */ ;
              fields[i * 4 + 2] = /* B */ ;

              // Opacity from Gaussians
              fields[i * 4 + 3] = /* alpha */ ;
          }
      }

Running the volume render
-------------------------

.. tab:: Python

   .. code-block:: python

      import vortrace as vt

      R, G, B, alpha = transfer_function(rho)
      fields_rgba = np.column_stack([R, G, B, alpha])

      pc_vol = vt.ProjectionCloud(
          pos, fields_rgba, vol=vol,
          boundbox=[0, BoxSize, 0, BoxSize, 0, BoxSize],
      )

      image = pc_vol.grid_projection(
          extent, npix, bounds, center=None, reduction="volume"
      )
      # image.shape == (256, 256, 3) -- an RGB image

.. tab:: C++

   .. code-block:: cpp

      #include <vortrace/vortrace.hpp>

      // fields is a flat array with 4 values per particle: R, G, B, alpha
      PointCloud cloud;
      cloud.loadPoints(pos, npart, fields, npart, 4, subbox);
      cloud.buildTree();

      Projection proj(starts, ends, ngrid);
      proj.makeProjection(cloud, ReductionMode::VolumeRender);

      const auto& data = proj.getProjectionData();
      // data has 3 values per ray (RGB): data[i * 3 + 0..2]

Displaying the result
---------------------

.. tab:: Python

   .. code-block:: python

      import matplotlib.pyplot as plt

      fig, ax = plt.subplots(figsize=(6, 6))
      img = image / image.max()  # normalize to [0, 1]
      ax.imshow(np.swapaxes(img, 0, 1), origin="lower",
                extent=[-L / 2, L / 2, -L / 2, L / 2])
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_title("Volume rendering")

.. image:: /images/volume_rendering.png
   :width: 70%
   :align: center
   :alt: Volume rendering of a galaxy interaction

.. note::

   Volume rendering requires exactly **4 input fields** interpreted as
   ``(R, G, B, alpha)``.  The output has 3 values per ray (RGB).

.. seealso::

   :doc:`/algorithm` for the full mathematical description of the
   compositing formula.
