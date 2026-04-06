Plotting
========

``vortrace`` provides two convenience plotting functions in the :mod:`vortrace.plot`
module. These are thin wrappers around matplotlib and are **Python-only**.

``plot_grid`` -- display a 2D projection
----------------------------------------

.. code-block:: python

   import vortrace as vt

   fig, ax, im = vt.plot.plot_grid(
       image,
       extent=[-L / 2, L / 2, -L / 2, L / 2],
       log=True,          # logarithmic color scale (default)
       cmap="inferno",    # colormap (default)
       colorbar=True,     # show colorbar (default)
       label="Column density",
   )
   ax.set_xlabel("x [kpc]")
   ax.set_ylabel("y [kpc]")

Pass an existing axes to embed in a multi-panel figure:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2)
   vt.plot.plot_grid(image_xy, ax=axes[0], label="xy")
   vt.plot.plot_grid(image_xz, ax=axes[1], label="xz")

``plot_ray`` -- plot a 1D ray profile
-------------------------------------

.. code-block:: python

   dens, cell_ids, s_vals, ds_vals = pc.single_projection(start, end)

   fig, ax = vt.plot.plot_ray(
       s_vals, rho[cell_ids],
       log=True,    # logarithmic y-axis (default)
   )
   ax.set_xlabel("Distance along ray")
   ax.set_ylabel("Density")

Both functions return ``(fig, ax)`` or ``(fig, ax, im)`` so you can
customize the plot further. Additional keyword arguments are forwarded to
``imshow`` (for ``plot_grid``) or ``plot`` (for ``plot_ray``).

.. note::

   ``matplotlib`` is imported lazily -- it is only required when you call
   a plot function, not when you ``import vortrace``.
