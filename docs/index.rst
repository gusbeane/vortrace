vortrace
========

Fast one-dimensional integrals through unstructured Voronoi meshes.

One-dimensional integrals through Voronoi meshes can be expensive. Because
the mesh is unstructured, it is not obvious a priori where a line intersects
the faces of which cells. Brute-force methods that sample large numbers of
points struggle with systems that have a large dynamic range in cell size,
like cosmological simulations.

``vortrace`` performs these integrals with the fewest nearest-neighbor calls
possible. With an optimized C++ backend and a user-friendly Python frontend,
it's easy to get started:

.. code-block:: python

   import vortrace as vt

   pc = vt.ProjectionCloud(pos, density)
   image = pc.grid_projection(extent, npix, bounds, center)

Features
--------

- **Fast ray tracing** -- recursive split-point algorithm minimizes kDTree queries
- **Multiple reduction modes** -- integrate, max, min, and volume rendering
- **Python and C++ APIs** -- thin Python wrapper over a standalone C++ library
- **Multi-field support** -- project multiple scalar fields simultaneously
- **Custom rotations** -- Tait-Bryan yaw/pitch/roll for arbitrary viewing angles
- **Periodic boundaries** -- full support for cosmological simulation boxes

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   algorithm
   edgecases

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   changelog
   contributing
