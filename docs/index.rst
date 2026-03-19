vortrace
========

Fast one-dimensional integrals through unstructured Voronoi meshes.

One-dimensional integrals through Voronoi meshes can be expensive.  Because
the mesh is unstructured, it is not obvious a priori where a line intersects
the faces of which cells.  Brute-force methods that sample large numbers of
points struggle with systems that have a large dynamic range in cell size,
like cosmological simulations.

``vortrace`` performs these integrals with the fewest nearest-neighbour calls
possible.  With an optimised C++ backend and a user-friendly Python frontend,
it's easy to get started.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   QuickStart
   algorithm
   api
