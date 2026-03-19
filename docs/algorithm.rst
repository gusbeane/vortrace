Algorithm
=========

A one-dimensional integral through an unstructured mesh at first glance seems
very complicated, as one must wrangle with the complex geometry of a Voronoi
mesh.  However, one can use the definition of the mesh to simplify the
operation considerably.

``vortrace`` is a recursive algorithm that works by continually splitting the
integral until you are simply integrating between two points in neighbouring
cells, at which point the integral is trivial.  It starts by constructing a
kDTree using `nanoflann <https://github.com/jlblancoc/nanoflann>`_ to allow
for efficient nearest-neighbour searches.  Then:

1. Assume the two points you are trying to integrate between (``p_a`` and
   ``p_b``) are in neighbouring Voronoi cells.  Find those cells (``v_a`` and
   ``v_b``) quickly using the kDTree.

2. Using these four points, find the **split point** ``p_s`` -- the point on
   the line connecting ``p_a`` to ``p_b`` that intersects the face between
   cells ``v_a`` and ``v_b``.

3. Query the kDTree for the cell that ``p_s`` is in.

   a. If it is one of ``v_a`` or ``v_b``, you are done.  Return the integral
      as ``|p_a - p_s| * rho_a + |p_s - p_b| * rho_b``.

   b. Otherwise, recursively integrate ``p_a -- p_s`` and ``p_s -- p_b``.

This recursive splitting ensures that the number of kDTree queries scales
with the number of cell crossings, not with a fixed sampling resolution.

Periodic boundary conditions
----------------------------

``vortrace`` supports periodic boundary conditions with the ``periodic=True`` 
flag on :class:`~vortrace.ProjectionCloud`.

Periodic nearest-neighbour queries use a **ghost-image** approach.  When the
kDTree is queried for the nearest neighbour of a point **q**, the algorithm:

1. Queries the tree with the original point **q** and records the result.
2. For each of the 26 non-trivial shifts ``(dx, dy, dz)`` in ``{-1, 0, 1}^3``,
   it shifts **q** by ``(dx * Lx, dy * Ly, dz * Lz)`` (where *L* is the box
   size in each dimension) and queries the tree again.
3. Returns the closest result across all images.

An early-exit optimisation makes this efficient: an image is skipped if the
query point is farther from the relevant boundary than the current best
distance.  In practice, most queries only perform a single tree search; only
points near a box face trigger additional searches.
