Degenerate Geometry
===================

The basic :doc:`algorithm <algorithm>` assumes that the split point ``p_s``
lands cleanly inside a single Voronoi cell. In rare cases, ``p_s`` can land
exactly on a vertex, edge, or face of the Voronoi mesh, making the
nearest-neighbor query ambiguous. This page describes how ``vortrace``
detects and resolves these degenerate cases.

Split-point classification
--------------------------

After computing the analytic split point between cells ``c`` and ``n``,
``vortrace`` queries the three nearest Voronoi generators (3-NN) at that
position and builds the **equidistant set** -- all mesh generating points whose squared
distance is within a tolerance of the nearest.

The classification checks whether ``c`` or ``n`` appears in this set:

- **If no** (Case 1): a third cell is strictly closer than ``c`` and ``n``.
  Insert it as an intermediate point and recurse into the two new
  sub-segments.

- **If yes** (Case 2): the split point lies on the ``c``/``n`` boundary.
  Accumulate both half-segments and advance along the ray. Even if other points
  are equidistant, we can rely on the convexity of the Voronoi mesh to ignore
  them.

Because the split point sits on the bisector of ``c`` and ``n`` by
construction, both are equidistant at that location. Any cell that appears
closer in the 3-NN result must lie between them.

High-valence vertices
~~~~~~~~~~~~~~~~~~~~~

At a vertex where many cells meet, the 3-NN may not return ``c`` or ``n``
even though they are equidistant -- other generators at the same distance
fill the top three slots. When neither ``c`` nor ``n`` appears in the
initial 3-NN equidistant set and all three cells are equidistant, the search
is expanded to 100-NN. If ``c`` or ``n`` is found in the expanded set, the
split point is classified as a boundary (Case 2).

Multiple equidistant intermediate cells (Case 1a)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When several non-``c``/``n`` cells are equidistant at the split point (e.g.
the ray crosses a Voronoi edge where three or more cells meet), the 3-NN
cannot unambiguously identify which cell lies on the ray. ``vortrace``
resolves this in two stages.

The algorithm finds the intermediate cell on **each side** independently:

**Stage 1 — perturb along the ray.**  The split point is perturbed by a small
epsilon backward along the ray (toward ``c``'s side) to find the cell just
before the split point (``sc``), and forward (toward ``n``'s side) to find the
cell just after (``sn``). Each perturbation is accepted only if the result is
unambiguous (a single clear nearest that is neither ``c`` nor ``n``).

**Stage 2 — perpendicular cycling (per-side fallback).**  If the along-ray
perturbation for a given side is ambiguous — which happens when the ray lies
along a Voronoi face on that side — the algorithm falls back to perpendicular
cycling *for that side only*. The other side may have already resolved via
Stage 1. The perpendicular directions have the form
``cross(dir, a*w1 + b*w2 + c*w3)`` for integer coefficients ``(a, b, c)``
and base vectors::

   w1 = (1, 1, 0) / sqrt(2)
   w2 = (0, 1, 1) / sqrt(2)
   w3 = (1, 0, 1) / sqrt(2)

The cross product with the ray direction keeps the perturbation perpendicular
to the ray, while the combinatorial cycling through coefficients ensures that
a sufficiently diverse set of directions is explored. The first perturbation
that lands in a cell other than ``c`` or ``n`` is accepted.

If ``sc == sn`` (the same cell on both sides), a single intermediate point is
inserted. If ``sc != sn``, both are inserted at the split-point position and
the zero-width segment between them is skipped during traversal. Specifically, the traversal list will look like ``current(c) → sc(s) → sn(s) → next(n)`` with sc and sn at the same point.

Parallel bisectors
------------------

When the ray runs exactly along a Voronoi face (e.g. a ray at ``x = 0.5``
between generators at ``x = 0`` and ``x = 1``), the bisector plane is
parallel to the ray and no analytic intersection exists.

In this case, ``findSplitPointDistance`` falls back to a **binary search**
along the current segment ``[s_lo, s_hi]``. At each step it queries the
nearest neighbor at the midpoint:

- If the midpoint lies in a third cell (neither ``c`` nor ``n``), that
  position is returned so that ``integrate()`` can insert the intermediate
  cell via the normal classification path.

- If the midpoint lies in ``c`` or ``n``, the interval is narrowed toward the
  boundary. Once the interval width drops below ``1e-12``, the converged
  boundary position is returned.

A warning is emitted to ``stderr`` whenever the parallel-bisector path is
triggered.

Endpoint clamping
-----------------

Floating-point coincidence can place the analytic split point exactly at a
segment endpoint (e.g. when the bisector passes through a RayPoint). To
avoid zero-width segments, the split point is **clamped** to the open
interior of the segment (offset inward by ``1e-15``). If the split point
falls outside the segment by more than the tolerance, a runtime error is
raised.
