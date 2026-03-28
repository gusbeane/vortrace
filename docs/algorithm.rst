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

Reduction modes
---------------

After walking a ray through the mesh, ``vortrace`` produces an ordered list of
**segments** -- each recording which cell was crossed and the path length
through it.  A *reduction* function then combines these segments into a single
result per ray.  Four modes are available:

**Sum** (``reduction='integrate'`` or ``'sum'``)
   Weighted integral: for each field *f*, accumulate
   ``ds * field[cell, f]`` over all segments.  Returns one value per field.

**Max / Min** (``reduction='max'`` / ``'min'``)
   Track the maximum or minimum field value encountered along the ray (no
   distance weighting).  Returns one value per field.

**Volume rendering** (``reduction='volume'``)
   Front-to-back emission-absorption compositing.  Requires exactly four
   input fields interpreted as (R, G, B, alpha), where alpha is the
   absorption coefficient per unit length.  For each segment:

   .. math::

      C_c \mathrel{+}= T \cdot \text{field}_c \cdot (1 - e^{-\alpha \, ds}),
      \qquad T \mathrel{*}= e^{-\alpha \, ds}

   where *T* is the transmittance (initially 1).  Returns three values (RGB)
   per ray.  The user defines a transfer function in Python that maps cell
   quantities to the (R, G, B, alpha) fields before passing them to
   ``ProjectionCloud``.
