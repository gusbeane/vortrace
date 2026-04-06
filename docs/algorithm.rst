Algorithm
=========

A one-dimensional integral through an unstructured mesh at first glance seems
very complicated, as one must wrangle with the complex geometry of a Voronoi
mesh. However, one can use the definition of the mesh to simplify the
operation considerably.

The algorithm proceeds in two phases: **walking** and **reduction**.

Ray walking
-----------

The walk phase traces a ray through the mesh and records the ordered list of
cell crossings as **segments** (cell ID, entry distance, exit distance). It
uses a recursive split-point algorithm that minimizes kDTree queries.

``vortrace`` starts by constructing a kDTree using
`nanoflann <https://github.com/jlblancoc/nanoflann>`_ to allow for efficient
nearest-neighbor searches. Then, for a ray from ``p_a`` to ``p_b``:

1. Assume ``p_a`` and ``p_b`` are in neighbouring Voronoi cells. Find the mesh generating points of those cells (``v_a`` and ``v_b``) quickly using the kDTree.

2. Using these four points, find the **split point** ``p_s`` -- the point on
   the line connecting ``p_a`` to ``p_b`` that intersects the face between
   cells ``v_a`` and ``v_b``.

3. Query the kDTree for the cell that ``p_s`` is in.

   a. If it is one of ``v_a`` or ``v_b``, the boundary has been found.
      Record the two segments (``p_a -- p_s`` through ``v_a`` and
      ``p_s -- p_b`` through ``v_b``) and move on.

   b. Otherwise, a third cell lies between them. Insert ``p_s`` and
      recursively resolve ``p_a -- p_s`` and ``p_s -- p_b``.

This recursive splitting ensures that the number of kDTree queries scales
with the number of cell crossings, not with a fixed sampling resolution.
The result is an ordered list of all ray-cell crossings and the intersection widths.

.. note::
   In periodic mode, if a ray starts and ends at the box edge and is aligned with the box's axes, the start and end points will be in the same cell. So, when `periodic=True`, the ray is always bisected at the start of the walk.

Reduction modes
---------------

After walking a ray through the mesh, ``vortrace`` applies a *reduction* function to the list of segments to produce a single result per ray. Four modes are available:

**Sum** (``reduction='integrate'`` or ``'sum'``)
   Weighted integral: for each field *f*, accumulate
   ``ds * field[cell, f]`` over all segments. Returns one value per field.

**Max / Min** (``reduction='max'`` / ``'min'``)
   Track the maximum or minimum field value encountered along the ray (no
   distance weighting). Returns one value per field.

**Volume rendering** (``reduction='volume'``)
   Front-to-back emission-absorption compositing. Requires exactly four
   input fields interpreted as (R, G, B, alpha), where alpha is the
   absorption coefficient per unit length. For each segment:

   .. math::

      C_c \mathrel{+}= T \cdot \text{field}_c \cdot (1 - e^{-\alpha \, ds}),
      \qquad T \mathrel{*}= e^{-\alpha \, ds}

   where *T* is the transmittance (initially 1). Returns three values (RGB)
   per ray. The user defines a transfer function in Python that maps cell
   quantities to the (R, G, B, alpha) fields. These four fields are then passed to
   ``ProjectionCloud``.
