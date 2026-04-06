C++ API
=======

All public headers live in ``include/``. Include
``<vortrace/vortrace.hpp>`` for the full API, or include individual
headers as needed.

CMake target: ``vortrace::vortrace_core``

Types
-----

Defined in ``mytypes.hpp``.

.. cpp:type:: Float = double

   Floating-point type used throughout the library.

.. cpp:type:: Point = std::array<Float, 3>

   A 3D point.

.. cpp:enum-class:: ReductionMode

   .. cpp:enumerator:: Sum = 0

      Weighted integral: accumulate ``field * ds`` over segments.

   .. cpp:enumerator:: Max = 1

      Maximum field value along the ray.

   .. cpp:enumerator:: Min = 2

      Minimum field value along the ray.

   .. cpp:enumerator:: VolumeRender = 3

      Front-to-back emission-absorption compositing. Requires 4 input
      fields (R, G, B, alpha). Returns 3 output values (RGB).

Verbose and warnings
^^^^^^^^^^^^^^^^^^^^

.. cpp:member:: std::atomic<bool> vortrace::verbose

   Runtime flag to enable verbose output. Default: ``false``.

.. cpp:type:: vortrace::WarningCallback = std::function<void(const std::string&)>

   Callback type for warnings.

.. cpp:member:: vortrace::WarningCallback vortrace::warning_handler

   The active warning callback. Default: prints to ``stderr``.

.. cpp:function:: void vortrace::warn(const std::string& msg)

   Emit a warning through the current handler.


PointCloud
----------

Defined in ``pointcloud.hpp``. Manages particle data and a nanoflann
kD-tree for nearest-neighbor queries.

.. cpp:class:: PointCloud

   .. cpp:function:: void loadPoints(const double* pos, size_t npart, \
                                     const double* fields, size_t npart_fields, \
                                     size_t nfields_in, \
                                     const std::array<Float,6>& subbox, \
                                     const double* vol = nullptr, size_t nvol = 0, \
                                     bool periodic = false)

      Load particle positions and field data.

      :param pos: Flat array of positions ``[x0,y0,z0, x1,y1,z1, ...]``.
      :param npart: Number of particles.
      :param fields: Flat row-major field array ``[f0_0, f0_1, ..., f1_0, ...]``.
      :param npart_fields: Number of particles in the fields array (must equal *npart*).
      :param nfields_in: Number of scalar fields per particle.
      :param subbox: Bounding box ``{xmin, xmax, ymin, ymax, zmin, zmax}``.
      :param vol: Optional per-cell volumes for adaptive padding.
      :param nvol: Length of *vol* array.
      :param periodic: Enable periodic boundary conditions.

   .. cpp:function:: void buildTree()

      Build the kD-tree. Must be called after ``loadPoints``.

   .. cpp:function:: size_t queryTree(const Point& query_pt) const

      Find the nearest particle to *query_pt*.

   .. cpp:function:: void queryTreeK(const Point& query_pt, size_t k, \
                                     size_t* results, Float* r2) const

      Find the *k* nearest particles.

   .. cpp:function:: size_t get_npart() const

      Number of particles after filtering.

   .. cpp:function:: size_t get_nfields() const

      Number of fields per particle.

   .. cpp:function:: Point get_pt(size_t idx) const

      Position of particle *idx*.

   .. cpp:function:: Float get_field(size_t idx, size_t f) const

      Field *f* of particle *idx*.

   .. cpp:function:: bool get_tree_built() const

      Whether the kD-tree has been built.

   .. cpp:function:: bool get_periodic() const

      Whether periodic mode is enabled.

   .. cpp:function:: const std::vector<size_t>& get_orig_ids() const

      Mapping from filtered particle indices to original indices.

   .. cpp:function:: std::array<Float,6> get_subbox() const

      The active bounding box.

   .. cpp:function:: double get_pad() const

      The padding added to the subbox.


Ray
---

Defined in ``ray.hpp``. Traces a single ray through the Voronoi mesh.

.. cpp:class:: Ray

   .. cpp:struct:: Segment

      A single cell crossing along the ray.

      .. cpp:member:: size_t cell_id

         Particle index of the Voronoi cell.

      .. cpp:member:: Float s_enter

         Distance along the ray where the cell begins.

      .. cpp:member:: Float s_exit

         Distance along the ray where the cell ends.

      .. cpp:function:: Float ds() const

         Path length through the cell (``s_exit - s_enter``).

   .. cpp:function:: Ray(const Point& start, const Point& end)

      Construct a ray from *start* to *end*.

   .. cpp:function:: const std::vector<Segment>& walk(const PointCloud& cloud)

      Trace the ray through *cloud* and return the ordered list of segments.

   .. cpp:function:: void integrate(const PointCloud& cloud, \
                                    ReductionMode mode = ReductionMode::Sum)

      Walk the ray and apply a reduction. Results are stored internally
      and accessed via the getter methods below.

   .. cpp:function:: Float findSplitPointDistance(const Point& pos1, const Point& pos2)

      Analytic bisector-ray intersection distance.

   .. cpp:function:: const std::vector<Float>& get_col() const

      Column values after ``Sum`` reduction.

   .. cpp:function:: const std::vector<Float>& get_max_val() const

      Maximum values after ``Max`` reduction.

   .. cpp:function:: const std::vector<Float>& get_min_val() const

      Minimum values after ``Min`` reduction.

   .. cpp:function:: const std::vector<Float>& get_vol_render_val() const

      RGB values (length 3) after ``VolumeRender`` reduction.

   .. cpp:function:: const std::vector<Segment>& get_segments() const

      The segments from the last ``walk`` call.


Projection
----------

Defined in ``projection.hpp``. Integrates fields along a batch of rays.

.. cpp:class:: Projection

   .. cpp:function:: Projection(const Float* pos_start, const Float* pos_end, size_t ngrid)

      Construct from flat arrays of ray start/end points.

      :param pos_start: ``[x0,y0,z0, x1,y1,z1, ...]`` (length ``3 * ngrid``).
      :param pos_end: Same layout as *pos_start*.
      :param ngrid: Number of rays.

   .. cpp:function:: void makeProjection(const PointCloud& cloud, \
                                         ReductionMode reduction = ReductionMode::Sum)

      Trace all rays and reduce.

   .. cpp:function:: const std::vector<Float>& getProjectionData() const

      Flat result array of size ``ngrid * nfields`` (or ``ngrid * 3`` for
      ``VolumeRender``).

   .. cpp:function:: size_t getNgrid() const

      Number of rays.

   .. cpp:function:: size_t getNfields() const

      Number of output values per ray.


BruteProjection
---------------

Defined in ``brute_projection.hpp``. Grid-based projection over a regular
3D extent.

.. cpp:class:: BruteProjection

   .. cpp:function:: BruteProjection(std::array<size_t,3> npix, std::array<Float,6> extent)

      :param npix: ``{nx, ny, nz}`` -- grid resolution. Rays are cast along
         the z-axis; the output has ``nx * ny`` pixels.
      :param extent: ``{xmin, xmax, ymin, ymax, zmin, zmax}``.

   .. cpp:function:: void makeProjection(const PointCloud& cloud, \
                                         ReductionMode reduction = ReductionMode::Sum)

      Trace rays and reduce.

   .. cpp:function:: void saveProjection(const std::string savename) const

      Write the projection to a binary file.

   .. cpp:function:: const std::vector<Float>& getProjectionData() const

      Flat result array of size ``nx * ny * nfields``.

   .. cpp:function:: size_t getNfields() const
   .. cpp:function:: std::array<size_t,3> getNpix() const


Slice
-----

Defined in ``slice.hpp``. Extracts a 2D slice at a constant depth.

.. cpp:class:: Slice

   .. cpp:function:: Slice(std::array<size_t,2> npix, std::array<Float,4> extent, Float depth)

      :param npix: ``{nx, ny}`` -- output resolution.
      :param extent: ``{xmin, xmax, ymin, ymax}``.
      :param depth: z-coordinate of the slice plane.

   .. cpp:function:: void makeSlice(const PointCloud& cloud)

      Compute the slice.

   .. cpp:function:: void saveSlice(const std::string savename) const

      Write the slice to a binary file.

   .. cpp:function:: const std::vector<Float>& getSliceData() const

      Flat result array of size ``nx * ny * nfields``.

   .. cpp:function:: size_t getNfields() const
   .. cpp:function:: std::array<size_t,2> getNpix() const


Reduction functions
-------------------

Defined in ``reduction.hpp``. Standalone functions that operate on walked
ray segments. These are used internally by ``Ray::integrate`` and
``Projection::makeProjection``.

.. cpp:function:: std::vector<Float> reduce_sum(const std::vector<Ray::Segment>& segments, \
                                                const PointCloud& cloud)

.. cpp:function:: std::vector<Float> reduce_max(const std::vector<Ray::Segment>& segments, \
                                                const PointCloud& cloud)

.. cpp:function:: std::vector<Float> reduce_min(const std::vector<Ray::Segment>& segments, \
                                                const PointCloud& cloud)

.. cpp:function:: std::vector<Float> reduce_volume_render(const std::vector<Ray::Segment>& segments, \
                                                          const PointCloud& cloud)

   Requires 4 fields (R, G, B, alpha). Returns a vector of length 3 (RGB).

.. cpp:function:: std::vector<Float> reduce(const std::vector<Ray::Segment>& segments, \
                                            const PointCloud& cloud, ReductionMode mode)

   Dispatch to the appropriate reduction function.

.. cpp:function:: size_t reduce_output_size(ReductionMode mode, size_t cloud_nfields)

   Number of output values for a given mode and field count.
