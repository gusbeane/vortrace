Saving and Loading
==================

``vortrace`` provides :mod:`vortrace.io` for saving and loading projection grids
and point clouds. Both **NPZ** (default) and **HDF5** formats are supported.

Saving and loading grids
------------------------

.. tab:: Python

   .. code-block:: python

      import vortrace as vt

      # Save a projection grid
      vt.io.save_grid(
          "projection.npz", image,
          extent=[-L / 2, L / 2, -L / 2, L / 2],
          metadata={"projection": "xy", "npix": 256},
      )

      # Load it back
      data, meta = vt.io.load_grid("projection.npz")
      print(data.shape)       # (256, 256)
      print(meta["extent"])   # [-37.5, 37.5, -37.5, 37.5]

   HDF5 format (requires ``h5py``):

   .. code-block:: python

      vt.io.save_grid("projection.hdf5", image, fmt="hdf5")
      data, meta = vt.io.load_grid("projection.hdf5")

.. tab:: C++

   The C++ ``BruteProjection`` and ``Slice`` classes have built-in save
   methods that write binary output:

   .. code-block:: cpp

      BruteProjection bp(npix, extent);
      bp.makeProjection(cloud, ReductionMode::Sum);
      bp.saveProjection("projection.dat");

      Slice slice(npix_2d, extent_2d, depth);
      slice.makeSlice(cloud);
      slice.saveSlice("slice.dat");

   For ``Projection`` results (custom ray grids), access the data via
   ``getProjectionData()`` and write it using your own I/O routines.

.. image:: /images/io_loaded.png
   :width: 80%
   :align: center
   :alt: Projection loaded from disk

Saving and loading point clouds
-------------------------------

.. tab:: Python

   You can save and reload an entire ``ProjectionCloud`` (including
   positions, fields, and bounding box). The kD-tree is rebuilt on load.

   .. code-block:: python

      # Save
      pc.save("cloud.npz")

      # Load
      pc_loaded = vt.ProjectionCloud.load("cloud.npz")

.. tab:: C++

   Cloud serialization is not provided by the C++ library. Use your own
   I/O format to store positions and fields, then call ``loadPoints()`` and
   ``buildTree()`` as usual.

Supported formats
-----------------

.. list-table::
   :header-rows: 1

   * - Format
     - Extension
     - Notes
   * - NPZ
     - ``.npz``
     - Default. No extra dependencies.
   * - HDF5
     - ``.hdf5``, ``.h5``
     - Requires ``h5py``. Richer metadata support.
