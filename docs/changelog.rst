Changelog
=========

v0.1 (2024)
-----------

Initial release.

- Recursive split-point ray-tracing algorithm
- Reduction modes: Sum (integrate), Max, Min, Volume Render
- Multi-field projection support
- Python ``ProjectionCloud`` high-level API
- C++ standalone library with CMake install and ``find_package`` support
- Grid projections with Tait-Bryan rotation support
- Arbitrary ray projections (custom start/end arrays)
- Single-ray segment inspection
- 2D slicing through the mesh
- Periodic boundary condition support
- I/O: save/load grids and clouds in NPZ and HDF5 formats
- Plotting helpers: ``plot_grid`` and ``plot_ray``
- OpenMP parallelisation (optional, auto-detected)
- nanoflann kD-tree (bundled, no external dependency)
