Comparison with COLT and SKIRT
==============================

``vortrace`` does projections through Voronoi meshes without explicitly building the mesh. We compare the performance of ``vortrace`` to two codes that do an explicit mesh construction: `COLT <https://colt.readthedocs.io>`_ and
`SKIRT <https://skirt.ugent.be>`_.

Setup
-----

We run our test on the same setup as the `yt` comparison described in the paper: an
isolated dwarf-galaxy snapshot from `Smith 2026 <https://ui.adsabs.harvard.edu/abs/2026MNRAS.549ag959S/abstract>`_ with 27.3M gas cells in a 180 kpc box,
imaged over a 6 kpc window at 1024\ :sup:`2` pixels through the 4 kpc slab
``88 < z < 92`` kpc. About 22.5M of the cells lie inside that column -- each code is given this subset of cells.

.. list-table::
   :widths: 22 78

   * - Machine
     - Intel Xeon Silver 4216 (16 cores, 32 threads), 124 GB RAM,
       Ubuntu 22.04
   * - Threads
     - 32 for every code. The ``vortrace`` kD-tree build is serial; its
       projection uses OpenMP.
   * - ``vortrace``
     - 0.1.3, built with gcc 11
   * - COLT
     - commit ``8d0702d11fe4``, built with g++-13 and CGAL 5.6.2, no MPI.
       Mesh from CGAL's 3D Delaunay triangulation.
   * - SKIRT
     - commit ``d8e79d41``, CMake release build. Mesh from its bundled
       Voro++.

Timings
-------

.. list-table::
   :header-rows: 1
   :widths: 22 22 18 16 22

   * - Code
     - Mesh or tree build [s]
     - Ray tracing [s]
     - Total [s]
     - Peak memory [GiB]
   * - ``vortrace``
     - 8.4
     - 16.7
     - 28.4
     - 3.2
   * - ``COLT``
     - 339
     - 2.3
     - 345
     - 57
   * - ``SKIRT``
     - 110
     - 3.4
     - 182
     - 16

If a mesh exists and fits in memory, then tracing rays through it is faster. However, if a mesh is not needed for other purposes, then ``vortrace`` is between 6 and 12 times faster and uses between 5 and 18 times less memory than ``COLT`` or ``SKIRT``.

Totals are for a single run and include reading the input and writing the image. For ``COLT``, we report Initializations and Projections for mesh construction and ray tracing, respectively. For SKIRT, we infer the breakdown from timings in the log file.


Agreement
---------

The relative difference for nearly all pixels between ``vortrace`` and each of ``COLT`` and ``SKIRT`` is within 1e-6. There is one ray with a relative difference of 2e-4 for both, which is caused by two equidistant generator points (i.e., a ray lying exactly on the face between two cells). In the case of ``SKIRT``, there is a pixel with a relative difference of 1.8e-3 -- a brute force integration agrees with ``vortrace``.


Reproducing
-----------

The harness lives in ``paper/code_comparison``. It builds COLT and SKIRT
into ``data/external`` on first use.

On Ubuntu 22.04:

.. code-block:: bash

   sudo apt install build-essential g++-13 cmake curl time python3-venv \
       python3-dev libhdf5-dev libyaml-cpp-dev libgsl-dev libgmp-dev \
       libmpfr-dev libboost-dev

Unpack the sources by hand:

- COLT (``bitbucket.org/aaron_smith/colt``) into ``data/external/colt``
- SKIRT9 (``github.com/SKIRT/SKIRT9``) into ``data/external/SKIRT9``
- the header-only CGAL 5.6 release into
  ``data/external/deps/src/CGAL-5.6.2``

COLT needs CGAL 5.6 or newer and a compiler newer than gcc 11; Ubuntu 22.04
ships CGAL 5.4, hence the separate download and ``g++-13``. For COLT and SKIRT9, we used commits `8d0702d11fe4` and `d8e79d41`, respectively.

.. code-block:: bash

   pip install . h5py matplotlib
   cd paper/code_comparison
   python colt_projection.py --build && python colt_projection.py --selftest
   python skirt_projection.py --selftest    # builds SKIRT on first use

   export OMP_NUM_THREADS=32
   python make_code_comparison.py --galaxy small --threads 32 --repeat 3
   python make_code_comparison.py --galaxy large --threads 32

The small case needs nothing beyond the repository. The large case needs the
dwarf-galaxy snapshot as ``data/snap_500.*.hdf5``, which is not shipped, and
about 60 GiB of memory for COLT. Results are written to
``stats_<galaxy>.json`` and ``timings.md`` next to the scripts.
