Installation
============

.. tab:: pip (recommended)

   .. code-block:: bash

      pip install vortrace

   This installs the pre-built package from PyPI, including the compiled C++
   extension.

   Optional extras:

   .. code-block:: bash

      pip install vortrace[plot]       # matplotlib
      pip install vortrace[dev]        # h5py, matplotlib, pylint, pytest

.. tab:: Python from source

   .. code-block:: bash

      git clone https://github.com/gusbeane/vortrace.git
      cd vortrace
      pip install ./

   This builds the C++ extension via scikit-build-core and CMake automatically.
   For an editable (development) install:

   .. code-block:: bash

      pip install -e .

.. tab:: C++ (standalone)

   ``vortrace`` can be used as a standalone C++17 library without Python.
   The only dependency is a C++17-capable compiler and CMake >= 3.15. The
   bundled `nanoflann <https://github.com/jlblancoc/nanoflann>`_ header-only
   library is included -- no external dependencies are required.

   .. code-block:: bash

      git clone https://github.com/gusbeane/vortrace.git
      cd vortrace
      mkdir build && cd build
      cmake .. -DBUILD_PYTHON_BINDINGS=OFF
      cmake --build .
      cmake --install . --prefix /usr/local

   Use ``find_package`` in your downstream ``CMakeLists.txt``:

   .. code-block:: cmake

      cmake_minimum_required(VERSION 3.15)
      project(my_project LANGUAGES CXX)

      find_package(vortrace REQUIRED)

      add_executable(my_app main.cpp)
      target_link_libraries(my_app PRIVATE vortrace::vortrace_core)

   Then build with the install prefix on the CMake search path:

   .. code-block:: bash

      cmake -DCMAKE_PREFIX_PATH=/usr/local ..
      cmake --build .

Requirements
------------

* **Python:** >= 3.8, with ``numpy`` and ``numba`` (installed automatically)
* **C++:** C++17 compiler (GCC >= 7, Clang >= 5, MSVC 2017+)
* **CMake:** >= 3.15

Optional Python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``h5py`` -- for saving/loading projections in HDF5 format
* ``matplotlib`` -- for the :mod:`vortrace.plot` helpers

CMake options
-------------

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``BUILD_PYTHON_BINDINGS``
     - ``ON``
     - Build the pybind11 Python extension
   * - ``BUILD_CPP_TESTS``
     - ``OFF``
     - Build C++ unit tests (requires Catch2 >= 3.1)

OpenMP is auto-detected. If found, ``vortrace_core`` is linked against
``OpenMP::OpenMP_CXX`` automatically.

Running C++ tests
-----------------

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DBUILD_CPP_TESTS=ON
   cmake --build .
   ctest --output-on-failure
