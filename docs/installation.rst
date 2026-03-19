Installation
============

From source
-----------

.. code-block:: bash

   git clone git@github.com:gusbeane/vortrace.git
   cd vortrace
   pip install ./

This builds the C++ extension via scikit-build-core and CMake automatically.

Requirements
------------

* Python >= 3.7
* A C++ compiler with C++17 support
* CMake >= 3.15

Python dependencies (installed automatically):

* ``numpy``
* ``numba``
* ``pybind11``

Optional dependencies
---------------------

* ``h5py`` -- for saving/loading projections in HDF5 format
* ``matplotlib`` -- for the :mod:`vortrace.plot` helpers
