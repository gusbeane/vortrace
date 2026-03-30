Contributing
============

Setting up a development environment
-------------------------------------

.. code-block:: bash

   git clone https://github.com/gusbeane/vortrace.git
   cd vortrace
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"

Running tests
-------------

**Python tests:**

.. code-block:: bash

   pytest

**C++ tests** (requires `Catch2 <https://github.com/catchorg/Catch2>`_ >= 3.1):

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DBUILD_CPP_TESTS=ON
   cmake --build .
   ctest --output-on-failure

Linting
-------

.. code-block:: bash

   pylint --rcfile=.pylintrc vortrace

The project follows the Google Python style guide with some checks disabled
(see ``.pylintrc``).

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

Open ``_build/html/index.html`` in your browser to view the result.

Code style
----------

- **C++:** C++17, no external dependencies beyond the bundled nanoflann
- **Python:** thin wrapper around the C++ backend; core logic belongs in C++
- Library code should not print to stdout by default -- use the runtime
  ``verbose`` flag or the ``vortrace::warn()`` callback
- Optional dependencies (``h5py``, ``matplotlib``) are imported lazily
