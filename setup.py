from glob import glob
from setuptools import setup
try:
    from pybind11.setup_helpers import Pybind11Extension
except:
    from setuptools import Extension as Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "Cvortrace",
        sorted(glob("src/*.cpp")),
        include_dirs=['include'],
        language='c++',
        extra_compile_args=['-std=c++11', '-DTIMING_INFO']  # Sort source files for reproducibility
    ),
]

setup(name='vortrace',
      version='0.1',
      description='Fast projections through voronoi meshes.',
      author='Angus Beane and Matthew Smith',
      author_email='angus.beane@cfa.harvard.edu',
      packages=['vortrace'],
      install_requires=['numpy', 'numba', 'pybind11'],
      setup_requires=['pybind11'],
      extra_requires={'dev': ['h5py', 'pylint', 'pytest'],
                      'test': ['h5py', 'pylint', 'pytest']}
      ext_modules=ext_modules)
