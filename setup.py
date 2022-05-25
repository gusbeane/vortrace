from glob import glob
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    #Pybind11Extension(
    Extension(
        "Cvortrace",
        sorted(glob("src/*.cpp")),
        include_dirs=['include'],
        language='c++',
        extra_compile_args=['-std=c++11', '-fopenmp']  # Sort source files for reproducibility
    ),
]

setup(name='vortrace',
      version='0.1',
      description='Fast projections through voronoi meshes.',
      author='Angus Beane and Matthew Smith',
      author_email='angus.beane@cfa.harvard.edu',
      packages=['vortrace'],
      ext_modules=ext_modules)