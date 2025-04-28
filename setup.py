from glob import glob
from setuptools import setup
import pybind11

# Prefer pybind11’s helper if available
try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "Cvortrace",                         # name of the generated .so
        sorted(glob("src/*.cpp")),           # all your .cpp sources
        include_dirs=[
            "include",                       # your existing headers
            pybind11.get_include(),         # pybind11 headers
        ],
        language="c++",
        extra_compile_args=[
            "-std=c++11", 
            "-DTIMING_INFO"
        ],
    ),
]

setup(
    name="vortrace",
    version="0.1",
    description="Fast projections through Voronoi meshes.",
    author="Angus Beane and Matthew Smith",
    author_email="angus.beane@cfa.harvard.edu",
    packages=["vortrace"],
    install_requires=[
        "numpy",
        "numba",
        "pybind11>=2.6.0",
    ],
    setup_requires=["pybind11"],
    extras_require={
        "dev": ["h5py", "pylint", "pytest"],
        "test": ["h5py", "pylint", "pytest"],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)