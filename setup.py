import os
import re
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        # no source files, CMake will take care of everything
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # verify CMake is installed
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake >= 3.14 must be installed to build Cvortrace")
        # build each extension
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # where to put the compiled .so
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        # configure step
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)

        # build step
        build_args = ["--config", cfg, "--", f"-j{os.cpu_count() or 2}"]
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

# now wire it all up
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
    # replace your Pybind11Extension with a single CMakeExtension
    ext_modules=[CMakeExtension("Cvortrace", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)