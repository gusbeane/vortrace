#!/bin/bash
# Build COLT and SKIRT for the comparison (Ubuntu 22.04).
#
# Packages needed:
#   sudo apt install build-essential g++-13 cmake curl unzip time \
#       libhdf5-dev libyaml-cpp-dev libgsl-dev libgmp-dev libmpfr-dev \
#       libboost-dev
#
# Unpack these sources by hand before running this script:
#   data/external/colt                  bitbucket.org/aaron_smith/colt, commit 8d0702d11fe4
#   data/external/SKIRT9                github.com/SKIRT/SKIRT9, commit d8e79d41
#   data/external/deps/src/CGAL-5.6.2   header-only CGAL release
set -e
cd "$(dirname "$0")/../../data/external"
EXT=$PWD

# COLT: Voronoi geometry with CGAL, no MPI.  It needs CGAL >= 5.6 and a
# compiler newer than gcc 11, hence the separate CGAL and g++-13.
mkdir -p colt_build
cat > colt_build/defines.yaml <<EOF
GEOMETRY: voronoi
HAVE_CGAL: true
NO_MPI: true
long_neighbors: true
EOF
make -C colt -j8 DEFS=$EXT/colt_build/defines.yaml BUILD=$EXT/colt_build/build \
    EXE=$EXT/colt_build/colt CFLAGS="-fopenmp -std=c++20 -O3" CC=g++-13 \
    CGAL=$EXT/deps/src/CGAL-5.6.2

# SKIRT: release build of the command-line program only.  At run time it
# looks for its resources in ../git and ../resources next to the build.
ln -sfn SKIRT9 git
mkdir -p SKIRT9_build resources
(cd SKIRT9_build &&
 cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SKIRT=ON -DBUILD_MAKE_UP=OFF \
       -DBUILD_DOX_STYLE=OFF ../SKIRT9 &&
 make -j8)
curl -sSL -o resources/core.zip \
    https://sciences.ugent.be/skirtextdat/SKIRT9/Resources/SKIRT9_Resources_Core_v8.zip
unzip -q -o resources/core.zip -d resources
rm resources/core.zip
