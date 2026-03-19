#ifndef SLICE_HPP
#define SLICE_HPP

#include "pointcloud.hpp"
#include "mytypes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Slice
{
  private:
    std::array<size_t,2> npix;
    std::array<Float,4> extent;
    Float depth;

    size_t nfields;
    std::vector<Float> slice_data;  // flat, (npix_x * npix_y) * nfields

  public:
    Slice(std::array<size_t,2> npix, std::array<Float,4> extent, Float depth) :
      npix(npix), extent(extent), depth(depth), nfields(0) {}

    void makeSlice(const PointCloud &cloud);
    void saveSlice(const std::string savename) const;
    py::array_t<double> returnSlice(void) const;

};

#endif
