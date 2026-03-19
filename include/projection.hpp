#ifndef PROJ_HPP
#define PROJ_HPP

#include "pointcloud.hpp"
#include "ray.hpp"
#include "mytypes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Projection
{
  private:
    std::vector<Point> pts_end;
    std::vector<Point> pts_start;

    size_t ngrid;    // Length of pts_start and pts_end.
    size_t nfields;  // Number of fields per ray.
    std::vector<Float> proj_data;  // flat, ngrid * nfields

  public:
    Projection(py::array_t<Float> pos_start, py::array_t<Float> pos_end);

    void makeProjection(const PointCloud &cloud, ReductionMode reduction = ReductionMode::Sum);
    py::array_t<double> returnProjection(void) const;
};

#endif