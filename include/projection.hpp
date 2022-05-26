#ifndef PROJ_HPP
#define PROJ_HPP

#include "pointcloud.hpp"
#include "mytypes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Projection
{
  private:
    std::vector<cartarr_t> pts_end;
    std::vector<cartarr_t> pts_start;
    
    size_t ngrid; // Length of pts_start and pts_end.
    std::vector<MyFloat> dens_proj;

  public:
    Projection(py::array_t<MyFloat> pos_start, py::array_t<MyFloat> pos_end);

    void makeProjection(const PointCloud &cloud);
    void saveProjection(const std::string savename) const;
    py::array_t<double> returnProjection(void) const;
};

#endif