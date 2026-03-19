
#include "projection.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#ifdef TIMING_INFO
#include <chrono>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

Projection::Projection(py::array_t<Float> pos_start, py::array_t<Float> pos_end)
{
  // Load input python arrays into pointers
  py::buffer_info buf_pos_start = pos_start.request();
  py::buffer_info buf_pos_end = pos_end.request();

  auto *pos_start_ptr = static_cast<Float *>(buf_pos_start.ptr);
  auto *pos_end_ptr = static_cast<Float *>(buf_pos_end.ptr);

  // Check to ensure they have the correct dimensions
  if (buf_pos_start.ndim != 2 || buf_pos_end.ndim != 2)
    throw std::runtime_error("PROJECTION: pos_start and pos_end array must both be two-dimensional\n");

  // Check to ensure they have the same number of particles.
  if (buf_pos_start.size != buf_pos_end.size)
  {
    throw std::runtime_error("PROJECTION: pos_start and pos_end must have same sizes");
  }

  // Allocate and load in arrrays.
  ngrid = buf_pos_start.size/3;
  nfields = 0;  // will be set in makeProjection from cloud

  pts_start.resize(ngrid);
  pts_end.resize(ngrid);

  for(size_t i=0; i<ngrid; i++)
  {
    for(int j=0; j<3; j++)
    {
      pts_start[i][j] = pos_start_ptr[i*3 + j];
      pts_end[i][j]   = pos_end_ptr[i*3 + j];
    }
  }

}

void Projection::makeProjection(const PointCloud &cloud, ReductionMode mode)
{

  if(!cloud.get_tree_built())
  {
    throw std::runtime_error("There is currently no valid tree for this point cloud");
  }

  nfields = cloud.get_nfields();

  //resize and zero result vector(s)
  proj_data.resize(ngrid * nfields);
  std::fill(proj_data.begin(), proj_data.end(), 0.0);

  if (vortrace::verbose) std::cout << "Making projection...\n";
#ifdef TIMING_INFO
  auto start = std::chrono::high_resolution_clock::now();
#endif
  std::exception_ptr eptr = nullptr;
  #pragma omp parallel for schedule(dynamic,256)
  for(size_t i = 0; i < ngrid; i++)
  {
    if (eptr) continue;
    try {
        Ray projray(pts_start[i], pts_end[i]);
        projray.integrate(cloud, mode);

        const std::vector<Float> *src;
        switch (mode) {
          case ReductionMode::Max: src = &projray.get_max_val(); break;
          case ReductionMode::Min: src = &projray.get_min_val(); break;
          default:                 src = &projray.get_col();     break;
        }

        for(size_t f = 0; f < nfields; f++)
          proj_data[i * nfields + f] = (*src)[f];
    } catch (...) {
      #pragma omp critical
      { if (!eptr) eptr = std::current_exception(); }
    }
  }
  if (eptr) std::rethrow_exception(eptr);

#ifdef TIMING_INFO
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    if (vortrace::verbose) std::cout << "Projection generation took " << duration.count() << " milliseconds." << std::endl;
#else
    if (vortrace::verbose) std::cout << "Projection complete." << std::endl;
#endif
}

py::array_t<double> Projection::returnProjection(void) const
{
  //First check if projection has been made
  if(proj_data.empty())
  {
    throw std::runtime_error("Projection has not yet been made");
  }

  if(nfields == 1)
  {
    // Backward compatible: return 1D array (ngrid,)
    auto result = py::array_t<double>(ngrid);
    py::buffer_info buf = result.request();
    double *result_ptr = static_cast<double *>(buf.ptr);
    for(size_t i = 0; i < ngrid; i++)
      result_ptr[i] = proj_data[i];
    return result;
  }
  else
  {
    // Return 2D array (ngrid, nfields)
    auto result = py::array_t<double>({(ssize_t)ngrid, (ssize_t)nfields});
    py::buffer_info buf = result.request();
    double *result_ptr = static_cast<double *>(buf.ptr);
    for(size_t i = 0; i < ngrid * nfields; i++)
      result_ptr[i] = proj_data[i];
    return result;
  }
}


