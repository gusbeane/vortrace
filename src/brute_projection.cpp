
#include "brute_projection.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <limits>
#include <chrono>
#include <stdexcept>

void BruteProjection::makeProjection(const PointCloud &cloud, ReductionMode mode)
{
  if (mode == ReductionMode::VolumeRender)
    throw std::invalid_argument(
      "VolumeRender is not supported by BruteProjection: "
      "it requires ordered ray segments");

  if(!cloud.get_tree_built())
  {
    throw std::runtime_error("There is currently no valid tree for this point cloud");
  }

  //First check extent is in point cloud bounds
  std::array<Float,6> subbox = cloud.get_subbox();
  if((extent[0] < subbox[0]) || (extent[1] > subbox[1]) ||
      (extent[2] < subbox[2]) || (extent[3] > subbox[3]) ||
      (extent[4] < subbox[4]) || (extent[5] > subbox[5]))
  {
    throw std::invalid_argument("Projection extent out of bounds of current cloud subbox");
  }

  nfields = cloud.get_nfields();

  //Pull out some elements in case of omp slowdown issues
  //Likely unnecessary, compiler should take care of it
  size_t npix_x = npix[0];
  size_t npix_y = npix[1];
  size_t npix_z = npix[2];
  Float start_x = extent[0];
  Float start_y = extent[2];
  Float start_z = extent[4];

  Float deltax = (extent[1] - extent[0]) / (npix_x - 1);
  Float deltay = (extent[3] - extent[2]) / (npix_y - 1);
  Float deltaz = (extent[5] - extent[4]) / (npix_z - 1);

  size_t ngrid = npix_x * npix_y;

  //resize and zero/init result vector(s)
  proj_data.resize(ngrid * nfields);
  if (mode == ReductionMode::Sum) {
    std::fill(proj_data.begin(), proj_data.end(), 0.0);
  } else if (mode == ReductionMode::Max) {
    std::fill(proj_data.begin(), proj_data.end(), -std::numeric_limits<Float>::infinity());
  } else {
    std::fill(proj_data.begin(), proj_data.end(), std::numeric_limits<Float>::infinity());
  }

  if (vortrace::verbose) std::cout << "Making projection...\n";
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(dynamic,256) collapse(2)
  for(size_t i = 0; i < npix_x; i++)
    for(size_t j = 0; j < npix_y; j++)
      for(size_t k = 0; k < npix_z; k++)
      {
        Point query_pt;
        size_t result_idx;
        query_pt[0] = start_x + deltax * i;
        query_pt[1] = start_y + deltay * j;
        query_pt[2] = start_z + deltaz * k;
        result_idx = cloud.queryTree(query_pt);
        size_t base = (i * npix_y + j) * nfields;
        for(size_t f = 0; f < nfields; f++) {
          Float val = cloud.get_field(result_idx, f);
          switch (mode) {
            case ReductionMode::Sum:
              proj_data[base + f] += val;
              break;
            case ReductionMode::Max:
              if (val > proj_data[base + f]) proj_data[base + f] = val;
              break;
            case ReductionMode::Min:
              if (val < proj_data[base + f]) proj_data[base + f] = val;
              break;
          }
        }
      }

  // Scale by deltaz for Sum mode only
  if (mode == ReductionMode::Sum) {
    #pragma omp parallel for schedule(dynamic,256)
    for(size_t i = 0; i < proj_data.size(); i++)
      proj_data[i] *= deltaz;
  }

  if (vortrace::verbose) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Projection generation took " << duration.count() << " milliseconds." << std::endl;
  }
}

void BruteProjection::saveProjection(const std::string savename) const
{
  if(proj_data.empty())
  {
    throw std::runtime_error("Projection has not yet been made");
  }

  if (vortrace::verbose) std::cout << "Saving projection to " << savename << "...   ";
  std::ofstream myfile(savename, std::ios::trunc);
  if (!myfile.is_open())
  {
    throw std::runtime_error("Unable to open savefile: " + savename);
  }

  for(size_t i = 0; i < proj_data.size(); i++)
    myfile << proj_data[i] << "\n";

  myfile.close();
  if (vortrace::verbose) std::cout << "Done." << std::endl;

}

py::array_t<double> BruteProjection::returnProjection(void) const
{
  if(proj_data.empty())
  {
    throw std::runtime_error("Projection has not yet been made");
  }

  size_t ngrid = npix[0] * npix[1];

  if(nfields == 1)
  {
    auto result = py::array_t<double>(ngrid);
    py::buffer_info buf = result.request();
    auto *ptr = static_cast<double *>(buf.ptr);
    for(size_t i = 0; i < ngrid; i++)
      ptr[i] = proj_data[i];
    return result;
  }
  else
  {
    auto result = py::array_t<double>({(ssize_t)ngrid, (ssize_t)nfields});
    py::buffer_info buf = result.request();
    auto *ptr = static_cast<double *>(buf.ptr);
    for(size_t i = 0; i < ngrid * nfields; i++)
      ptr[i] = proj_data[i];
    return result;
  }
}
