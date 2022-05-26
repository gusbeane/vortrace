
#include "projection.hpp"
#include "ray.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#ifdef TIMING_INFO
#include <chrono>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

Projection::Projection(py::array_t<MyFloat> pos_start, py::array_t<MyFloat> pos_end)
{
  // Load input python arrays into pointers
  py::buffer_info buf_pos_start = pos_start.request();
  py::buffer_info buf_pos_end = pos_end.request();

  MyFloat *pos_start_ptr = (MyFloat *) buf_pos_start.ptr,
          *pos_end_ptr = (MyFloat *) buf_pos_end.ptr;
  
  // Check to ensure they have the correct dimensions
  if (buf_pos_start.ndim != 2 || buf_pos_end.ndim != 2)
    throw std::runtime_error("PROJECTION: pos_start and pos_end array must both be two-dimensional\n");
  
  // Check to ensure they have the same number of particles.
  if (buf_pos_start.size != buf_pos_end.size)
  {
    std::cout << "buf_pos_start.size=" << buf_pos_start.size << "buf_pos_end.size=" << buf_pos_end.size <<"\n";
    throw std::runtime_error("PROJECTION: pos_start and pos_end must have same sizes");
  }

  // Allocate and load in arrrays.
  ngrid = buf_pos_start.size/3;

  pts_start.reserve(ngrid);
  pts_end.reserve(ngrid);

  for(size_t i=0; i<ngrid; i++)
  {
    for(int j=0; j<3; j++)
    {
      pts_start[i][j] = pos_start_ptr[i*3 + j];
      pts_end[i][j]   = pos_end_ptr[i*3 + j];
    }
  }

}

void Projection::makeProjection(const PointCloud &cloud)
{

  if(!cloud.get_tree_built())
  {
    std::cout << "There is currently no valid tree for this point cloud.\n";
    std::cout << "Aborting projection.\n" << std::endl;
    return;
  }

  //resize and zero result vector(s)
  dens_proj.resize(ngrid);
  memset(&dens_proj[0], 0.0, dens_proj.size() * sizeof dens_proj[0]);
  
  std::cout << "Making projection...\n";
#ifdef TIMING_INFO
  auto start = std::chrono::high_resolution_clock::now();
#endif
  #pragma omp parallel for schedule(dynamic,256) collapse(2)
  for(size_t i = 0; i < ngrid; i++)
  {
        Ray projray(pts_start[i], pts_end[i]);
        projray.integrate(cloud);
        dens_proj[i] = projray.get_dens_col();
  }

#ifdef TIMING_INFO
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Projection generation took " << duration.count() << " milliseconds." << std::endl;
#else
    std::cout << "Projection complete." << std::endl;
#endif
}

/*Currently we save as text, for debugging. Smarter method should go
here*/
void Projection::saveProjection(const std::string savename) const
{
  std::cout << "Saving projection to " << savename << "...   ";
  //First check if slice has been made
  if(dens_proj.empty())
  {
    std::cout << "Projection has not yet been made. Aborting save." << std::endl;
    return;
  }

  std::ofstream myfile(savename, std::ios::trunc);
  if (myfile.is_open())
  {
    for(size_t i = 0; i < dens_proj.size(); i++)
      myfile << dens_proj[i] << "\n";

    myfile.close();
    std::cout << "Done." << std::endl;
  }
  else std::cout << "Unable to open savefile." << std::endl;

}

py::array_t<double> Projection::returnProjection(void) const
{
  // std::cout << "Saving projection to " << savename << "...   ";
  //First check if slice has been made
  if(dens_proj.empty())
  {
    std::cout << "Projection has not yet been made. Aborting." << std::endl;
    exit(1);
  }

  // std::ofstream myfile(savename, std::ios::trunc);
  // if (myfile.is_open())
  // {
  auto result = py::array_t<double>(dens_proj.size());
  py::buffer_info buf = result.request();
  double *result_ptr = static_cast<double *>(buf.ptr);

  for(size_t i = 0; i < dens_proj.size(); i++){
    result_ptr[i] = dens_proj[i];
  }

  return result;
}


