
#include "slice.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

void Slice::makeSlice(const PointCloud &cloud)
{

  if(!cloud.get_tree_built())
  {
    throw std::runtime_error("There is currently no valid tree for this point cloud");
  }

  //First check extent is in cloud bounds
  std::array<Float,6> subbox = cloud.get_subbox();
  if((extent[0] < subbox[0]) || (extent[1] > subbox[1]) ||
      (extent[2] < subbox[2]) || (extent[3] > subbox[3]) ||
      (depth < subbox[4]) || (depth > subbox[5]))
  {
    throw std::runtime_error("Slice extent out of bounds of current cloud subbox");
  }

  nfields = cloud.get_nfields();

  size_t npix_x = npix[0];
  size_t npix_y = npix[1];

  Float start_x = extent[0];
  Float start_y = extent[2];
  Float deltax = (extent[1] - extent[0]) / (npix_x - 1);
  Float deltay = (extent[3] - extent[2]) / (npix_y - 1);

  size_t ngrid = npix_x * npix_y;
  slice_data.resize(ngrid * nfields);

  if (vortrace::verbose) std::cout << "Making slice...\n";
  auto start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for schedule(dynamic,256) collapse(2)
  for(size_t i = 0; i < npix_x; i++)
    for(size_t j = 0; j < npix_y; j++)
    {
      Point query_pt;
      size_t result_idx;
      query_pt[0] = start_x + deltax * i;
      query_pt[1] = start_y + deltay * j;
      query_pt[2] = depth;
      result_idx = cloud.queryTree(query_pt);
      size_t base = (i * npix_y + j) * nfields;
      for(size_t f = 0; f < nfields; f++)
        slice_data[base + f] = cloud.get_field(result_idx, f);
    }

  if (vortrace::verbose) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Slice generation took " << duration.count() << " microseconds." << std::endl;
  }
}

void Slice::saveSlice(const std::string savename) const
{
  if(slice_data.empty())
  {
    throw std::runtime_error("Slice has not yet been made");
  }

  if (vortrace::verbose) std::cout << "Saving slice to " << savename << "...\n";
  std::ofstream myfile(savename, std::ios::trunc);
  if (!myfile.is_open())
  {
    throw std::runtime_error("Unable to open savefile: " + savename);
  }

  for(size_t i = 0; i < slice_data.size(); i++)
    myfile << slice_data[i] << "\n";

  myfile.close();
}

