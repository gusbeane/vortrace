
#include "brute_projection.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#ifdef TIMING_INFO
#include <chrono>
#endif

void BruteProjection::makeProjection(const PointCloud &cloud)
{

  if(!cloud.get_tree_built())
  {
    std::cout << "There is currently no valid tree for this point cloud.\n";
    std::cout << "Aborting projection.\n" << std::endl;
    return;
  }

  //First check extent is in point cloud bounds
  std::array<Float,6> subbox = cloud.get_subbox();
  if((extent[0] < subbox[0]) || (extent[1] > subbox[1]) || 
      (extent[2] < subbox[2]) || (extent[3] > subbox[3]) ||
      (extent[4] < subbox[4]) || (extent[5] > subbox[5]))
  {
    std::cout << "Projection extent out of bounds of current cloud subbox.\n";
    std::cout << "Aborting projection production." << std::endl;
    return;
  }

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

  //resize and zero result vector(s)
  dens_proj.resize(npix_x * npix_y);
  memset(&dens_proj[0], 0, dens_proj.size() * sizeof dens_proj[0]);
  
  std::cout << "Making projection...\n";
#ifdef TIMING_INFO
  auto start = std::chrono::high_resolution_clock::now();
#endif
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
        dens_proj[i * npix_y + j] += cloud.get_dens(result_idx);
      }

  #pragma omp parallel for schedule(dynamic,256)
  for(size_t i = 0; i < dens_proj.size(); i++)
    dens_proj[i] *= deltaz;

#ifdef TIMING_INFO
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Projection generation took " << duration.count() << " milliseconds.\n";
#endif
  std::cout << "Projection complete." << std::endl;
}

void BruteProjection::saveProjection(const std::string savename) const
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


