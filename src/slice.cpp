
#include "slice.hpp"
#include <iostream>
#include <fstream>
#ifdef TIMING_INFO
#include <chrono>
#endif

void Slice::makeSlice(const PointCloud &cloud)
{

  if(!cloud.get_tree_built())
  {
    std::cout << "There is currently no valid tree for this point cloud.\n";
    std::cout << "Aborting projection.\n" << std::endl;
    return;
  }

  //First check extent is in cloud bounds
  std::array<MyFloat,6> subbox = cloud.get_subbox();
  if((extent[0] < subbox[0]) || (extent[1] > subbox[1]) || 
      (extent[2] < subbox[2]) || (extent[3] > subbox[3]) ||
      (depth < subbox[4]) || (depth > subbox[5]))
  {
    std::cout << "Slice extent out of bounds of current cloud subbox.\n";
    std::cout << "Aborting slice production." << std::endl;
    return;
  }

  //Pull out some elements in case of omp slowdown issues
  //Likely unnecessary, compiler should take care of it
  size_t npix_x = npix[0];
  size_t npix_y = npix[1];

  MyFloat start_x = extent[0];
  MyFloat start_y = extent[2];
  //Create slice(s)
  MyFloat deltax = (extent[1] - extent[0]) / (npix_x - 1);
  MyFloat deltay = (extent[3] - extent[2]) / (npix_y - 1); 

  //resize result vector(s)
  dens_slice.resize(npix_x * npix_y);
  
  std::cout << "Making slice...\n";
#ifdef TIMING_INFO
  auto start = std::chrono::high_resolution_clock::now();
#endif

  #pragma omp parallel for schedule(dynamic,256) collapse(2)
  for(size_t i = 0; i < npix_x; i++)
    for(size_t j = 0; j < npix_y; j++)
    {
      MyFloat query_pt[3];
      size_t result_idx;
      query_pt[0] = start_x + deltax * i;
      query_pt[1] = start_y + deltay * j;
      query_pt[2] = depth;
      result_idx = cloud.queryTree(query_pt);
      dens_slice[i * npix_y + j] = cloud.get_dens(result_idx);
    }
#ifdef TIMING_INFO
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Slice generation took " << duration.count() << " microseconds\n";
#endif
  std::cout << "Slice complete." << std::endl;
}

void Slice::saveSlice(const std::string savename) const
{
  std::cout << "Saving slice to " << savename << "...\n";
  //First check if slice has been made
  if(dens_slice.empty())
  {
    std::cout << "Slice has not yet been made. Aborting save." << std::endl;
    return;
  }

  std::ofstream myfile(savename, std::ios::trunc);
  if (myfile.is_open())
  {
    for(size_t i = 0; i < dens_slice.size(); i++)
      myfile << dens_slice[i] << "\n";

    myfile.close();
  }
  else std::cout << "Unable to open savefile." << std::endl;

}


