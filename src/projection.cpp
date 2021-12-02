
#include "projection.hpp"
#include "ray.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#ifdef TIMING_INFO
#include <chrono>
#endif

void Projection::makeProjection(const PointCloud &cloud)
{

  if(!cloud.get_tree_built())
  {
    std::cout << "There is currently no valid tree for this point cloud.\n";
    std::cout << "Aborting projection.\n" << std::endl;
    return;
  }

  //First check extent is in cloud bounding box
  std::array<MyFloat,6> subbox = cloud.get_subbox();
  if((extent[0] < subbox[0]) || (extent[1] > subbox[1]) || 
      (extent[2] < subbox[2]) || (extent[3] > subbox[3]) ||
      (extent[4] < subbox[4]) || (extent[5] > subbox[5]))
  {
    std::cout << "Projection extent out of bounds of current point cloud bounding box.\n";
    std::cout << "Aborting projection production." << std::endl;
    return;
  }

  //Pull out some elements in case of omp slowdown issues
  //Likely unnecessary, compiler should take care of it
  size_t npix_x = npix[0];
  size_t npix_y = npix[1];
  size_t start_x = extent[0];
  size_t start_y = extent[2];


  MyFloat deltax = (extent[1] - extent[0]) / (npix_x - 1);
  MyFloat deltay = (extent[3] - extent[2]) / (npix_y - 1);

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
      {
        cartarr_t pos_start = {start_x + deltax * i, start_y + deltay * j,extent[4]};
        cartarr_t pos_end = {start_x + deltax * i, start_y + deltay * j,extent[5]};
        Ray projray(pos_start,pos_end);
        projray.integrate(cloud);
        dens_proj[i * npix_y + j] = projray.get_dens_col();
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


