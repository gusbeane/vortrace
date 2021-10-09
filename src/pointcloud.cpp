
#include "pointcloud.hpp"
#ifdef TIMING_INFO
#include <chrono>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

void PointCloud::loadPoints(py::array_t<double> pos_in, py::array_t<double> dens_in, const std::array<MyFloat,6> newsubbox)
{
  py::buffer_info buf_pos = pos_in.request();
  py::buffer_info buf_dens = dens_in.request();

  double *pos_in_ptr = (double *) buf_pos.ptr,
         *dens_in_ptr = (double *) buf_dens.ptr;

  // Check to ensure pos and dens have the correct dimensions
  if (buf_pos.ndim != 2 || buf_dens.ndim != 1)
    throw std::runtime_error("pos array must be two-dimensional and dens array must be one-dimensional");
  
  // Check to ensure they have the same number of particles.
  if (buf_pos.size != 3 * buf_dens.size)
  {
    std::cout << "buf_pos.size=" << buf_pos.size << "buf_dens.size=" << buf_dens.size <<"\n";
    throw std::runtime_error("Input sizes must match");
  }

  size_t npart_in = buf_dens.size;

  std::cout << "Applying bounding box...\n";

  subbox = newsubbox;
  //Find particles that are inside the (padded) frame
  MyFloat xmin = BOX_PAD_MIN * subbox[0];
  MyFloat xmax = BOX_PAD_MAX * subbox[1];
  MyFloat ymin = BOX_PAD_MIN * subbox[2];
  MyFloat ymax = BOX_PAD_MAX * subbox[3];
  MyFloat zmin = BOX_PAD_MIN * subbox[4];
  MyFloat zmax = BOX_PAD_MAX * subbox[5];

  std::vector<size_t> limit_idx;
  limit_idx.reserve(npart_in);

  // Apply the bounding box
  for(size_t i=0; i<npart_in; i++)
  {
    if((pos_in_ptr[i*3 + 0] >= xmin) && (pos_in_ptr[i*3 + 0] <= xmax) 
      && (pos_in_ptr[i*3 + 1] >= ymin) && (pos_in_ptr[i*3 + 1] <= ymax)
      && (pos_in_ptr[i*3 + 2] >= zmin) && (pos_in_ptr[i*3 + 2] <= zmax)) 
    {
      limit_idx.push_back(i);
    }
  }

  npart = limit_idx.size();

  // Load selected particles
  pts.resize(npart);
  dens.resize(npart);
  size_t idx;
  for(size_t i = 0; i < npart; i++) 
  { 
    idx = limit_idx[i];
    pts[i][0] = pos_in_ptr[idx*3 + 0];
    pts[i][1] = pos_in_ptr[idx*3 + 1];
    pts[i][2] = pos_in_ptr[idx*3 + 2];
    dens[i] = dens_in_ptr[idx]; 
  }

  std::cout << "npart: " << npart << "\n";

  std::cout << "Snapshot loaded." << std::endl;

  tree_built=false;
}

void PointCloud::buildTree()
{
  if(npart <= 0)
  {
    std::cout << "There are no points in the cloud.\n";
    std::cout << "Aborting tree construction." << std::endl;
    return;
  }

  //Now build tree
  //reset here (vs make_unique) in case snap is reloaded
  std::cout << "Building tree...\n";
#ifdef TIMING_INFO
  auto start = std::chrono::high_resolution_clock::now();
#endif
  tree.reset(new my_kd_tree_t(3,*this,nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
  tree->buildIndex();
  tree_built = true;
#ifdef TIMING_INFO
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Tree build took " << duration.count() << " milliseconds." << std::endl;
#else
    std::cout << " Done." << std::endl;
#endif
}

size_t PointCloud::queryTree(const MyFloat query_pt[3]) const
{
  size_t result;
  MyFloat r2; //
  tree->knnSearch(&query_pt[0], 1, &result, &r2);
  return result;
}

size_t PointCloud::queryTree(const cartarr_t &query_pt) const
{
  //Need native array to pass to knnSearch
  MyFloat query_pt_native[3] = {query_pt[0], query_pt[1], query_pt[2]};
  size_t result;
  MyFloat r2; //
  tree->knnSearch(&query_pt_native[0], 1, &result, &r2);
  return result;
}
