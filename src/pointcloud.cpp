#include "pointcloud.hpp"
#ifdef TIMING_INFO
#include <chrono>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#define TOLERANCE 1E-9

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

size_t PointCloud::queryTree(const cartarr_t &query_pt) const
{
  //Need native array to pass to knnSearch
  size_t result;
  MyFloat r2; //
  tree->knnSearch(query_pt.data(), 1, &result, &r2);
  return result;
}

size_t PointCloud::checkMode(const cartarr_t &query_pt, size_t ctree_id, 
                           size_t ntree_id, int *mode) const
{
  size_t result[8];
  MyFloat r2[8]; //
  
  // Set mode to initially be 3
  // We want it set to be 
  //   0: if we are on an edge between ctree_id and ntree_id
  //   1: if we are on an edge between ctree_id and another cell(s)
  //   2: if we are on an edge between ntree_id and another cell(s)
  //   3: if we are not on an edge between either ctree_id and ntree_id

  *mode = 3;
  int i = 1;

  tree->knnSearch(query_pt.data(), i+1, &result[0], &r2[0]);
  
  if(result[0]==ctree_id)
    *mode -= 2;
  if(result[0]==ntree_id)
    *mode -= 1;

  while(i < 8)
  {
    if(r2[i]-r2[0] <= TOLERANCE)
    {
      if(result[i]==ctree_id)
        *mode -= 2;
      if(result[i]==ntree_id)
        *mode -= 1;
      
      if(*mode == 0)
        break;

      i += 1;
      tree->knnSearch(query_pt.data(), i+1, &result[0], &r2[0]);
    }
    else{
      break;
    }
  }

// Print debug info
  if(*mode==1 || *mode==2){
    printf("mode=%d, ctree_id=%ld, ntree_id=%ld\n", *mode, ctree_id, ntree_id);
    printf("r2=%g|%g|%g|%g\n", r2[0], r2[1], r2[2], r2[3]);
    printf("r2-r2[0]=%g|%g|%g|%g\n", r2[0]-r2[0], r2[1]-r2[0], r2[2]-r2[0], r2[3]-r2[0]);
    printf("result=%ld|%ld|%ld|%ld\n", result[0], result[1], result[2], result[3]);
  }

  return result[0];
}
