#include "pointcloud.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#define TOLERANCE 1E-9

namespace py = pybind11;

void PointCloud::loadPoints(py::array_t<double> pos_in, py::array_t<double> fields_in, const std::array<Float,6> newsubbox)
{
  py::buffer_info buf_pos = pos_in.request();
  py::buffer_info buf_fields = fields_in.request();

  auto *pos_in_ptr = static_cast<double *>(buf_pos.ptr);
  auto *fields_in_ptr = static_cast<double *>(buf_fields.ptr);

  // Check to ensure pos has correct dimensions
  if (buf_pos.ndim != 2)
    throw std::runtime_error("pos array must be two-dimensional");

  // fields_in can be 1D (npart,) or 2D (npart, nfields)
  if (buf_fields.ndim == 1) {
    nfields = 1;
    npart = buf_fields.shape[0];
  } else if (buf_fields.ndim == 2) {
    nfields = buf_fields.shape[1];
    npart = buf_fields.shape[0];
  } else {
    throw std::runtime_error("fields array must be one-dimensional or two-dimensional");
  }

  // Check to ensure they have the same number of particles.
  if (buf_pos.shape[0] != (ssize_t)npart)
  {
    throw std::runtime_error("Input sizes must match: pos has " +
      std::to_string(buf_pos.shape[0]) + " rows but fields has " +
      std::to_string(npart));
  }

  subbox = newsubbox;

  if (vortrace::verbose) std::cout << "Loading pre-filtered points...\n";

  // Load all points directly (they're already filtered in Python)
  pts.resize(npart);
  fields.resize(npart * nfields);

  for(size_t i = 0; i < npart; i++)
  {
    pts[i][0] = pos_in_ptr[i*3 + 0];
    pts[i][1] = pos_in_ptr[i*3 + 1];
    pts[i][2] = pos_in_ptr[i*3 + 2];
    for(size_t f = 0; f < nfields; f++)
      fields[i * nfields + f] = fields_in_ptr[i * nfields + f];
  }

  // Validation check that points are roughly within expected bounds
  Float dx = subbox[1] - subbox[0];
  Float dy = subbox[3] - subbox[2];
  Float dz = subbox[5] - subbox[4];

  Float pad_x = BOX_PAD_FRACTION * dx;
  Float pad_y = BOX_PAD_FRACTION * dy;
  Float pad_z = BOX_PAD_FRACTION * dz;

  Float xmin = subbox[0] - pad_x;
  Float xmax = subbox[1] + pad_x;
  Float ymin = subbox[2] - pad_y;
  Float ymax = subbox[3] + pad_y;
  Float zmin = subbox[4] - pad_z;
  Float zmax = subbox[5] + pad_z;

  size_t out_of_bounds = 0;
  for(size_t i = 0; i < npart; i++)
  {
    if(!(pts[i][0] >= xmin && pts[i][0] <= xmax &&
         pts[i][1] >= ymin && pts[i][1] <= ymax &&
         pts[i][2] >= zmin && pts[i][2] <= zmax))
    {
      out_of_bounds++;
    }
  }

  if(out_of_bounds > 0)
  {
    throw std::runtime_error("Validation failed: " + std::to_string(out_of_bounds) + " points are outside expected bounds. This indicates a bug in Python-side filtering.");
  }

  if (vortrace::verbose) {
    std::cout << "npart: " << npart << "\n";
    std::cout << "Snapshot loaded." << std::endl;
  }

  tree_built=false;
}

void PointCloud::buildTree()
{
  if(npart <= 0)
  {
    throw std::runtime_error("There are no points in the cloud; cannot build tree");
  }

  //Now build tree
  //reset here (vs make_unique) in case snap is reloaded
  if (vortrace::verbose) std::cout << "Building tree...\n";
  auto start = std::chrono::high_resolution_clock::now();
  tree.reset(new my_kd_tree_t(3,*this,nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
  tree->buildIndex();
  tree_built = true;
  if (vortrace::verbose) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Tree build took " << duration.count() << " milliseconds." << std::endl;
  }
}

size_t PointCloud::queryTree(const Point &query_pt) const
{
  //Need native array to pass to knnSearch
  size_t result;
  Float r2; //
  tree->knnSearch(query_pt.data(), 1, &result, &r2);
  return result;
}

size_t PointCloud::checkMode(const Point &query_pt, size_t ctree_id,
                           size_t ntree_id, int *mode) const
{
  size_t result[8];
  Float r2[8]; //

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

  return result[0];
}
