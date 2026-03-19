#include "pointcloud.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>

#define TOLERANCE 1E-9

namespace py = pybind11;

void PointCloud::loadPoints(py::array_t<double> pos_in, py::array_t<double> fields_in,
                            const std::array<Float,6> newsubbox,
                            py::array_t<double> vol,
                            bool periodic_in, bool filter_in)
{
  py::buffer_info buf_pos = pos_in.request();
  py::buffer_info buf_fields = fields_in.request();

  auto *pos_in_ptr = static_cast<double *>(buf_pos.ptr);
  auto *fields_in_ptr = static_cast<double *>(buf_fields.ptr);

  // Check to ensure pos has correct dimensions
  if (buf_pos.ndim != 2)
    throw std::runtime_error("pos array must be two-dimensional");

  // fields_in can be 1D (npart,) or 2D (npart, nfields)
  size_t npart_in;
  if (buf_fields.ndim == 1) {
    nfields = 1;
    npart_in = buf_fields.shape[0];
  } else if (buf_fields.ndim == 2) {
    nfields = buf_fields.shape[1];
    npart_in = buf_fields.shape[0];
  } else {
    throw std::runtime_error("fields array must be one-dimensional or two-dimensional");
  }

  // Check to ensure they have the same number of particles.
  if (buf_pos.shape[0] != (ssize_t)npart_in)
  {
    throw std::runtime_error("Input sizes must match: pos has " +
      std::to_string(buf_pos.shape[0]) + " rows but fields has " +
      std::to_string(npart_in));
  }

  subbox = newsubbox;
  periodic = periodic_in;

  // Compute box size from subbox
  Float dx = subbox[1] - subbox[0];
  Float dy = subbox[3] - subbox[2];
  Float dz = subbox[5] - subbox[4];
  box_size = {dx, dy, dz};

  // Resolve filter option
  bool do_filter = filter_in;
  if (periodic && do_filter) {
    PyErr_WarnEx(PyExc_UserWarning,
      "periodic=True is incompatible with filter=True; setting "
      "filter=False. To suppress this warning, pass filter=False "
      "explicitly.", 1);
    do_filter = false;
  }

  pts.clear();
  fields.clear();
  orig_ids.clear();

  if (do_filter) {
    // Compute padding
    py::buffer_info buf_vol = vol.request();
    if (buf_vol.size > 0) {
      // Adaptive padding from cell volumes
      auto *vol_ptr = static_cast<double *>(buf_vol.ptr);
      double max_vol = 0.0;
      for (ssize_t i = 0; i < buf_vol.size; i++) {
        if (vol_ptr[i] > max_vol) max_vol = vol_ptr[i];
      }
      double max_radius = std::cbrt(3.0 * max_vol / (4.0 * M_PI));
      pad = 3.0 * max_radius;
    } else {
      // Fallback: uniform 15% of longest box side
      pad = 0.15 * std::max({dx, dy, dz});
      PyErr_WarnEx(PyExc_UserWarning,
        "No cell volumes provided; using 15% of longest box side as "
        "padding. Pass vol= for adaptive padding.", 1);
    }

    Float xmin = subbox[0] - pad;
    Float xmax = subbox[1] + pad;
    Float ymin = subbox[2] - pad;
    Float ymax = subbox[3] + pad;
    Float zmin = subbox[4] - pad;
    Float zmax = subbox[5] + pad;

    if (vortrace::verbose) std::cout << "Filtering and loading points...\n";

    // Filter particles into padded bounding box
    for(size_t i = 0; i < npart_in; i++)
    {
      double px = pos_in_ptr[i*3 + 0];
      double py_val = pos_in_ptr[i*3 + 1];
      double pz = pos_in_ptr[i*3 + 2];

      if (px >= xmin && px <= xmax &&
          py_val >= ymin && py_val <= ymax &&
          pz >= zmin && pz <= zmax)
      {
        Point pt;
        pt[0] = px;
        pt[1] = py_val;
        pt[2] = pz;
        pts.push_back(pt);

        for(size_t f = 0; f < nfields; f++)
          fields.push_back(fields_in_ptr[i * nfields + f]);

        orig_ids.push_back(i);
      }
    }
  } else {
    // No filtering: load all particles
    if (vortrace::verbose) std::cout << "Loading all points (no filtering)...\n";

    pad = 0.0;
    for(size_t i = 0; i < npart_in; i++)
    {
      Point pt;
      pt[0] = pos_in_ptr[i*3 + 0];
      pt[1] = pos_in_ptr[i*3 + 1];
      pt[2] = pos_in_ptr[i*3 + 2];
      pts.push_back(pt);

      for(size_t f = 0; f < nfields; f++)
        fields.push_back(fields_in_ptr[i * nfields + f]);

      orig_ids.push_back(i);
    }
  }

  npart = pts.size();

  if (vortrace::verbose) {
    std::cout << "Loaded " << npart << " particles";
    if (do_filter)
      std::cout << " (filtered from " << npart_in << ", padding: " << pad << ")";
    std::cout << "\nSnapshot loaded." << std::endl;
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
  size_t best_idx;
  Float best_r2;
  tree->knnSearch(query_pt.data(), 1, &best_idx, &best_r2);

  if (!periodic) return best_idx;

  Float d = std::sqrt(best_r2);

  // Wrap query into box-relative coords for boundary distance checks
  Float q_rel[3];
  for (int i = 0; i < 3; i++)
    q_rel[i] = query_pt[i] - subbox[2*i];

  for (int dx = -1; dx <= 1; dx++)
  for (int dy = -1; dy <= 1; dy++)
  for (int dz = -1; dz <= 1; dz++) {
    if (dx == 0 && dy == 0 && dz == 0) continue;

    // Skip this image if query is far from the relevant boundaries
    if (dx != 0 && std::min(q_rel[0], box_size[0] - q_rel[0]) > d) continue;
    if (dy != 0 && std::min(q_rel[1], box_size[1] - q_rel[1]) > d) continue;
    if (dz != 0 && std::min(q_rel[2], box_size[2] - q_rel[2]) > d) continue;

    Point shifted = {
      query_pt[0] + dx * box_size[0],
      query_pt[1] + dy * box_size[1],
      query_pt[2] + dz * box_size[2]
    };

    size_t idx;
    Float r2;
    tree->knnSearch(shifted.data(), 1, &idx, &r2);
    if (r2 < best_r2) {
      best_idx = idx;
      best_r2 = r2;
      d = std::sqrt(best_r2);
    }
  }

  return best_idx;
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
