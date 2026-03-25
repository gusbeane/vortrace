#include "pointcloud.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

#define TOLERANCE 1E-9

namespace py = pybind11;

void PointCloud::loadPoints(py::array_t<double> pos_in, py::array_t<double> fields_in,
                            const std::array<Float,6> newsubbox,
                            py::array_t<double> vol,
                            bool periodic_in)
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

  if (periodic) {
    box_size[0] = subbox[1] - subbox[0];
    box_size[1] = subbox[3] - subbox[2];
    box_size[2] = subbox[5] - subbox[4];
  }

  pts.clear();
  fields.clear();
  orig_ids.clear();

  // Warn if any cells are small enough to cause tolerance issues
  {
    py::buffer_info buf_vol_check = vol.request();
    if (buf_vol_check.size > 0) {
      auto *vptr = static_cast<double *>(buf_vol_check.ptr);
      double min_vol = std::numeric_limits<double>::max();
      for (ssize_t i = 0; i < buf_vol_check.size; i++) {
        if (vptr[i] > 0 && vptr[i] < min_vol) min_vol = vptr[i];
      }
      if (min_vol < std::numeric_limits<double>::max()) {
        double min_radius = std::cbrt(3.0 * min_vol / (4.0 * M_PI));
        if (min_radius < 1e-6) {
          PyErr_WarnEx(PyExc_UserWarning,
            "Some cells have radii smaller than 1e-6, which approaches the "
            "internal ray-tracing tolerance (1e-9). Results may be unreliable "
            "for very small cells. Consider rescaling your coordinate system.", 1);
        }
      }
    }
  }

  if (!periodic) {
    // Compute padding
    Float dx = subbox[1] - subbox[0];
    Float dy = subbox[3] - subbox[2];
    Float dz = subbox[5] - subbox[4];

    py::buffer_info buf_vol = vol.request();
    if (buf_vol.size > 0) {
      // Adaptive padding from cell volumes
      auto *vol_ptr = static_cast<double *>(buf_vol.ptr);
      double max_vol = 0.0;
      for (ssize_t i = 0; i < buf_vol.size; i++) {
        if (vol_ptr[i] > max_vol) max_vol = vol_ptr[i];
      }
      double max_radius = std::cbrt(3.0 * max_vol / (4.0 * M_PI));
      pad = 10.0 * max_radius;
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
    // Periodic: load all particles, no filtering
    pad = 0.0;

    if (vortrace::verbose) std::cout << "Loading all points (periodic)...\n";

    // Track particle extent for validation
    double pmin[3] = {std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max()};
    double pmax[3] = {std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest()};

    for(size_t i = 0; i < npart_in; i++)
    {
      Point pt;
      pt[0] = pos_in_ptr[i*3 + 0];
      pt[1] = pos_in_ptr[i*3 + 1];
      pt[2] = pos_in_ptr[i*3 + 2];
      pts.push_back(pt);

      for (int d = 0; d < 3; d++) {
        if (pt[d] < pmin[d]) pmin[d] = pt[d];
        if (pt[d] > pmax[d]) pmax[d] = pt[d];
      }

      for(size_t f = 0; f < nfields; f++)
        fields.push_back(fields_in_ptr[i * nfields + f]);

      orig_ids.push_back(i);
    }

    // Error if any particles lie outside the bounding box
    if (pmin[0] < subbox[0] || pmax[0] > subbox[1] ||
        pmin[1] < subbox[2] || pmax[1] > subbox[3] ||
        pmin[2] < subbox[4] || pmax[2] > subbox[5]) {
      throw std::runtime_error(
        "Periodic mode: some particles lie outside the bounding box. "
        "All particles must be within the periodic domain.");
    }

    // Warn if particle extent is small relative to box size
    const double extent_threshold = 0.6;
    for (int d = 0; d < 3; d++) {
      double extent = pmax[d] - pmin[d];
      if (extent < extent_threshold * box_size[d]) {
        PyErr_WarnEx(PyExc_UserWarning,
          "Periodic mode: particle extent is less than 60% of the bounding "
          "box in at least one dimension. The bounding box may be too large "
          "for the particle distribution.", 1);
        break;
      }
    }
  }

  npart = pts.size();

  if (vortrace::verbose) {
    if (!periodic)
      std::cout << "Filtered " << npart_in << " -> " << npart << " particles\n";
    else
      std::cout << "Loaded " << npart << " particles (periodic, no filtering)\n";
    std::cout << "Padding: " << pad << "\n";
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

Float PointCloud::minDistSqToBox(const Point &query_pt) const
{
  Float d2 = 0.0;
  for (int dim = 0; dim < 3; dim++) {
    Float lo = subbox[2*dim];
    Float hi = subbox[2*dim + 1];
    if (query_pt[dim] < lo) {
      Float d = lo - query_pt[dim];
      d2 += d * d;
    } else if (query_pt[dim] > hi) {
      Float d = query_pt[dim] - hi;
      d2 += d * d;
    }
  }
  return d2;
}

size_t PointCloud::queryTree(const Point &query_pt) const
{
  size_t result;
  Float r2;
  queryTreeK(query_pt, 1, &result, &r2);
  return result;
}

void PointCloud::queryTreeK(const Point &query_pt, size_t k,
                            size_t *results, Float *r2) const
{
  tree->knnSearch(query_pt.data(), k, results, r2);

  if (!periodic) return;

  // Periodic: check up to 27 images, merge results
  struct IdDist { size_t id; Float dist2; };
  std::vector<IdDist> merged(k);
  for (size_t i = 0; i < k; i++)
    merged[i] = {results[i], r2[i]};

  std::vector<size_t> img_ids(k);
  std::vector<Float> img_r2(k);

  for (int ix = -1; ix <= 1; ix++) {
    for (int iy = -1; iy <= 1; iy++) {
      for (int iz = -1; iz <= 1; iz++) {
        if (ix == 0 && iy == 0 && iz == 0) continue;

        Point shifted = {
          query_pt[0] + ix * box_size[0],
          query_pt[1] + iy * box_size[1],
          query_pt[2] + iz * box_size[2]
        };

        Float minD2 = minDistSqToBox(shifted);
        if (minD2 > merged[k-1].dist2 + TOLERANCE) continue;

        tree->knnSearch(shifted.data(), k, img_ids.data(), img_r2.data());

        // Merge: deduplicate by particle ID, keep min distance
        for (size_t i = 0; i < k; i++) {
          bool found = false;
          for (auto &m : merged) {
            if (m.id == img_ids[i]) {
              m.dist2 = std::min(m.dist2, img_r2[i]);
              found = true;
              break;
            }
          }
          if (!found) {
            merged.push_back({img_ids[i], img_r2[i]});
          }
        }

        // Sort and truncate to k
        std::sort(merged.begin(), merged.end(),
                  [](const IdDist &a, const IdDist &b){ return a.dist2 < b.dist2; });
        if (merged.size() > k) merged.resize(k);
      }
    }
  }

  // Write back sorted results
  for (size_t i = 0; i < k; i++) {
    results[i] = merged[i].id;
    r2[i] = merged[i].dist2;
  }
}
