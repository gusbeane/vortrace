#include "pointcloud.hpp"
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <stdexcept>

#define TOLERANCE 1E-9

void PointCloud::loadPoints(const double* pos_in, size_t npart_in,
                            const double* fields_in, size_t npart_fields, size_t nfields_in,
                            const std::array<Float,6>& newsubbox,
                            const double* vol, size_t nvol,
                            bool periodic_in)
{
  if (npart_in != npart_fields) {
    throw std::invalid_argument("Input sizes must match: pos has " +
      std::to_string(npart_in) + " rows but fields has " +
      std::to_string(npart_fields));
  }

  nfields = nfields_in;
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
  if (nvol > 0) {
    double min_vol = std::numeric_limits<double>::max();
    for (size_t i = 0; i < nvol; i++) {
      if (vol[i] > 0 && vol[i] < min_vol) min_vol = vol[i];
    }
    if (min_vol < std::numeric_limits<double>::max()) {
      double min_radius = std::cbrt(3.0 * min_vol / (4.0 * M_PI));
      if (min_radius < 1e-6) {
        vortrace::warn(
          "Some cells have radii smaller than 1e-6, which approaches the "
          "internal ray-tracing tolerance (1e-9). Results may be unreliable "
          "for very small cells. Consider rescaling your coordinate system.");
      }
    }
  }

  if (!periodic) {
    // Compute padding
    Float dx = subbox[1] - subbox[0];
    Float dy = subbox[3] - subbox[2];
    Float dz = subbox[5] - subbox[4];

    if (nvol > 0) {
      // Adaptive padding from cell volumes
      double max_vol = 0.0;
      for (size_t i = 0; i < nvol; i++) {
        if (vol[i] > max_vol) max_vol = vol[i];
      }
      double max_radius = std::cbrt(3.0 * max_vol / (4.0 * M_PI));
      pad = 10.0 * max_radius;
    } else {
      // Fallback: uniform 15% of longest box side
      pad = 0.15 * std::max({dx, dy, dz});
      vortrace::warn(
        "No cell volumes provided; using 15% of longest box side as "
        "padding. Pass vol= for adaptive padding.");
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
      double px = pos_in[i*3 + 0];
      double py_val = pos_in[i*3 + 1];
      double pz = pos_in[i*3 + 2];

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
          fields.push_back(fields_in[i * nfields + f]);

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
      pt[0] = pos_in[i*3 + 0];
      pt[1] = pos_in[i*3 + 1];
      pt[2] = pos_in[i*3 + 2];
      pts.push_back(pt);

      for (int d = 0; d < 3; d++) {
        if (pt[d] < pmin[d]) pmin[d] = pt[d];
        if (pt[d] > pmax[d]) pmax[d] = pt[d];
      }

      for(size_t f = 0; f < nfields; f++)
        fields.push_back(fields_in[i * nfields + f]);

      orig_ids.push_back(i);
    }

    // Error if any particles lie outside the bounding box
    if (pmin[0] < subbox[0] || pmax[0] > subbox[1] ||
        pmin[1] < subbox[2] || pmax[1] > subbox[3] ||
        pmin[2] < subbox[4] || pmax[2] > subbox[5]) {
      throw std::invalid_argument(
        "Periodic mode: some particles lie outside the bounding box. "
        "All particles must be within the periodic domain.");
    }

    // Warn if particle extent is small relative to box size
    const double extent_threshold = 0.6;
    for (int d = 0; d < 3; d++) {
      double extent = pmax[d] - pmin[d];
      if (extent < extent_threshold * box_size[d]) {
        vortrace::warn(
          "Periodic mode: particle extent is less than 60% of the bounding "
          "box in at least one dimension. The bounding box may be too large "
          "for the particle distribution.");
        break;
      }
    }
  }

  npart = pts.size();

  // Check if fields constitute valid RGBA for volume rendering
  valid_rgba = false;
  if (nfields == 4 && npart > 0) {
    valid_rgba = true;
    for (size_t i = 0; i < npart; i++) {
      for (size_t c = 0; c < 3; c++) {
        Float val = fields[i * nfields + c];
        if (val < 0.0 || val > 1.0) { valid_rgba = false; break; }
      }
      if (!valid_rgba) break;
      if (fields[i * nfields + 3] < 0.0) { valid_rgba = false; break; }
    }
  }

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

void PointCloud::saveTree(const std::string& filename) const
{
  if (!tree_built) {
    throw std::runtime_error("Cannot save tree: tree has not been built");
  }
  FILE* f = fopen(filename.c_str(), "wb");
  if (!f) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  tree->saveIndex(f);
  fclose(f);
}

void PointCloud::loadTree(const std::string& filename)
{
  if (npart == 0) {
    throw std::runtime_error("Cannot load tree: no points loaded");
  }
  FILE* f = fopen(filename.c_str(), "rb");
  if (!f) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }
  tree.reset(new my_kd_tree_t(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  tree->loadIndex(f);
  fclose(f);
  tree_built = true;
}

std::vector<char> PointCloud::saveTreeToBuffer() const
{
  if (!tree_built) {
    throw std::runtime_error("Cannot save tree: tree has not been built");
  }
  FILE* f = tmpfile();
  if (!f) {
    throw std::runtime_error("Failed to create temporary file");
  }
  tree->saveIndex(f);
  long size = ftell(f);
  rewind(f);
  std::vector<char> buf(size);
  if (fread(buf.data(), 1, size, f) != static_cast<size_t>(size)) {
    fclose(f);
    throw std::runtime_error("Failed to read tree data from temporary file");
  }
  fclose(f);
  return buf;
}

void PointCloud::loadTreeFromBuffer(const char* data, size_t size)
{
  if (npart == 0) {
    throw std::runtime_error("Cannot load tree: no points loaded");
  }
  FILE* f = tmpfile();
  if (!f) {
    throw std::runtime_error("Failed to create temporary file");
  }
  if (fwrite(data, 1, size, f) != size) {
    fclose(f);
    throw std::runtime_error("Failed to write tree data to temporary file");
  }
  rewind(f);
  tree.reset(new my_kd_tree_t(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  tree->loadIndex(f);
  fclose(f);
  tree_built = true;
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
