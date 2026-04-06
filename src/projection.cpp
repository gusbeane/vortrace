
#include "projection.hpp"
#include "reduction.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <stdexcept>

Projection::Projection(const Float* pos_start, const Float* pos_end, size_t ngrid_in)
  : ngrid(ngrid_in), nfields(0)
{
  pts_start.resize(ngrid);
  pts_end.resize(ngrid);

  for(size_t i=0; i<ngrid; i++)
  {
    for(int j=0; j<3; j++)
    {
      pts_start[i][j] = pos_start[i*3 + j];
      pts_end[i][j]   = pos_end[i*3 + j];
    }
  }
}

void Projection::makeProjection(const PointCloud &cloud, ReductionMode mode)
{

  if(!cloud.get_tree_built())
  {
    throw std::runtime_error("There is currently no valid tree for this point cloud");
  }

  nfields = reduce_output_size(mode, cloud.get_nfields());

  //resize and zero result vector(s)
  proj_data.resize(ngrid * nfields);
  std::fill(proj_data.begin(), proj_data.end(), 0.0);

  if (vortrace::verbose) std::cout << "Making projection...\n";
  auto start = std::chrono::high_resolution_clock::now();
  std::exception_ptr eptr = nullptr;
  #pragma omp parallel for schedule(dynamic,256)
  for(size_t i = 0; i < ngrid; i++)
  {
    if (eptr) continue;
    try {
        Ray projray(pts_start[i], pts_end[i]);
        projray.integrate(cloud, mode);

        const std::vector<Float> *src;
        switch (mode) {
          case ReductionMode::Max:          src = &projray.get_max_val(); break;
          case ReductionMode::Min:          src = &projray.get_min_val(); break;
          case ReductionMode::VolumeRender: src = &projray.get_vol_render_val(); break;
          default:                          src = &projray.get_col();     break;
        }

        for(size_t f = 0; f < nfields; f++)
          proj_data[i * nfields + f] = (*src)[f];
    } catch (...) {
      #pragma omp critical
      { if (!eptr) eptr = std::current_exception(); }
    }
  }
  if (eptr) std::rethrow_exception(eptr);

  if (vortrace::verbose) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Projection generation took " << duration.count() << " milliseconds." << std::endl;
  }
}

void Projection::makeDetailedProjection(const PointCloud &cloud, ReductionMode mode)
{
  if(!cloud.get_tree_built())
  {
    throw std::runtime_error("There is currently no valid tree for this point cloud");
  }

  nfields = reduce_output_size(mode, cloud.get_nfields());

  proj_data.resize(ngrid * nfields);
  std::fill(proj_data.begin(), proj_data.end(), 0.0);

  // Per-ray segment storage, filled in parallel.
  std::vector<std::vector<Ray::Segment>> all_segments(ngrid);

  if (vortrace::verbose) std::cout << "Making detailed projection...\n";
  auto start = std::chrono::high_resolution_clock::now();
  std::exception_ptr eptr = nullptr;
  #pragma omp parallel for schedule(dynamic,256)
  for(size_t i = 0; i < ngrid; i++)
  {
    if (eptr) continue;
    try {
        Ray projray(pts_start[i], pts_end[i]);
        projray.integrate(cloud, mode);

        const std::vector<Float> *src;
        switch (mode) {
          case ReductionMode::Max:          src = &projray.get_max_val(); break;
          case ReductionMode::Min:          src = &projray.get_min_val(); break;
          case ReductionMode::VolumeRender: src = &projray.get_vol_render_val(); break;
          default:                          src = &projray.get_col();     break;
        }

        for(size_t f = 0; f < nfields; f++)
          proj_data[i * nfields + f] = (*src)[f];

        all_segments[i] = projray.take_segments();
    } catch (...) {
      #pragma omp critical
      { if (!eptr) eptr = std::current_exception(); }
    }
  }
  if (eptr) std::rethrow_exception(eptr);

  // Build flat offset-indexed arrays from per-ray segments.
  seg_offsets.resize(ngrid + 1);
  seg_offsets[0] = 0;
  for (size_t i = 0; i < ngrid; i++) {
    seg_offsets[i + 1] = seg_offsets[i] + all_segments[i].size();
  }
  size_t total = seg_offsets[ngrid];
  seg_cell_ids.resize(total);
  seg_s_enter.resize(total);
  seg_s_exit.resize(total);
  for (size_t i = 0; i < ngrid; i++) {
    size_t off = seg_offsets[i];
    for (size_t j = 0; j < all_segments[i].size(); j++) {
      seg_cell_ids[off + j] = all_segments[i][j].cell_id;
      seg_s_enter[off + j]  = all_segments[i][j].s_enter;
      seg_s_exit[off + j]   = all_segments[i][j].s_exit;
    }
  }

  if (vortrace::verbose) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Detailed projection generation took " << duration.count() << " milliseconds." << std::endl;
  }
}
