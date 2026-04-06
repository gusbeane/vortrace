#ifndef PROJ_HPP
#define PROJ_HPP

#include "pointcloud.hpp"
#include "ray.hpp"
#include "mytypes.hpp"

class Projection
{
  private:
    std::vector<Point> pts_end;
    std::vector<Point> pts_start;

    size_t ngrid;    // Length of pts_start and pts_end.
    size_t nfields;  // Number of fields per ray.
    std::vector<Float> proj_data;  // flat, ngrid * nfields

    // Detailed (segment-level) output, populated by makeDetailedProjection.
    std::vector<size_t> seg_cell_ids;  // flat concatenated cell IDs
    std::vector<Float>  seg_s_enter;   // flat concatenated entry distances
    std::vector<Float>  seg_s_exit;    // flat concatenated exit distances
    std::vector<size_t> seg_offsets;   // CSR offsets, length ngrid+1

  public:
    Projection(const Float* pos_start, const Float* pos_end, size_t ngrid);

    void makeProjection(const PointCloud &cloud, ReductionMode reduction = ReductionMode::Sum);
    void makeDetailedProjection(const PointCloud &cloud, ReductionMode reduction = ReductionMode::Sum);

    const std::vector<Float>& getProjectionData() const { return proj_data; }
    size_t getNgrid() const { return ngrid; }
    size_t getNfields() const { return nfields; }

    const std::vector<size_t>& getSegCellIds() const { return seg_cell_ids; }
    const std::vector<Float>&  getSegSEnter()  const { return seg_s_enter; }
    const std::vector<Float>&  getSegSExit()   const { return seg_s_exit; }
    const std::vector<size_t>& getSegOffsets() const { return seg_offsets; }
};

#endif