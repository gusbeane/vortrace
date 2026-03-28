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

  public:
    Projection(const Float* pos_start, const Float* pos_end, size_t ngrid);

    void makeProjection(const PointCloud &cloud, ReductionMode reduction = ReductionMode::Sum);

    const std::vector<Float>& getProjectionData() const { return proj_data; }
    size_t getNgrid() const { return ngrid; }
    size_t getNfields() const { return nfields; }
};

#endif