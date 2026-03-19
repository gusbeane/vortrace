#ifndef BRUTE_PROJ_HPP
#define BRUTE_PROJ_HPP

#include "pointcloud.hpp"
#include "ray.hpp"
#include "mytypes.hpp"

class BruteProjection
{
  private:
    std::array<size_t,3> npix;
    std::array<Float,6> extent;

    size_t nfields;
    std::vector<Float> proj_data;  // flat, (npix_x * npix_y) * nfields

  public:
    BruteProjection(std::array<size_t,3> npix, std::array<Float,6> extent) :
      npix(npix), extent(extent), nfields(0) {}

    void makeProjection(const PointCloud &cloud, ReductionMode reduction = ReductionMode::Sum);
    void saveProjection(const std::string savename) const;

};

#endif