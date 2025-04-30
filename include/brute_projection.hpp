#ifndef BRUTE_PROJ_HPP
#define BRUTE_PROJ_HPP

#include "pointcloud.hpp"
#include "mytypes.hpp"

class BruteProjection
{
  private:
    std::array<size_t,3> npix;
    std::array<Float,6> extent;

    std::vector<Float> dens_proj;

  public:
    BruteProjection(std::array<size_t,3> npix, std::array<Float,6> extent) : 
      npix(npix), extent(extent) {}

    void makeProjection(const PointCloud &cloud);
    void saveProjection(const std::string savename) const;

};

#endif