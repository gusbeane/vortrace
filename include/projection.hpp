#ifndef PROJ_HPP
#define PROJ_HPP

#include "pointcloud.hpp"
#include "mytypes.hpp"

class Projection
{
  private:
    std::array<size_t,2> npix;
    std::array<MyFloat,6> extent;

    std::vector<MyFloat> dens_proj;

  public:
    Projection(std::array<size_t,2> npix, std::array<MyFloat,6> extent) : 
      npix(npix), extent(extent) {}

    void makeProjection(const PointCloud &cloud);
    void saveProjection(const std::string savename) const;

};

#endif