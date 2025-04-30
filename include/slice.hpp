#ifndef SLICE_HPP
#define SLICE_HPP

#include "pointcloud.hpp"
#include "mytypes.hpp"

class Slice
{
  private:
    std::array<size_t,2> npix;
    std::array<Float,4> extent;
    Float depth;

    std::vector<Float> dens_slice;

  public:
    Slice(std::array<size_t,2> npix, std::array<Float,4> extent, Float depth) : 
      npix(npix), extent(extent), depth(depth) {}

    void makeSlice(const PointCloud &cloud);
    void saveSlice(const std::string savename) const;

};

#endif