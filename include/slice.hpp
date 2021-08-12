#ifndef SLICE_HPP
#define SLICE_HPP

#include "pointcloud.hpp"
#include "mytypes.hpp"

class Slice
{
  private:
    std::array<size_t,2> npix;
    std::array<MyFloat,4> extent;
    MyFloat depth;

    std::vector<MyFloat> dens_slice;

  public:
    Slice(std::array<size_t,2> npix, std::array<MyFloat,4> extent, MyFloat depth) : 
      npix(npix), extent(extent), depth(depth) {}

    void makeSlice(const PointCloud &cloud);
    void saveSlice(const std::string savename) const;

};

#endif