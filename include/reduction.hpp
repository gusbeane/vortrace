#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "mytypes.hpp"
#include "ray.hpp"

#include <vector>

// Built-in reduction functions operating on walked ray segments.
// Each returns a vector of length cloud.get_nfields().

std::vector<Float> reduce_sum(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud);

std::vector<Float> reduce_max(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud);

std::vector<Float> reduce_min(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud);

// Dispatch by ReductionMode enum (backward compatibility).
std::vector<Float> reduce(const std::vector<Ray::Segment> &segments,
                          const PointCloud &cloud,
                          ReductionMode mode);

#endif // REDUCTION_HPP
