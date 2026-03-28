#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "mytypes.hpp"
#include "ray.hpp"

#include <vector>

// Built-in reduction functions operating on walked ray segments.
// Each returns a vector of length cloud.get_nfields(), except
// reduce_volume_render which returns 3 (RGB).

std::vector<Float> reduce_sum(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud);

std::vector<Float> reduce_max(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud);

std::vector<Float> reduce_min(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud);

std::vector<Float> reduce_volume_render(const std::vector<Ray::Segment> &segments,
                                        const PointCloud &cloud);

// Dispatch by ReductionMode enum (backward compatibility).
std::vector<Float> reduce(const std::vector<Ray::Segment> &segments,
                          const PointCloud &cloud,
                          ReductionMode mode);

// Returns the number of output values for a given reduction mode.
size_t reduce_output_size(ReductionMode mode, size_t cloud_nfields);

#endif // REDUCTION_HPP
