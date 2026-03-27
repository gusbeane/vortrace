#include "reduction.hpp"

#include <limits>

std::vector<Float> reduce_sum(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud)
{
  size_t nf = cloud.get_nfields();
  std::vector<Float> result(nf, 0.0);

  for (const auto &seg : segments) {
    for (size_t f = 0; f < nf; f++) {
      result[f] += seg.ds() * cloud.get_field(seg.cell_id, f);
    }
  }

  return result;
}

std::vector<Float> reduce_max(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud)
{
  size_t nf = cloud.get_nfields();
  std::vector<Float> result(nf, -std::numeric_limits<Float>::infinity());

  for (const auto &seg : segments) {
    for (size_t f = 0; f < nf; f++) {
      Float val = cloud.get_field(seg.cell_id, f);
      if (val > result[f]) result[f] = val;
    }
  }

  return result;
}

std::vector<Float> reduce_min(const std::vector<Ray::Segment> &segments,
                              const PointCloud &cloud)
{
  size_t nf = cloud.get_nfields();
  std::vector<Float> result(nf, std::numeric_limits<Float>::infinity());

  for (const auto &seg : segments) {
    for (size_t f = 0; f < nf; f++) {
      Float val = cloud.get_field(seg.cell_id, f);
      if (val < result[f]) result[f] = val;
    }
  }

  return result;
}

std::vector<Float> reduce(const std::vector<Ray::Segment> &segments,
                          const PointCloud &cloud,
                          ReductionMode mode)
{
  switch (mode) {
    case ReductionMode::Sum: return reduce_sum(segments, cloud);
    case ReductionMode::Max: return reduce_max(segments, cloud);
    case ReductionMode::Min: return reduce_min(segments, cloud);
    default: throw std::runtime_error("Unrecognized reduction mode");
  }
  return reduce_sum(segments, cloud); // unreachable, silences compiler warning
}
