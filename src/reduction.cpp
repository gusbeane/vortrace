#include "reduction.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

std::vector<Float> reduce_volume_render(const std::vector<Ray::Segment> &segments,
                                        const PointCloud &cloud)
{
  size_t nf = cloud.get_nfields();
  if (nf != 4)
    throw std::runtime_error(
      "VolumeRender requires exactly 4 fields (R, G, B, alpha), got "
      + std::to_string(nf));

  Float T = 1.0;  // transmittance
  std::vector<Float> color(3, 0.0);

  for (const auto &seg : segments) {
    Float alpha = cloud.get_field(seg.cell_id, 3);
    Float ds = seg.ds();
    Float opacity = 1.0 - std::exp(-alpha * ds);

    for (size_t c = 0; c < 3; c++) {
      color[c] += T * cloud.get_field(seg.cell_id, c) * opacity;
    }
    T *= (1.0 - opacity);
  }

  return color;
}

std::vector<Float> reduce(const std::vector<Ray::Segment> &segments,
                          const PointCloud &cloud,
                          ReductionMode mode)
{
  switch (mode) {
    case ReductionMode::Sum: return reduce_sum(segments, cloud);
    case ReductionMode::Max: return reduce_max(segments, cloud);
    case ReductionMode::Min: return reduce_min(segments, cloud);
    case ReductionMode::VolumeRender: return reduce_volume_render(segments, cloud);
    default: throw std::runtime_error("Unrecognized reduction mode");
  }
  return reduce_sum(segments, cloud); // unreachable, silences compiler warning
}

size_t reduce_output_size(ReductionMode mode, size_t cloud_nfields) {
  if (mode == ReductionMode::VolumeRender) return 3;
  return cloud_nfields;
}
