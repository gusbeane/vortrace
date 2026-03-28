#ifndef MYTYPES_HPP
#define MYTYPES_HPP

#include <array>
#include <atomic>
#include <vector>

using Float = double;
using Point = std::array<Float,3>;

enum class ReductionMode { Sum = 0, Max = 1, Min = 2, VolumeRender = 3 };

namespace vortrace {
  inline std::atomic<bool> verbose{false};
}

#endif //MYTYPES_HPP