#ifndef MYTYPES_HPP
#define MYTYPES_HPP

#include <array>
#include <vector>

using Float = double;
using Point = std::array<Float,3>;

enum class ReductionMode { Sum = 0, Max = 1, Min = 2 };

namespace vortrace {
  inline bool verbose = false;
}

#endif //MYTYPES_HPP