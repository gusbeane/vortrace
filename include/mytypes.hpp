#ifndef MYTYPES_HPP
#define MYTYPES_HPP

#include <array>
#include <vector>

using Float = double;
using Point = std::array<Float,3>;

constexpr double BOX_PAD_FRACTION = 0.15;

namespace vortrace {
  inline bool verbose = false;
}

#endif //MYTYPES_HPP