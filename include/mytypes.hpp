#ifndef MYTYPES_HPP
#define MYTYPES_HPP

#include <array>
#include <atomic>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using Float = double;
using Point = std::array<Float,3>;

enum class ReductionMode { Sum = 0, Max = 1, Min = 2, VolumeRender = 3 };

namespace vortrace {
  inline std::atomic<bool> verbose{false};

  using WarningCallback = std::function<void(const std::string&)>;
  inline WarningCallback warning_handler = [](const std::string& msg) {
      std::cerr << "vortrace warning: " << msg << std::endl;
  };
  inline void warn(const std::string& msg) {
      if (warning_handler) warning_handler(msg);
  }
}

#endif //MYTYPES_HPP