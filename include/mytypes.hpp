#ifndef MYTYPES_HPP
#define MYTYPES_HPP

#include <array>
#include <vector>

#define BOX_PAD_MIN 0.85
#define BOX_PAD_MAX 1.15

#ifdef DOUBLE_PRECISION
typedef double MyFloat;
#else
typedef double MyFloat;
#endif

typedef std::array<MyFloat,3> cartarr_t;

#endif //MYTYPES_HPP