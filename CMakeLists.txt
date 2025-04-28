cmake_minimum_required(VERSION 3.14)
project(vortrace LANGUAGES CXX)

# find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ask pip‑installed pybind11 where its CMake files live
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE _PYBIND11_CMAKEDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${_PYBIND11_CMAKEDIR}")

# enable tests
enable_testing()

# --- find Python & pybind11 -----------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- your headers and sources --------------------
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_CPP ${CMAKE_SOURCE_DIR}/src/*.cpp)

# --- build the Python extension ------------------
pybind11_add_module(Cvortrace ${SRC_CPP})
target_compile_features(Cvortrace PRIVATE cxx_std_11)
target_compile_definitions(Cvortrace PRIVATE TIMING_INFO)

# --- (optional) add your Catch2/GTest tests -------
# find_package(Catch2 REQUIRED)
# add_executable(test_ray tests/test_ray.cpp)
# target_link_libraries(test_ray PRIVATE Catch2::Catch2 Cvortrace)
# add_test(NAME RayTests COMMAND test_ray)

# --- install rule so `cmake --install` works -------
install(TARGETS Cvortrace LIBRARY DESTINATION ${Python3_SITEARCH})