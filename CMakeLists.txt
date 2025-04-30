cmake_minimum_required(VERSION 3.15)

option(BUILD_CPP_TESTS "Enable C++ unit tests (Catch2)" OFF)

# ensure we search the active conda env first
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

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

# --- find Python & pybind11 -----------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- your headers and sources --------------------
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_CPP ${CMAKE_SOURCE_DIR}/src/*.cpp)

# 1) build a real static library from your sources
add_library(vortrace_core STATIC ${SRC_CPP})
target_include_directories(vortrace_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(vortrace_core PUBLIC cxx_std_11)
target_compile_definitions(vortrace_core PRIVATE TIMING_INFO)
# tell your core library about pybind11’s include dirs:
target_link_libraries(vortrace_core
  PUBLIC
    pybind11::headers
    Python3::Python
)


# --- build the Python extension ------------------
pybind11_add_module(Cvortrace ${SRC_CPP})

# Build the extension directly into the vortrace package so that
# `from .Cvortrace import …` just works in‑place.
set_target_properties(Cvortrace PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/vortrace/vortrace
)

target_link_libraries(Cvortrace PRIVATE vortrace_core)

# --- (optional) add your Catch2/GTest tests -------
# find_package(Catch2 REQUIRED)
# add_executable(test_cpp tests/test_cpp.cpp)
# target_link_libraries(test_ray PRIVATE Catch2::Catch2 Cvortrace)
# add_test(NAME CPPTests COMMAND test_cpp)

# --- C++ unit tests with Catch2 --------------------
if(BUILD_CPP_TESTS)
  enable_testing()

  find_package(Catch2 3.1.0 REQUIRED)    # require Catch2 v3.1 or newer

  add_executable(test_cpp
    tests/test_cpp.cpp
  )

  target_include_directories(test_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )

  target_link_libraries(test_cpp PRIVATE
    Catch2::Catch2WithMain
    vortrace_core
  )

  add_test(NAME CPPRayTests COMMAND test_cpp)
endif()

# --- install rule so `cmake --install` works -------
install(TARGETS Cvortrace 
        LIBRARY DESTINATION ${Python3_SITEARCH}/vortrace)