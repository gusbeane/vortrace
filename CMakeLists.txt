cmake_minimum_required(VERSION 3.15)

option(BUILD_CPP_TESTS "Enable C++ unit tests (Catch2)" OFF)
option(BUILD_PYTHON_BINDINGS "Build the pybind11 Python extension" ON)

# ensure we search the active conda env first
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

project(vortrace LANGUAGES CXX)

# --- find OpenMP (optional) ---
find_package(OpenMP)

# --- your headers and sources --------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CORE_CPP
  ${CMAKE_SOURCE_DIR}/src/pointcloud.cpp
  ${CMAKE_SOURCE_DIR}/src/ray.cpp
  ${CMAKE_SOURCE_DIR}/src/reduction.cpp
  ${CMAKE_SOURCE_DIR}/src/projection.cpp
  ${CMAKE_SOURCE_DIR}/src/brute_projection.cpp
  ${CMAKE_SOURCE_DIR}/src/slice.cpp
)

# 1) build a real static library from core sources (excludes bindings)
add_library(vortrace_core STATIC ${CORE_CPP})
set_target_properties(vortrace_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vortrace_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/vortrace>
)
target_compile_features(vortrace_core PUBLIC cxx_std_17)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vortrace_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- build the Python extension from bindings only --
if(BUILD_PYTHON_BINDINGS)
  # find Python
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  # ask pip-installed pybind11 where its CMake files live
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  list(APPEND CMAKE_PREFIX_PATH "${_PYBIND11_CMAKEDIR}")

  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(Cvortrace ${CMAKE_SOURCE_DIR}/src/bindings.cpp)

  # Build the extension directly into the vortrace package so that
  # `from .Cvortrace import …` just works in-place.
  set_target_properties(Cvortrace PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/vortrace
  )

  target_link_libraries(Cvortrace PRIVATE vortrace_core pybind11::headers Python3::Python)

  # --- install rule for the Python extension -------
  install(TARGETS Cvortrace
          LIBRARY DESTINATION ${Python3_SITEARCH}/vortrace)
endif()

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

  add_test(NAME CPPTests COMMAND test_cpp)
endif()

# --- C++ library install rules ---------------------
include(GNUInstallDirs)

install(TARGETS vortrace_core
    EXPORT vortraceTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vortrace
    FILES_MATCHING PATTERN "*.hpp"
)
install(EXPORT vortraceTargets
    FILE vortraceTargets.cmake
    NAMESPACE vortrace::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vortrace
)

# --- CMake package config -------------------------
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/vortraceConfig.cmake.in
    ${CMAKE_BINARY_DIR}/vortraceConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vortrace
)
install(FILES ${CMAKE_BINARY_DIR}/vortraceConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vortrace
)
