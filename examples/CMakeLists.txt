cmake_minimum_required(VERSION 3.15)
project(vortrace_examples LANGUAGES CXX)

find_package(vortrace REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# Each example is a standalone executable
set(EXAMPLES
    basic_projection
    multifield
    volume_rendering
    single_ray
    slicing
    periodic
)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE vortrace::vortrace_core ${HDF5_C_LIBRARIES})
    target_include_directories(${example} PRIVATE ${HDF5_INCLUDE_DIRS})
    target_compile_features(${example} PRIVATE cxx_std_17)
endforeach()
