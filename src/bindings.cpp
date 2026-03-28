#include "class_includes.hpp"
#include "ray.hpp"
#include "reduction.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(Cvortrace, m) {
    m.doc() = "C++ backend for vortrace: fast projections through Voronoi meshes.";

    // Map std::invalid_argument -> Python ValueError
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    m.def("set_verbose", [](bool v){ vortrace::verbose.store(v); },
          py::arg("verbose"),
          "Enable or disable verbose C++ status messages to stdout.");
    m.def("get_verbose", [](){ return vortrace::verbose.load(); },
          "Return the current verbose setting.");

    py::enum_<ReductionMode>(m, "ReductionMode",
        "Reduction strategy applied along each ray.")
        .value("Sum", ReductionMode::Sum,
               "Integrate (sum field * ds) along the ray.")
        .value("Max", ReductionMode::Max,
               "Take the maximum field value along the ray.")
        .value("Min", ReductionMode::Min,
               "Take the minimum field value along the ray.")
        .value("VolumeRender", ReductionMode::VolumeRender,
               "Front-to-back alpha compositing (requires 4 RGBA fields).")
        .export_values();

    py::class_<PointCloud>(m, "PointCloud",
        "Spatial container holding particle positions, fields, and a kD-tree.")
        .def(py::init<>(), "Create an empty PointCloud.")
        .def("loadPoints", &PointCloud::loadPoints,
             "Load particle positions and field data, filtering to the bounding box.",
             py::arg("pos"), py::arg("fields_in"),
             py::arg("subbox"), py::arg("vol") = py::array_t<double>(),
             py::arg("periodic") = false)
        .def("buildTree", &PointCloud::buildTree,
             "Build the kD-tree index for nearest-neighbor queries.")
        .def("get_nfields", &PointCloud::get_nfields,
             "Return the number of scalar fields per particle.")
        .def("get_pt", &PointCloud::get_pt,
             "Return the 3D position of particle at the given index.")
        .def("get_field", &PointCloud::get_field,
             "Return the value of field *f* for particle at index *idx*.")
        .def("get_subbox", &PointCloud::get_subbox,
             "Return the bounding box [xmin,xmax,ymin,ymax,zmin,zmax].")
        .def("get_tree_built", &PointCloud::get_tree_built,
             "Return True if the kD-tree has been built.")
        .def("get_orig_ids", &PointCloud::get_orig_ids,
             "Return the mapping from filtered indices to original particle indices.")
        .def("get_pad", &PointCloud::get_pad,
             "Return the padding distance used when filtering particles.")
        .def("get_npart", &PointCloud::get_npart,
             "Return the number of particles loaded (after filtering).")
        .def("get_periodic", &PointCloud::get_periodic,
             "Return True if periodic boundary conditions are enabled.")
        .def("get_valid_rgba", &PointCloud::get_valid_rgba,
             "Return True if the loaded fields are valid RGBA for volume rendering "
             "(4 fields with R,G,B in [0,1] and alpha >= 0).")
        .def("queryTree",
             [](const PointCloud &cloud, const Point &query_pt, size_t k) -> py::object {
               if (k == 1) {
                 return py::cast(cloud.queryTree(query_pt));
               }
               std::vector<size_t> ids(k);
               std::vector<Float> r2(k);
               cloud.queryTreeK(query_pt, k, ids.data(), r2.data());
               return py::make_tuple(
                 py::array_t<size_t>(k, ids.data()),
                 py::array_t<Float>(k, r2.data()));
             },
             "Query the kD-tree for the *k* nearest neighbors of a point.",
             py::arg("query_pt"), py::arg("k") = 1);

    py::class_<Projection>(m, "Projection",
        "Ray-based projection through a point cloud.")
        .def(py::init<
            py::array_t<Float, py::array::c_style | py::array::forcecast>,
            py::array_t<Float, py::array::c_style | py::array::forcecast>>(),
             "Create a Projection from arrays of ray start and end points.")
        .def("makeProjection", &Projection::makeProjection,
             "Run the projection through the point cloud.",
             py::call_guard<py::gil_scoped_release>(),
             py::arg("cloud"), py::arg("reduction") = ReductionMode::Sum)
        .def("returnProjection", &Projection::returnProjection,
             "Return the projection result as a numpy array.");

    py::class_<BruteProjection>(m, "BruteProjection",
        "Grid-based brute-force projection (samples nearest cell at each voxel).")
        .def(py::init<std::array<size_t,3>, std::array<Float,6>>(),
             "Create a BruteProjection with given pixel counts and spatial extent.")
        .def("makeProjection", &BruteProjection::makeProjection,
             "Run the brute-force projection through the point cloud.",
             py::call_guard<py::gil_scoped_release>(),
             py::arg("cloud"), py::arg("reduction") = ReductionMode::Sum)
        .def("saveProjection", &BruteProjection::saveProjection,
             "Save the projection result to a text file.")
        .def("returnProjection", &BruteProjection::returnProjection,
             "Return the projection result as a numpy array.");

    py::class_<Slice>(m, "Slice",
        "2D slice extraction at a fixed depth.")
        .def(py::init<std::array<size_t,2>, std::array<Float,4>, Float>(),
             "Create a Slice with given pixel counts, 2D extent, and depth.")
        .def("makeSlice", &Slice::makeSlice,
             "Extract the 2D slice from the point cloud.",
             py::call_guard<py::gil_scoped_release>())
        .def("saveSlice", &Slice::saveSlice,
             "Save the slice result to a text file.")
        .def("returnSlice", &Slice::returnSlice,
             "Return the slice result as a numpy array.");

    py::class_<Ray>(m, "Ray",
        "A single ray between two points for integration through a Voronoi mesh.")
        .def(py::init<Point, Point>(),
             "Create a Ray from a start point to an end point.")
        .def("walk", &Ray::walk,
             "Trace the ray through the point cloud, populating segments.",
             py::arg("cloud"))
        .def("integrate", &Ray::integrate,
             "Walk the ray and apply a reduction to compute the result.",
             py::arg("cloud"), py::arg("mode") = ReductionMode::Sum)
        .def("get_dens_col", &Ray::get_dens_col,
             "Return the integrated column density (Sum mode, single field).")
        .def("get_col", &Ray::get_col,
             "Return the integrated column values (Sum mode, all fields).")
        .def("get_max_val", &Ray::get_max_val,
             "Return the maximum field values along the ray.")
        .def("get_min_val", &Ray::get_min_val,
             "Return the minimum field values along the ray.")
        .def("get_vol_render_val", &Ray::get_vol_render_val,
             "Return the volume-rendered RGB values (3 elements).")
        .def(
            "get_segments",
            [](const Ray &r){
              py::list out;
              for (auto &seg : r.get_segments()) {
                out.append(py::make_tuple(seg.cell_id, seg.s_enter, seg.s_exit, seg.ds()));
              }
              return out;
            },
            "Return the traced segments as a list of (cell_id, s_enter, s_exit, ds) tuples."
          );

    // Standalone reduction functions
    m.def("reduce_sum", &reduce_sum,
          "Apply Sum reduction to walked ray segments.",
          py::arg("segments"), py::arg("cloud"));
    m.def("reduce_max", &reduce_max,
          "Apply Max reduction to walked ray segments.",
          py::arg("segments"), py::arg("cloud"));
    m.def("reduce_min", &reduce_min,
          "Apply Min reduction to walked ray segments.",
          py::arg("segments"), py::arg("cloud"));
    m.def("reduce_volume_render", &reduce_volume_render,
          "Apply VolumeRender (front-to-back compositing) to walked ray segments.",
          py::arg("segments"), py::arg("cloud"));
    m.def("reduce", &reduce,
          "Dispatch a reduction by mode enum on walked ray segments.",
          py::arg("segments"), py::arg("cloud"), py::arg("mode"));

}
