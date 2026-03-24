#include "class_includes.hpp"
#include "ray.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(Cvortrace, m) {
    m.def("set_verbose", [](bool v){ vortrace::verbose = v; },
          py::arg("verbose"),
          "Enable or disable C++ stdout messages");
    m.def("get_verbose", [](){ return vortrace::verbose; },
          "Return current verbosity setting");

    py::enum_<ReductionMode>(m, "ReductionMode")
        .value("Sum", ReductionMode::Sum)
        .value("Max", ReductionMode::Max)
        .value("Min", ReductionMode::Min)
        .export_values();

    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def("loadPoints", &PointCloud::loadPoints,
             py::arg("pos"), py::arg("fields_in"),
             py::arg("subbox"), py::arg("vol") = py::array_t<double>(),
             py::arg("periodic") = false)
        .def("buildTree", &PointCloud::buildTree)
        .def("get_nfields", &PointCloud::get_nfields)
        .def("get_pt", &PointCloud::get_pt)
        .def("get_field", &PointCloud::get_field)
        .def("get_subbox", &PointCloud::get_subbox)
        .def("get_tree_built", &PointCloud::get_tree_built)
        .def("get_orig_ids", &PointCloud::get_orig_ids)
        .def("get_pad", &PointCloud::get_pad)
        .def("get_npart", &PointCloud::get_npart)
        .def("get_periodic", &PointCloud::get_periodic)
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
             py::arg("query_pt"), py::arg("k") = 1);

    py::class_<Projection>(m, "Projection")
        .def(py::init<
            py::array_t<Float, py::array::c_style | py::array::forcecast>,
            py::array_t<Float, py::array::c_style | py::array::forcecast>>())
        .def("makeProjection", &Projection::makeProjection, py::call_guard<py::gil_scoped_release>(),
             py::arg("cloud"), py::arg("reduction") = ReductionMode::Sum)
        .def("returnProjection", &Projection::returnProjection);

    py::class_<BruteProjection>(m, "BruteProjection")
        .def(py::init<std::array<size_t,3>, std::array<Float,6>>())
        .def("makeProjection", &BruteProjection::makeProjection, py::call_guard<py::gil_scoped_release>(),
             py::arg("cloud"), py::arg("reduction") = ReductionMode::Sum)
        .def("saveProjection", &BruteProjection::saveProjection)
        .def("returnProjection", &BruteProjection::returnProjection);

    py::class_<Slice>(m, "Slice")
        .def(py::init<std::array<size_t,2>, std::array<Float,4>, Float>())
        .def("makeSlice", &Slice::makeSlice, py::call_guard<py::gil_scoped_release>())
        .def("saveSlice", &Slice::saveSlice)
        .def("returnSlice", &Slice::returnSlice);

    py::class_<Ray>(m, "Ray")
        .def(py::init<Point, Point>())
        .def("integrate", &Ray::integrate,
             py::arg("cloud"), py::arg("mode") = ReductionMode::Sum)
        .def("get_dens_col", &Ray::get_dens_col)
        .def("get_col", &Ray::get_col)
        .def("get_max_val", &Ray::get_max_val)
        .def("get_min_val", &Ray::get_min_val)
        .def(
            "get_segments",
            [](const Ray &r){
              py::list out;
              for (auto &seg : r.get_segments()) {
                out.append(py::make_tuple(seg.cell_id, seg.s, seg.s_edge, seg.ds));
              }
              return out;
            }
          );

}