#include "class_includes.hpp"
#include "ray.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(Cvortrace, m) {
    py::enum_<ReductionMode>(m, "ReductionMode")
        .value("Sum", ReductionMode::Sum)
        .value("Max", ReductionMode::Max)
        .value("Min", ReductionMode::Min)
        .export_values();

    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def("loadPoints", &PointCloud::loadPoints)
        .def("buildTree", &PointCloud::buildTree)
        .def("get_nfields", &PointCloud::get_nfields);

    py::class_<Projection>(m, "Projection")
        .def(py::init<
            py::array_t<Float, py::array::c_style | py::array::forcecast>,
            py::array_t<Float, py::array::c_style | py::array::forcecast>>())
        .def("makeProjection", &Projection::makeProjection, py::call_guard<py::gil_scoped_release>(),
             py::arg("cloud"), py::arg("reduction") = 0)
        .def("returnProjection", &Projection::returnProjection);

    py::class_<BruteProjection>(m, "BruteProjection")
        .def(py::init<std::array<size_t,3>, std::array<Float,6>>())
        .def("makeProjection", &BruteProjection::makeProjection, py::call_guard<py::gil_scoped_release>(),
             py::arg("cloud"), py::arg("reduction") = 0)
        .def("saveProjection", &BruteProjection::saveProjection);

    py::class_<Slice>(m, "Slice")
        .def(py::init<std::array<size_t,2>, std::array<Float,4>, Float>())
        .def("makeSlice", &Slice::makeSlice, py::call_guard<py::gil_scoped_release>())
        .def("saveSlice", &Slice::saveSlice);

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