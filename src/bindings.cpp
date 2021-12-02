#include "class_includes.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(Cvortrace, m) {
    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def("loadPoints", &PointCloud::loadPoints)
        .def("buildTree", &PointCloud::buildTree);

    py::class_<Projection>(m, "Projection")
        .def(py::init<std::array<size_t,2>, std::array<MyFloat,6>>())
        .def("makeProjection", &Projection::makeProjection, py::call_guard<py::gil_scoped_release>())
        .def("saveProjection", &Projection::saveProjection);

    py::class_<BruteProjection>(m, "BruteProjection")
        .def(py::init<std::array<size_t,3>, std::array<MyFloat,6>>())
        .def("makeProjection", &BruteProjection::makeProjection, py::call_guard<py::gil_scoped_release>())
        .def("saveProjection", &BruteProjection::saveProjection);

    py::class_<Slice>(m, "Slice")
        .def(py::init<std::array<size_t,2>, std::array<MyFloat,4>, MyFloat>())
        .def("makeSlice", &Slice::makeSlice, py::call_guard<py::gil_scoped_release>())
        .def("saveSlice", &Slice::saveSlice);

}