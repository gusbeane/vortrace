#include "class_includes.hpp"
#include "ray.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(Cvortrace, m) {
    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def("loadPoints", &PointCloud::loadPoints)
        .def("buildTree", &PointCloud::buildTree);

    py::class_<Projection>(m, "Projection")
        .def(py::init<
            py::array_t<MyFloat, py::array::c_style | py::array::forcecast>,
            py::array_t<MyFloat, py::array::c_style | py::array::forcecast>>())
        .def("makeProjection", &Projection::makeProjection, py::call_guard<py::gil_scoped_release>())
        .def("saveProjection", &Projection::saveProjection)
        .def("returnProjection", &Projection::returnProjection);

    py::class_<BruteProjection>(m, "BruteProjection")
        .def(py::init<std::array<size_t,3>, std::array<MyFloat,6>>())
        .def("makeProjection", &BruteProjection::makeProjection, py::call_guard<py::gil_scoped_release>())
        .def("saveProjection", &BruteProjection::saveProjection);

    py::class_<Slice>(m, "Slice")
        .def(py::init<std::array<size_t,2>, std::array<MyFloat,4>, MyFloat>())
        .def("makeSlice", &Slice::makeSlice, py::call_guard<py::gil_scoped_release>())
        .def("saveSlice", &Slice::saveSlice);
    
    py::class_<Ray>(m, "Ray")
        .def(py::init<cartarr_t, cartarr_t>())  // assume constructor signature
        .def("integrate", &Ray::integrate)
        .def("get_dens_col", &Ray::get_dens_col)
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