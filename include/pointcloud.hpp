#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

#include "mytypes.hpp"
#include <nanoflann.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//Forward declare for subsequent typedef
class PointCloud;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Float, PointCloud>,PointCloud,3> my_kd_tree_t;

class PointCloud
{
  private:
    size_t npart = 0;
    std::array<Float,6> subbox;
    std::vector<Point> pts;
    std::vector<Float> dens;

    bool tree_built = false;
    std::unique_ptr<my_kd_tree_t> tree;

  public:

    //Load gas from snapshot, applying subbox {xmin,xmax,ymin,ymax,zmin,zmax}
    //and build tree
    void loadPoints(py::array_t<double> pos, py::array_t<double> dens, const std::array<Float,6> newsubbox);
    void buildTree();

    size_t queryTree(const Point &query_pt) const;
    size_t checkMode(const Point &query_pt, size_t ctree_id, size_t ntree_id, int *mode) const;

    //Getters, we need some read-only access to data
    std::array<Float,6> get_subbox() const {return subbox;}
    bool get_tree_built() const {return tree_built;}
    Point get_pt(const size_t idx) const {return pts[idx];}
    Float get_dens(const size_t idx) const {return dens[idx];}

    //Required methods for nanoflann adaptor.
    inline size_t kdtree_get_point_count() const { return npart;}

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline Float kdtree_get_pt(const size_t idx, const size_t dim) const {return pts[idx][dim];}

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

#endif //POINT_CLOUD_HPP