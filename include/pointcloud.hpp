#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

#include "mytypes.hpp"
#include <memory>
#include <vector>
#include <string>
#include <nanoflann.hpp>

//Forward declare for subsequent typedef
class PointCloud;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Float, PointCloud>,PointCloud,3> my_kd_tree_t;

class PointCloud
{
  private:
    size_t npart = 0;
    size_t nfields = 1;
    double pad = 0.0;
    std::array<Float,6> subbox;
    std::vector<Point> pts;
    std::vector<Float> fields;  // flat, particle-major: fields[idx * nfields + f]
    std::vector<size_t> orig_ids;  // maps filtered index -> original index

    bool periodic = false;
    Point box_size = {0.0, 0.0, 0.0};
    bool valid_rgba = false;

    bool tree_built = false;
    std::unique_ptr<my_kd_tree_t> tree;

    Float minDistSqToBox(const Point &query_pt) const;

  public:

    //Load all particles, filter to padded subbox, build internal arrays.
    //fields is flat, row-major: fields[i * nfields_in + f].
    //vol is optional per-cell volumes (length nvol) for adaptive padding.
    void loadPoints(const double* pos, size_t npart,
                    const double* fields, size_t npart_fields, size_t nfields_in,
                    const std::array<Float,6>& subbox,
                    const double* vol = nullptr, size_t nvol = 0,
                    bool periodic = false);
    void buildTree();
    void saveTree(const std::string& filename) const;
    void loadTree(const std::string& filename);
    std::vector<char> saveTreeToBuffer() const;
    void loadTreeFromBuffer(const char* data, size_t size);

    size_t queryTree(const Point &query_pt) const;
    void queryTreeK(const Point &query_pt, size_t k, size_t *results, Float *r2) const;

    //Getters, we need some read-only access to data
    std::array<Float,6> get_subbox() const {return subbox;}
    bool get_tree_built() const {return tree_built;}
    Point get_pt(const size_t idx) const {return pts[idx];}
    Float get_field(const size_t idx, const size_t f) const {return fields[idx * nfields + f];}
    size_t get_nfields() const {return nfields;}
    const std::vector<size_t>& get_orig_ids() const {return orig_ids;}
    double get_pad() const {return pad;}
    size_t get_npart() const {return npart;}
    bool get_periodic() const {return periodic;}
    bool get_valid_rgba() const {return valid_rgba;}
    Point get_box_size() const {return box_size;}

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