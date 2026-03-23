#ifndef RAY_HPP
#define RAY_HPP

#define RAY_PTS_RESERVE 1000

#include "pointcloud.hpp"
#include "mytypes.hpp"

#include <limits>

class Ray
{
  private:

    Point pos_start;
    Point pos_end;

    struct RayPoint
    {

      size_t tree_id;
      size_t next;
      Float s;

      //Constructor
      RayPoint(const size_t tree_id, const size_t next, const Float s) :
        tree_id(tree_id), next(next), s(s) {}
    };

    struct Segment {
      size_t cell_id; // id of cell which intersects ray
      Float s;      // distance from start of ray to the intersection point
      Float s_edge; // distance from start of ray to one edge
      Float ds;     // intersection width
    };

    std::vector<Segment> segments;
    std::vector<RayPoint> pts;

    std::vector<Float> col;      // accumulated column values (Sum mode)
    std::vector<Float> max_val;  // max values along ray
    std::vector<Float> min_val;  // min values along ray

  public:
    Point dir;

    Ray(const Point &start, const Point &end);

    // Analytic bisector-ray intersection distance.  Returns Float::max()
    // sentinel when the bisector is parallel to the ray.
    Float findSplitPointDistance(const Point &pos1, const Point &pos2);

    // Full version: tries the analytic intersection first, then falls back
    // to a binary search along [s_lo, s_hi] when the bisector is parallel.
    Float findSplitPointDistance(const PointCloud &cloud,
                                 size_t id1, size_t id2,
                                 Float s_lo, Float s_hi);

    // Perturb the split point in cycling directions (cross(dir, wN))
    // to find a cell that is neither exclude1 nor exclude2.
    size_t perturbToFindCell(const Point &pos, const PointCloud &cloud,
                             size_t exclude1, size_t exclude2) const;

    void integrate(const PointCloud &cloud, ReductionMode mode = ReductionMode::Sum);

    //Getters
    Float get_dens_col() const {return col.empty() ? 0.0 : col[0];}
    const std::vector<Float>& get_col() const {return col;}
    const std::vector<Float>& get_max_val() const {return max_val;}
    const std::vector<Float>& get_min_val() const {return min_val;}
    const std::vector<Segment>& get_segments() const {return segments;}

};


#endif