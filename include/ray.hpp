#ifndef RAY_HPP
#define RAY_HPP

#define RAY_PTS_RESERVE 1000

#include "pointcloud.hpp"
#include "mytypes.hpp"

class Ray
{
  private:

    Point pos_start;
    Point pos_end;
    Point dir;

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

    Float dens_col = 0;

  public:

    Ray(const Point &start, const Point &end);

    //Find the distance (from pos_start) of the point splitting points idx1 and idx2
    Float findSplitPointDistance(const Point &pos1, const Point &pos2);

    void integrate(const PointCloud &cloud);

    //Getters
    Float get_dens_col() const {return dens_col;}
    const std::vector<Segment>& get_segments() const {return segments;}

};


#endif