#ifndef RAY_HPP
#define RAY_HPP

#define RAY_PTS_RESERVE 200

#include "pointcloud.hpp"
#include "mytypes.hpp"

class Ray
{
  private:

    cartarr_t pos_start;
    cartarr_t pos_end;
    cartarr_t dir;

    struct RayPoint
    {

      size_t tree_id;
      size_t next;
      MyFloat s;

      //Constructor
      RayPoint(const size_t tree_id, const size_t next, const MyFloat s) :
        tree_id(tree_id), next(next), s(s) {}
    };

    struct Segment {
      size_t cell_id; // id of cell which intersects ray
      MyFloat s;      // distance from start of ray to the intersection point
      MyFloat ds;     // intersection width
    };

    std::vector<Segment> segments;

    std::vector<RayPoint> pts;
    

    MyFloat dens_col = 0;

  public:

    Ray(const cartarr_t &start, const cartarr_t &end);

    //Find the distance (from pos_start) of the point splitting points idx1 and idx2
    MyFloat findSplitPointDistance(const cartarr_t &pos1, const cartarr_t &pos2);

    void integrate(const PointCloud &cloud);

    //Getters
    MyFloat get_dens_col() const {return dens_col;}
    const std::vector<Segment>& get_segments() const {return segments;}

};


#endif