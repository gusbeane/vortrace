#include "ray.hpp"
#include <iostream>

Ray::Ray(const Point &start, const Point &end)
{
  pos_start = start;
  pos_end = end;
  
  dir[0] = end[0] - start[0];
  dir[1] = end[1] - start[1];
  dir[2] = end[2] - start[2];

  Float s = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  dir[0] /= s;
  dir[1] /= s;
  dir[2] /= s;

  pts.reserve(RAY_PTS_RESERVE);

  //Push start and end points onto the ray
  //Temporarily use SIZE_MAX for tree_id, to be replaced in integrate() when tree is loaded.
  pts.emplace_back(SIZE_MAX, 1, 0.0);
  pts.emplace_back(SIZE_MAX, SIZE_MAX, s);

}

//Find the distance (from pos_start) of the point splitting points pos1 and pos2
Float Ray::findSplitPointDistance(const Point &pos1, const Point &pos2)
{
  Point ppl, norm;

  //The (unnormalised) plane normal
  norm[0] = pos2[0] - pos1[0];
  norm[1] = pos2[1] - pos1[1];
  norm[2] = pos2[2] - pos1[2];

  //A point on the plane, relative to pos_start
  ppl[0] = 0.5 * (pos1[0] + pos2[0]) - pos_start[0];
  ppl[1] = 0.5 * (pos1[1] + pos2[1]) - pos_start[1];
  ppl[2] = 0.5 * (pos1[2] + pos2[2]) - pos_start[2];

  Float norm_dot_ppl = norm[0] * ppl[0] + norm[1] * ppl[1] + norm[2] * ppl[2];
  Float norm_dot_dir = norm[0] * dir[0] + norm[1] * dir[1] + norm[2] * dir[2];

  return norm_dot_ppl / norm_dot_dir;

}

void Ray::integrate(const PointCloud &cloud, ReductionMode reduction)
{
  size_t current, next;
  size_t ctree_id, ntree_id, stree_id;
  Float s, ds;
  Point pos;
  int mode;

  size_t nf = cloud.get_nfields();

  col.assign(nf, 0.0);
  max_val.assign(nf, -std::numeric_limits<Float>::infinity());
  min_val.assign(nf, std::numeric_limits<Float>::infinity());

  segments.clear();
  // dynamic doubling: initial reserve done in constructor, further doubling handled automatically

  // Helper lambda to accumulate a segment for all fields
  auto accumulate = [&](size_t cell_id, Float seg_ds) {
    for (size_t f = 0; f < nf; f++) {
      Float val = cloud.get_field(cell_id, f);
      switch (reduction) {
        case ReductionMode::Sum:
          col[f] += seg_ds * val;
          break;
        case ReductionMode::Max:
          if (val > max_val[f]) max_val[f] = val;
          break;
        case ReductionMode::Min:
          if (val < min_val[f]) min_val[f] = val;
          break;
      }
    }
  };

  //Find nearest neighbour for start and end ray points

  pts[0].tree_id = cloud.queryTree(pos_start);
  pts[1].tree_id = cloud.queryTree(pos_end);

  //If tree_id matches, in same cell already so integrate and stop
  if(pts[0].tree_id == pts[1].tree_id)
    {
      throw std::runtime_error(
        "Start and end point are in the same cell. "
        "Start point tree_id: " + std::to_string(pts[0].tree_id) +
        ", End point tree_id: " + std::to_string(pts[1].tree_id)
      );
    }

  //Otherwise start integration
  current = 0;
  ctree_id = pts[current].tree_id;
  next = 1;
  ntree_id = pts[next].tree_id;

  int not_done = 1;
  //While we haven't reached the last point
  while(not_done)
  {
    //Look for the split point
    //First find the distance from r_start of the split point along dir
    s = findSplitPointDistance(cloud.get_pt(ctree_id),cloud.get_pt(ntree_id));
    //Convert into a position
    pos[0] = pos_start[0] + s * dir[0];
    pos[1] = pos_start[1] + s * dir[1];
    pos[2] = pos_start[2] + s * dir[2];

    //Neighbour search for this position
    stree_id = cloud.checkMode(pos, ctree_id, ntree_id, &mode);

    // mode has the following values:
    //   0: if we are on an edge between ctree_id and ntree_id
    //   1: if we are on an edge between ctree_id and another cell(s)
    //   2: if we are on an edge between ntree_id and another cell(s)
    //   3: if we are not on an edge between either ctree_id and ntree_id

    switch(mode) {
      case 0:
        // pts[*].s gives the position along the ray to the mesh generating point
        // s gives the position along the ray to the edge between cells
        ds = s - pts[current].s;
        accumulate(ctree_id, ds);
        segments.push_back({ ctree_id, pts[current].s, s, ds });

        ds = pts[next].s - s;
        accumulate(ntree_id, ds);
        segments.push_back({ ntree_id, pts[next].s, s, ds });

        //Move on
        current = pts[current].next;
        ctree_id = pts[current].tree_id;
        next = pts[current].next;

        //Check to see if we reached the end
        if(next == SIZE_MAX)
          not_done = 0;
        ntree_id = pts[next].tree_id;
        break;
      case 1:
        throw std::runtime_error("Ray integration encountered degenerate mode=1 at split point");
      case 2:
        throw std::runtime_error("Ray integration encountered degenerate mode=2 at split point");
      case 3:
        pts.emplace_back(stree_id, next, s);
        next = pts.size() - 1;
        pts[current].next = next;
        ntree_id = stree_id;
    }
  } //while()
}




