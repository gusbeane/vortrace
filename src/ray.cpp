#include "ray.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <stdexcept>

#define TOLERANCE 1E-9
#define PARALLEL_BISECTOR_TOL 1E-12
#define MAX_ITERATIONS 100000
#define PERTURBATION_EPS 1E-10
#define MAX_PERTURBATION_CYCLES 1000

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

Float Ray::findSplitPointDistance(const PointCloud &cloud,
                                  size_t id1, size_t id2,
                                  Float s_lo, Float s_hi)
{
  Point pos1 = cloud.get_pt(id1);
  Point pos2 = cloud.get_pt(id2);
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

  // Normal case: analytic intersection of ray with bisector plane
  if (std::abs(norm_dot_dir) >= PARALLEL_BISECTOR_TOL)
    return norm_dot_ppl / norm_dot_dir;

  // Parallel bisector: the ray runs along (or very near) the bisector
  // plane of id1 and id2.  Binary search along [s_lo, s_hi] to find
  // where the cell identity changes or to locate an intermediate cell.
  std::cerr << "Warning: parallel bisector detected between cells "
            << id1 << " and " << id2 << std::endl;

  Float lo = s_lo;
  Float hi = s_hi;

  for (int bsiter = 0; bsiter < 64; bsiter++) {
    Float mid_s = 0.5 * (lo + hi);
    Point mid_pt;
    for (int j = 0; j < 3; j++)
      mid_pt[j] = pos_start[j] + mid_s * dir[j];
    size_t mid_cell = cloud.queryTree(mid_pt);

    if (mid_cell != id1 && mid_cell != id2) {
      // Found a third cell — return this position so integrate()
      // will detect it via 3-NN and insert it as intermediate.
      return mid_s;
    }

    // Narrow the interval toward the boundary
    if (mid_cell == id1) lo = mid_s;
    else hi = mid_s;

    if (hi - lo < 1e-12) {
      // Converged to the boundary between id1 and id2
      return 0.5 * (lo + hi);
    }
  }

  throw std::runtime_error("Binary search did not converge for parallel bisector");
}

size_t Ray::perturbToFindCell(const Point &pos, const PointCloud &cloud,
                               size_t exclude1, size_t exclude2) const
{
  // Base directions for generating perturbation candidates.
  // Integer linear combinations a*w1 + b*w2 + c*w3 produce a rich set
  // of directions; crossing with dir keeps them perpendicular to the ray.
  const Float INV_SQRT2 = 1.0 / std::sqrt(2.0);
  const Point w[3] = {
    {INV_SQRT2, INV_SQRT2, 0.0},
    {0.0, INV_SQRT2, INV_SQRT2},
    {INV_SQRT2, 0.0, INV_SQRT2}
  };

  int cycle = 0;
  // Expand outward through integer coefficient shells.
  // At each "bound" we try all (a,b,c) with max(|a|,|b|,|c|) == bound,
  // skipping combinations already tried at smaller bounds.
  for (int bound = 1; cycle < MAX_PERTURBATION_CYCLES; bound++) {
    for (int a = -bound; a <= bound && cycle < MAX_PERTURBATION_CYCLES; a++) {
      for (int b = -bound; b <= bound && cycle < MAX_PERTURBATION_CYCLES; b++) {
        for (int c = -bound; c <= bound && cycle < MAX_PERTURBATION_CYCLES; c++) {
          if (a == 0 && b == 0 && c == 0) continue;
          // Skip if this combination was already covered by a smaller bound
          if (std::abs(a) < bound && std::abs(b) < bound && std::abs(c) < bound)
            continue;

          Point pdir;
          for (int j = 0; j < 3; j++)
            pdir[j] = a * w[0][j] + b * w[1][j] + c * w[2][j];

          // Cross product with ray direction → perpendicular to ray
          Point cross_d;
          cross_d[0] = dir[1] * pdir[2] - dir[2] * pdir[1];
          cross_d[1] = dir[2] * pdir[0] - dir[0] * pdir[2];
          cross_d[2] = dir[0] * pdir[1] - dir[1] * pdir[0];
          Float mag = std::sqrt(cross_d[0]*cross_d[0] + cross_d[1]*cross_d[1]
                                + cross_d[2]*cross_d[2]);
          if (mag < 1e-15) { cycle++; continue; } // parallel to dir, skip
          for (int j = 0; j < 3; j++) cross_d[j] /= mag;

          Point perturbed;
          for (int j = 0; j < 3; j++)
            perturbed[j] = pos[j] + PERTURBATION_EPS * cross_d[j];

          size_t cell = cloud.queryTree(perturbed);
          if (cell != exclude1 && cell != exclude2)
            return cell;

          cycle++;
        }
      }
    }
  }
  throw std::runtime_error(
    "Perturbation did not resolve degenerate split point after "
    + std::to_string(MAX_PERTURBATION_CYCLES) + " cycles");
}

void Ray::integrate(const PointCloud &cloud, ReductionMode reduction)
{
  size_t current, next;
  size_t ctree_id, ntree_id;
  Float s, ds;
  Point pos;

  size_t nf = cloud.get_nfields();

  col.assign(nf, 0.0);
  max_val.assign(nf, -std::numeric_limits<Float>::infinity());
  min_val.assign(nf, std::numeric_limits<Float>::infinity());

  segments.clear();

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

  size_t iteration = 0;
  int not_done = 1;
  //While we haven't reached the last point
  while(not_done)
  {
    if (++iteration > MAX_ITERATIONS)
      throw std::runtime_error("Ray integration exceeded maximum iteration count");

    // Find the split point (handles parallel bisectors internally)
    s = findSplitPointDistance(cloud, ctree_id, ntree_id,
                               pts[current].s, pts[next].s);

    // Allow s at endpoints (within tolerance) — this happens when the
    // bisector passes exactly through the next (or current) RayPoint.
    if (s < pts[current].s - TOLERANCE || s > pts[next].s + TOLERANCE)
    {
      throw std::runtime_error(
        "Split point outside segment bounds: s=" + std::to_string(s) +
        " not in (" + std::to_string(pts[current].s) + ", " +
        std::to_string(pts[next].s) + ") for cells " +
        std::to_string(ctree_id) + " and " + std::to_string(ntree_id));
    }
    // Clamp to segment interior to avoid zero/negative-width segments
    s = std::clamp(s, pts[current].s + 1e-15, pts[next].s - 1e-15);

    //Convert s into a position
    for (int j = 0; j < 3; j++)
      pos[j] = pos_start[j] + s * dir[j];

    // --- Classify the split point by cell-ID membership ---
    //
    // Query 3-NN and collect the "equidistant set" (all cells whose
    // squared distance is within TOLERANCE of the nearest).  If c or n
    // appears in that set we are at the c/n boundary (Case 2).
    // Otherwise a third cell is strictly closer (Case 1).
    //
    // When neither c nor n appears in the initial 3-NN equidistant set,
    // expand to 100-NN to be sure — at a high-valence vertex c/n could
    // be pushed out of the top 3 despite being equidistant.
    size_t ids[3];
    Float r2_vals[3];
    cloud.queryTreeK(pos, 3, ids, r2_vals);

    bool has_c = false, has_n = false;
    for (int i = 0; i < 3; i++) {
      if (r2_vals[i] - r2_vals[0] >= TOLERANCE) break;
      if (ids[i] == ctree_id) has_c = true;
      if (ids[i] == ntree_id) has_n = true;
    }

    // Expand search if neither c nor n found among equidistant nearest
    if (!has_c && !has_n) {
      const size_t K_EXPAND = 100;
      size_t k_big = std::min(K_EXPAND, cloud.get_npart());
      size_t ids_big[100];
      Float r2_big[100];
      cloud.queryTreeK(pos, k_big, ids_big, r2_big);

      for (size_t i = 0; i < k_big; i++) {
        if (r2_big[i] - r2_big[0] >= TOLERANCE) break;
        if (ids_big[i] == ctree_id) has_c = true;
        if (ids_big[i] == ntree_id) has_n = true;
        if (i == k_big -1 && !has_c && !has_n) {
          throw std::runtime_error(
            "Could not resolve split point classification after expanding to "
            + std::to_string(K_EXPAND) + "-NN. This should be very unlikely; "
            "consider increasing TOLERANCE or K_EXPAND.");
        }
      }
    }

    if (has_c || has_n) {
      // Case 2/2b: c or n is among the equidistant nearest — the
      // boundary between c and n is at this split point.
      ds = s - pts[current].s;
      accumulate(ctree_id, ds);
      segments.push_back({ ctree_id, pts[current].s, s, ds });

      ds = pts[next].s - s;
      accumulate(ntree_id, ds);
      segments.push_back({ ntree_id, pts[next].s, s, ds });

      //Advance past this boundary
      current = pts[current].next;
      ctree_id = pts[current].tree_id;
      next = pts[current].next;
      if (next == SIZE_MAX) not_done = 0;
      else ntree_id = pts[next].tree_id;
    }
    else {
      // Case 1/1a: all equidistant cells are non-{c,n}.
      // A third cell lies between c and n — insert it.
      size_t insert_id;
      if (r2_vals[0] + TOLERANCE < r2_vals[1]) {
        // Case 1: single cell strictly closest.
        insert_id = ids[0];
      } else {
        // Case 1a: multiple equidistant cells.
        // Perturb perpendicular to the ray to break the tie.
        insert_id = perturbToFindCell(pos, cloud, ctree_id, ntree_id);
      }
      pts.emplace_back(insert_id, next, s);
      next = pts.size() - 1;
      pts[current].next = next;
      ntree_id = insert_id;
    }
  } //while()
}
