#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "ray.hpp"
#include "pointcloud.hpp"
#include "projection.hpp"
#include "brute_projection.hpp"
#include "slice.hpp"

// ---- Ray tests ----

TEST_CASE("Ray ctor normalizes direction", "[Ray]") {
  Point a = {0,0,0}, b = {3,4,0};
  Ray r(a,b);
  auto &d = r.dir;                      // note: dir must be public for this test
  Float len = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
  REQUIRE( Catch::Approx(len) == 1.0 );
}

TEST_CASE("findSplitPointDistance simple", "[Ray]") {
  Point a={0,0,0}, b={0,0,1};
  Ray r(a,b);
  Point p1={0,0,0}, p2={0,0,1};
  REQUIRE( Catch::Approx(r.findSplitPointDistance(p1,p2)) == 0.5 );
}

TEST_CASE("findSplitPointDistance non-axis aligned", "[Ray]") {
  Point a = {0,0,0}, b = {0,0,2};
  Ray r(a,b);
  Point p1 = {0,0,1}, p2 = {0,0,3};
  // mid-plane is at z = 2 => s = 2
  REQUIRE( Catch::Approx(r.findSplitPointDistance(p1,p2)) == 2.0 );
}

TEST_CASE("findSplitPointDistance diagonal ray", "[Ray]") {
  Point a = {0,0,0}, b = {1,1,1};
  Ray r(a,b);
  Point p1 = {0.25,0.25,0.25}, p2 = {0.75,0.75,0.75};
  // mid-point at (0.5,0.5,0.5) => s = |(0.5,0.5,0.5)| = 0.5 * sqrt(3)
  Float expected = 0.5 * std::sqrt(3.0);
  REQUIRE( Catch::Approx(r.findSplitPointDistance(p1,p2)) == expected );
}

// ---- PointCloud tests ----

TEST_CASE("PointCloud loadPoints and queryTree", "[PointCloud]") {
  // 4 points at corners of a square in z=0.5 plane
  double pos[] = {
    0.25, 0.25, 0.5,
    0.75, 0.25, 0.5,
    0.25, 0.75, 0.5,
    0.75, 0.75, 0.5
  };
  double fields[] = {1.0, 2.0, 3.0, 4.0};
  std::array<Float,6> subbox = {0, 1, 0, 1, 0, 1};

  PointCloud cloud;
  cloud.loadPoints(pos, 4, fields, 4, 1, subbox);
  REQUIRE(cloud.get_npart() == 4);
  REQUIRE(cloud.get_nfields() == 1);

  cloud.buildTree();
  REQUIRE(cloud.get_tree_built());

  // Query near (0.25, 0.25, 0.5) should find particle 0
  Point q = {0.3, 0.3, 0.5};
  size_t nearest = cloud.queryTree(q);
  REQUIRE(nearest == 0);

  // Query near (0.75, 0.75, 0.5) should find particle 3
  Point q2 = {0.7, 0.7, 0.5};
  nearest = cloud.queryTree(q2);
  REQUIRE(nearest == 3);
}

TEST_CASE("PointCloud multi-field", "[PointCloud]") {
  // 2 points, 3 fields each
  double pos[] = {0.25, 0.5, 0.5, 0.75, 0.5, 0.5};
  double fields[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::array<Float,6> subbox = {0, 1, 0, 1, 0, 1};

  PointCloud cloud;
  cloud.loadPoints(pos, 2, fields, 2, 3, subbox);
  REQUIRE(cloud.get_nfields() == 3);

  // Check field values
  REQUIRE(cloud.get_field(0, 0) == 1.0);
  REQUIRE(cloud.get_field(0, 2) == 3.0);
  REQUIRE(cloud.get_field(1, 0) == 4.0);
  REQUIRE(cloud.get_field(1, 2) == 6.0);
}

TEST_CASE("PointCloud filters particles to padded subbox", "[PointCloud]") {
  // 3 particles, one well outside the subbox
  double pos[] = {
    0.5, 0.5, 0.5,
    0.9, 0.9, 0.9,
    10.0, 10.0, 10.0  // far away
  };
  double fields[] = {1.0, 2.0, 3.0};
  std::array<Float,6> subbox = {0, 1, 0, 1, 0, 1};

  PointCloud cloud;
  cloud.loadPoints(pos, 3, fields, 3, 1, subbox);
  // The far-away particle should be filtered out
  REQUIRE(cloud.get_npart() == 2);
}

// ---- Projection tests ----

TEST_CASE("Projection basic integration", "[Projection]") {
  // Create a uniform point cloud
  double pos[] = {0.5, 0.5, 0.5};
  double fields[] = {1.0};
  std::array<Float,6> subbox = {0, 1, 0, 1, 0, 1};

  PointCloud cloud;
  cloud.loadPoints(pos, 1, fields, 1, 1, subbox);
  cloud.buildTree();

  // Single ray from (0.5, 0.5, 0.0) to (0.5, 0.5, 1.0)
  Float starts[] = {0.5, 0.5, 0.0};
  Float ends[]   = {0.5, 0.5, 1.0};

  Projection proj(starts, ends, 1);
  proj.makeProjection(cloud, ReductionMode::Sum);

  REQUIRE(proj.getNgrid() == 1);
  REQUIRE(proj.getNfields() == 1);

  const auto& data = proj.getProjectionData();
  REQUIRE(data.size() == 1);
  // With a single particle and field=1.0, the integrated column should be ~1.0
  REQUIRE(data[0] == Catch::Approx(1.0).epsilon(0.01));
}

// ---- BruteProjection tests ----

TEST_CASE("BruteProjection basic", "[BruteProjection]") {
  double pos[] = {0.5, 0.5, 0.5};
  double fields[] = {2.0};
  std::array<Float,6> subbox = {0, 1, 0, 1, 0, 1};

  PointCloud cloud;
  cloud.loadPoints(pos, 1, fields, 1, 1, subbox);
  cloud.buildTree();

  std::array<size_t,3> npix = {2, 2, 2};
  std::array<Float,6> extent = {0.1, 0.9, 0.1, 0.9, 0.1, 0.9};

  BruteProjection bp(npix, extent);
  bp.makeProjection(cloud, ReductionMode::Sum);

  REQUIRE(bp.getNfields() == 1);
  const auto& data = bp.getProjectionData();
  REQUIRE(data.size() == 4);  // npix_x * npix_y = 2*2
  // With uniform field=2.0 and Sum mode, all pixels should have the same value
  REQUIRE(data[0] == Catch::Approx(data[1]));
}

// ---- Slice tests ----

TEST_CASE("Slice basic", "[Slice]") {
  double pos[] = {0.5, 0.5, 0.5};
  double fields[] = {3.0};
  std::array<Float,6> subbox = {0, 1, 0, 1, 0, 1};

  PointCloud cloud;
  cloud.loadPoints(pos, 1, fields, 1, 1, subbox);
  cloud.buildTree();

  std::array<size_t,2> npix = {3, 3};
  std::array<Float,4> extent = {0.1, 0.9, 0.1, 0.9};
  Float depth = 0.5;

  Slice slice(npix, extent, depth);
  slice.makeSlice(cloud);

  REQUIRE(slice.getNfields() == 1);
  const auto& data = slice.getSliceData();
  REQUIRE(data.size() == 9);  // 3*3
  // With a single particle with field=3.0, all slice values should be 3.0
  for (size_t i = 0; i < data.size(); i++) {
    REQUIRE(data[i] == Catch::Approx(3.0));
  }
}
