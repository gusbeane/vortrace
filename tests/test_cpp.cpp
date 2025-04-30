#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "ray.hpp"
#include "pointcloud.hpp"

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
  // mid‐plane is at z = 2 ⇒ s = 2
  REQUIRE( Catch::Approx(r.findSplitPointDistance(p1,p2)) == 2.0 );
}

TEST_CASE("findSplitPointDistance diagonal ray", "[Ray]") {
  Point a = {0,0,0}, b = {1,1,1};
  Ray r(a,b);
  Point p1 = {0.25,0.25,0.25}, p2 = {0.75,0.75,0.75};
  // mid‐point at (0.5,0.5,0.5) ⇒ s = |(0.5,0.5,0.5)| = 0.5 * √3
  Float expected = 0.5 * std::sqrt(3.0);
  REQUIRE( Catch::Approx(r.findSplitPointDistance(p1,p2)) == expected );
}