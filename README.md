# ``vortrace``

Exact projections through voronoi meshes faster than you can say "I have a problem with your AGN model."

## What does ``vortrace`` do?

One-dimensional integrals through voronoi meshes can be expensive. Because the mesh is unstructured, it is not obvious a priori where a line interesects the faces of which cells. Brute force methods which just sample large numbers of points struggle with systems with a large dynamic range in cell size, like cosmological simulations.

``vortrace`` does one-dimensional integrals with the fewest number of nearest neighbor calls possible. With an optimized yet lightweight ``C++`` backend and a user-friendly ``python`` frontend, it's easy to get started.

## Installation

<!-- Installing ``vortrace`` can be done using ``pip``:

```
pip install vortrace
```

Or if you prefer to build from source:

```
git clone git@github.com:gusbeane/vortrace.git
cd vortrace
pip install ./
``` -->

Installing ``vortrace`` can be done using ``pip``:

```
git clone git@github.com:gusbeane/vortrace.git
cd vortrace
pip install ./
```

## Getting Started

Making your first projection takes only three lines if you've already prepared your coordinates (`pos`), density (`rho`), and projection parameters (`BoxSize`, `L`, `npix`):

```
# pos and rho defined elsewhere
BoxSize = 100
center = np.array([BoxSize]*3)/2.
L = 20
extent = [center[0]-L/2., center[0]+L/2., center[1]-L/2., center[1]+L/2.]
bounds = [0, BoxSize]
npix = 256

# now we make the projection
import vortrace as vt
pc = vt.ProjectionCloud(pos, rho)
proj_xy = pc.grid_projection(extent, npix, bounds)
```

## How Does ``vortrace`` Work?

A one-dimensional integral through an unstructured mesh at first glance seems very complicated, as one must wrangle with the complex geometry of a Voronoi mesh. However, one can use the definition of the mesh to simplify the operation considerably.

`vortrace` is a recursive algorithm that works by continually splitting the integral up until you are simply integrating between two points in neighboring cells, at which point the integral is trivial. It starts by constructing a `kDTree` using `nanoflann` to allow for efficient nearest neighbor searches. Then:

1. Assume the two points you are trying to integrate between (`p_a` and `p_b`) are in neighboring Voronoi cells. Find those cells (`v_a` and `v_b`) quickly using the `kDTree`.

2. Using these four points, find the split point `p_s` - the point on the line connecting `p_a` to `p_b` which intersects the face between cells `v_a` and `v_b`.

3. Query the `kDTree` for the cell that `p_s` is in. If it is one of `v_a` or `v_b`, go to 3a. If not, go to 3b. 

    a. You are done and can return the integral as the sum of `|p_a - p_s| * rho_a` and `|p_s - p_b| * rho_b`, where `rho_i` is the density of cell `i`.

    b. Return to 1 and start the integrals `p_a -- p_s` and `p_s -- p_b`.


## License

`vortrace` is released under the MIT License. 

This project incorporates [`nanoflann`](https://github.com/jlblancoc/nanoflann) `v1.3.2`, a header-only library distributed under the BSD License.
