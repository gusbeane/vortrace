---
title: '`vortrace`: Fast and Exact Projections through Voronoi Meshes'
tags:
  - Python
  - astrophysics
  - simulations
authors:
  - name: Angus Beane
    orcid: 0000-0002-8658-1453
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Matthew C. Smith
    orcid: 0000-0002-9849-877X
    affiliation: 2
affiliations:
 - name: New York University, United States
   index: 1
 - name: Max-Planck-Institut für Astrophysik, Germany
   index: 2
date: 9 September 2026
bibliography: paper.bib
---

<!-- compilation command (from within paper/)
docker run --rm -it -v $PWD:/data -u $(id -u):$(id -g) openjournals/inara -o pdf,crossref ./paper.md 
-->

# Summary

It is common to use a Voronoi mesh to represent a continuous distribution, for example in astrophysical fluid simulations. This approach has several attractive properties, such as the ability for the mesh to smoothly adapt its resolution and for the mesh to move with the fluid's bulk velocity. However, Voronoi meshes are a more complicated data structure to represent and manipulate, and so integrating through them is more cumbersome. `vortrace` is a Python package for performing fast and exact integrals through Voronoi meshes. It makes the minimum number of nearest neighbor searches possible and does not perform a costly mesh construction. Its intended use case is for post-processing simulation output from codes that use Voronoi mesh representations, such as the magnetohydrodynamics code `AREPO` [@AREPO]. Other codes which use or support Voronoi mesh representations include `TESS` [@Duffell2011], `FVMHD3D` [@Gaburov2012], `SHADOWFAX` [@Vadenbroucke2016], `ChaNGa` [@Chang2017], and `CMACIONIZE` [@Vandenbroucke2018].

# Statement of need

Nearly all systems in astrophysics are observed in projection, and so efficient methods for taking projections of astrophysical simulations is essential. The ease of this projection is directly impacted by the choice for discretizing the continuous fluid distribution. For example, in a Cartesian or adaptive mesh one can geometrically select the cells that intersect a ray and determine their intersection widths. Or in smoothed particle hydrodynamics, one can efficiently select the gas particles where the ray intersects their smoothing kernel.

However, for approaches that rely on an unstructured Voronoi mesh, these projections are significantly more complicated. It is not obvious which cells intersect a ray and what their intersection widths are. If the Voronoi mesh is available, then the integral can be done exactly. But the mesh construction can be a costly operation, and navigating Voronoi mesh data structures adds algorithmic complexity. If the Voronoi mesh is not available, then one can attempt to sample points along the ray, determine the value of the field at each point, and take a sum. However, because one does not know *a priori* where the intersections are, and because a cell along the ray might be very small but nonetheless very dense, it is necessary to grossly oversample the ray to get an accurate result.

`vortrace` solves the issues with both of these prior methods by segmenting the ray at exactly the faces of each cell. It does this by using the properties of the Voronoi mesh without relying on an explicit mesh construction. This allows `vortrace` to make the fewest number of nearest neighbor searches possible, and gives the exact integral even if the ray intersects tiny cells.

# State of the field

Many tools exist for this common operation of projections through astrophysical fluids. We focus here specifically on their treatment of unstructured Voronoi meshes.

Many packages simply treat the Voronoi mesh as if it were an SPH simulation, taking the smoothing length of each particle to be a constant multiple of the cell size. This is done in, e.g., `yt` [@yt] and `gadgetviewer` [@gadgetviewer]. These codes have an advantage of being efficient, easy to use and `yt` integrates naturally into Python-based workflows. However, the computed projections are not exact and will overly smooth the integral, especially in dense regions.

We demonstrate this in \autoref{fig:yt}, showing a projection through an isolated dwarf-galaxy simulation from [@Smith2026], with a mass resolution of $2.5\,M_{\odot}$ at $t\sim800\,\text{Myr}$. The left and middle panels show the exact (`vortrace`) and SPH (`yt`) projections. Insets for an underdense region and a dense region are shown with sizes of 250\ pc and 15\ pc respectively. These insets have their own color bars to better show the difference between the two projections. It is clear from the visible edges in the insets that the `vortrace` projection more faithfully represents the underlying Voronoi mesh, while the `yt` projection smooths this small-scale structure. This is shown quantitatively in the right panel, where errors in the insets are typically on the order of $10\text{--}20\%$.

For the projection in \autoref{fig:yt}, we further show the distribution of errors for all pixels in \autoref{fig:hist}. The distribution is roughly symmetric with $10\%$ of pixels having errors greater than $\sim7\%$. However, the `yt` projection is significantly faster than the `vortrace` projection (9.3\ s vs. 282\ s single-threaded). For quick visualization purposes, the accuracy differences between the two is not so important, and so `yt` may be preferred. However, for use cases where accuracy is important, especially in the tails of the integrated distribution, then taking the exact projection would be preferred.

![Column density through a 4 kpc slab of an isolated dwarf-galaxy simulation. The mass resolution is RES and cells vary in size from 1 to 7 pc in the disk. The main image has side length of $6\,\text{kpc}$ with $1024^2$ pixels. The left panel shows the exact computation with `vortrace` and the middle panel shows the SPH-kernel calculation with `yt`. We use the default smoothing length of $h=2\,r_\mathrm{cell}$. The right panel shows the relative difference between the two. Each panel has two insets, a 250\ pc window on an underdense region (upper) and a 5\ pc window on a dense region (lower). Each inset has a separate $256^2$ grid and its own colorbar in the left and middle panels. The script that produces this figure is included in the repository (`paper/yt_comparison/make_yt_comparison.py`).\label{fig:yt}](yt_comparison/yt_comparison.pdf)

![Distribution of the per-pixel relative error of the SPH-kernel treatment, $(\mathrm{yt} - \mathrm{vortrace}) / \mathrm{vortrace}$, for the projection of \autoref{fig:yt}.\label{fig:hist}](yt_comparison/error_histogram.pdf){ width=70% }

Other packages compute the exact result by constructing the Voronoi mesh, either for the full box or in a sufficiently large cylinder around the ray. This is done in, e.g., `COLT` [@COLT] and `SKIRT` [@SKIRT9]. These codes perform full radiative transfer calculations, but for simpler integrals the extra machinery is not necessary. A similar method using the full mesh is implemented in the private version of `AREPO`. These codes do not have simple Python interfaces, and the mesh construction can be expensive, making their integration in workflows that do not require radiative transfer or the full mesh cumbersome.

# Software design

The main philosophy behind `vortrace` is to give the exact integral through a Voronoi mesh without sacrificing performance and ease of use. The algorithm itself (described in the README file) is inherently efficient and exact, and is implemented in a C++ backend. We then implement a Python wrapper around this backend using `pybind11` [@pybind11]. The package is installable via `pip`.

The general workflow of using the package is to first provide the interface with a set of mesh generating points, generating a point cloud object. This object contains methods that allow the user to provide an arbitrary set of start and end points of rays to be integrated along. Common projections (e.g., a flat $x$-$y$ grid) and plotting routines are provided, though we expect in many cases the user will want to fine tune the projections or plots to their needs or taste.

`vortrace` attempts to strike a balance between being useful in a wide variety of use cases without being overly sprawling and complicated. For example, we used the kD-tree package `nanoflann` [@blanco2014nanoflann] in lieu of implementing our own. We have also provided some minimal utilities for common projections, e.g. along coordinate axes, and simple plotting routines (`vortrace.plot`). However, we leave more complicated operations like movie making to the user.

# Minimal example

This example assumes standard `Gadget`/`AREPO` units of kpc for length and $10^{10}\,M_{\odot}$ for mass. A more thorough demonstration of the capabilities of `vortrace` is given in the documentation.

```
# assuming pos and rho have been defined elsewhere
# pos is a (N,3) numpy array
# rho is a (N,) numpy array
BoxSize = 100
center = np.array([BoxSize]*3)/2.

# define image grid. e.g., want to make a 20 kpc x 20 kpc image
L = 20
extent = [center[0]-L/2., center[0]+L/2., center[1]-L/2., center[1]+L/2.]

# integrate through the full box along the z-axis with resolution 256^2
bounds = [0, BoxSize]
npix = 256

# now we make the projection. proj_xy is a (npix, npix) numpy array
import vortrace as vt
pc = vt.ProjectionCloud(pos, rho)
proj_xy = pc.grid_projection(extent, npix, bounds)
```

# Research impact statement

`vortrace` has been used in a number of publications for generating images and videos of galaxies [@Smith21;@Smith21b;@Smith24a;@Smith24b;@Beane25a;@Beane25b;@Ortame26;@Lucchini26;@Lucchini26b;@Smith2026;@Sullivan2026]. Furthermore, the algorithm developed by @Konietzka25 to compute electron dispersion measures for fast radio bursts is closely related to vortrace, demonstrating the broad utility of this type of approach.

# AI usage disclosure

The initial version of the code did not use generative AI tools. Later development has used Anthropic's Claude Code for refactoring, fixing bugs, writing documentation, and adding some new convenience features like support for periodic boundary conditions. All commits authored by Claude Code are indicated in the commit itself, a standard testing framework is used with continuous integration through GitHub actions, and all changes were submitted as pull requests and vetted by the first author. This manuscript was written by the authors with suggestions made by Claude.

# Acknowledgements

We would like to thank Scott Lucchini for providing extensive feedback and testing `vortrace` at various stages of development. We would also like to thank Lars Hernquist, Ruediger Pakmor, and Aaron Smith for helpful discussions.

# References