// basic_projection.cpp
// Grid projection of density through an Arepo snapshot.
//
// Usage: ./basic_projection <snapshot.hdf5>

#include <vortrace/vortrace.hpp>
#include "hdf5_reader.hpp"

#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <snapshot.hdf5>\n";
        return 1;
    }

    // Load the snapshot
    auto snap = read_arepo_snap(argv[1]);
    std::cout << "Loaded " << snap.npart << " particles, BoxSize = "
              << snap.BoxSize << "\n";

    // Build the point cloud
    std::array<double, 6> subbox = {
        0, snap.BoxSize, 0, snap.BoxSize, 0, snap.BoxSize
    };
    PointCloud cloud;
    cloud.loadPoints(snap.pos.data(), snap.npart,
                     snap.rho.data(), snap.npart, 1, subbox,
                     snap.vol.data(), snap.npart);
    cloud.buildTree();

    // Set up a grid of rays along the z-axis
    double L = 75.0;
    size_t npix = 256;
    double cx = snap.BoxSize / 2, cy = snap.BoxSize / 2;
    double dx = L / npix;

    std::vector<Float> starts(3 * npix * npix);
    std::vector<Float> ends(3 * npix * npix);

    for (size_t iy = 0; iy < npix; iy++) {
        for (size_t ix = 0; ix < npix; ix++) {
            size_t idx = 3 * (iy * npix + ix);
            double x = cx - L / 2 + (ix + 0.5) * dx;
            double y = cy - L / 2 + (iy + 0.5) * dx;
            starts[idx]     = x;
            starts[idx + 1] = y;
            starts[idx + 2] = 0.0;
            ends[idx]       = x;
            ends[idx + 1]   = y;
            ends[idx + 2]   = snap.BoxSize;
        }
    }

    // Run the projection
    Projection proj(starts.data(), ends.data(), npix * npix);
    proj.makeProjection(cloud, ReductionMode::Sum);

    const auto& data = proj.getProjectionData();

    // Print some statistics
    double total = 0, maxval = 0;
    for (size_t i = 0; i < data.size(); i++) {
        total += data[i];
        if (data[i] > maxval) maxval = data[i];
    }
    std::cout << "Projection complete: " << npix << "x" << npix << " pixels\n";
    std::cout << "Total column: " << total << ", max pixel: " << maxval << "\n";

    return 0;
}
