// periodic.cpp
// Projection of a cosmological box with periodic boundary conditions.
//
// Usage: ./periodic <cosmo_box.hdf5>

#include <vortrace/vortrace.hpp>
#include "hdf5_reader.hpp"

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cosmo_box.hdf5>\n";
        return 1;
    }

    auto snap = read_arepo_snap(argv[1]);
    std::cout << "Loaded " << snap.npart << " particles, BoxSize = "
              << snap.BoxSize << "\n";

    // Load with periodic boundary conditions
    std::array<double, 6> subbox = {
        0, snap.BoxSize, 0, snap.BoxSize, 0, snap.BoxSize
    };
    PointCloud cloud;
    cloud.loadPoints(snap.pos.data(), snap.npart,
                     snap.rho.data(), snap.npart, 1, subbox,
                     snap.vol.data(), snap.npart,
                     true);  // periodic = true
    cloud.buildTree();

    // Full-box projection
    size_t npix = 128;
    double dx = snap.BoxSize / npix;

    std::vector<Float> starts(3 * npix * npix);
    std::vector<Float> ends(3 * npix * npix);

    for (size_t iy = 0; iy < npix; iy++) {
        for (size_t ix = 0; ix < npix; ix++) {
            size_t idx = 3 * (iy * npix + ix);
            double x = (ix + 0.5) * dx;
            double y = (iy + 0.5) * dx;
            starts[idx] = x;  starts[idx+1] = y;  starts[idx+2] = 0.0;
            ends[idx]   = x;  ends[idx+1]   = y;  ends[idx+2]   = snap.BoxSize;
        }
    }

    Projection proj(starts.data(), ends.data(), npix * npix);
    proj.makeProjection(cloud, ReductionMode::Sum);

    const auto& data = proj.getProjectionData();

    double total = 0, maxval = 0;
    for (size_t i = 0; i < data.size(); i++) {
        total += data[i];
        if (data[i] > maxval) maxval = data[i];
    }
    std::cout << "Periodic projection complete: " << npix << "x" << npix
              << " pixels\n";
    std::cout << "Total column: " << total << ", max pixel: " << maxval << "\n";

    return 0;
}
