// slicing.cpp
// Extract a 2D slice of density at the midplane of the box.
//
// Usage: ./slicing <snapshot.hdf5>

#include <vortrace/vortrace.hpp>
#include "hdf5_reader.hpp"

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <snapshot.hdf5>\n";
        return 1;
    }

    auto snap = read_arepo_snap(argv[1]);
    std::cout << "Loaded " << snap.npart << " particles\n";

    std::array<double, 6> subbox = {
        0, snap.BoxSize, 0, snap.BoxSize, 0, snap.BoxSize
    };
    PointCloud cloud;
    cloud.loadPoints(snap.pos.data(), snap.npart,
                     snap.rho.data(), snap.npart, 1, subbox,
                     snap.vol.data(), snap.npart);
    cloud.buildTree();

    // Slice at the midplane
    std::array<size_t, 2> npix = {128, 128};
    double margin = 0.1;
    std::array<Float, 4> extent = {
        margin, snap.BoxSize - margin,
        margin, snap.BoxSize - margin
    };
    Float depth = snap.BoxSize / 2;

    Slice slice(npix, extent, depth);
    slice.makeSlice(cloud);

    const auto& data = slice.getSliceData();
    std::cout << "Slice complete: " << npix[0] << "x" << npix[1]
              << " pixels, " << data.size() << " values\n";

    // Print statistics
    double minval = data[0], maxval = data[0], total = 0;
    for (double v : data) {
        if (v < minval) minval = v;
        if (v > maxval) maxval = v;
        total += v;
    }
    std::cout << "Min: " << minval << ", Max: " << maxval
              << ", Mean: " << total / data.size() << "\n";

    return 0;
}
