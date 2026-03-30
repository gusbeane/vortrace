// volume_rendering.cpp
// Volume rendering of density with a simple transfer function.
//
// Usage: ./volume_rendering <snapshot.hdf5>

#include <vortrace/vortrace.hpp>
#include "hdf5_reader.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Simple transfer function: map log-density to RGBA.
// Uses a linear ramp for colour and a Gaussian for opacity.
static void apply_transfer(const std::vector<double>& rho, size_t n,
                           std::vector<double>& fields) {
    fields.resize(n * 4);

    std::vector<double> logq(n);
    for (size_t i = 0; i < n; i++)
        logq[i] = std::log10(rho[i]);

    // Find range (approximate percentiles with min/max)
    auto sorted = logq;
    std::sort(sorted.begin(), sorted.end());
    double logq_min = sorted[n / 100 + 1];
    double logq_max = sorted[n - n / 100 - 1];
    double range = logq_max - logq_min;

    for (size_t i = 0; i < n; i++) {
        double norm = (logq[i] - logq_min) / range;
        norm = std::max(0.0, std::min(1.0, norm));

        // Simple RGB: blue (low) -> red (high)
        fields[i * 4 + 0] = norm;           // R
        fields[i * 4 + 1] = 0.2;            // G
        fields[i * 4 + 2] = 1.0 - norm;     // B

        // Gaussian opacity centred at the median
        double mid = (logq_min + logq_max) / 2;
        double sigma = range * 0.25;
        double x = (logq[i] - mid) / sigma;
        fields[i * 4 + 3] = 0.01 * std::exp(-0.5 * x * x);  // alpha
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <snapshot.hdf5>\n";
        return 1;
    }

    auto snap = read_arepo_snap(argv[1]);
    std::cout << "Loaded " << snap.npart << " particles\n";

    // Apply transfer function to get RGBA fields
    std::vector<double> fields;
    apply_transfer(snap.rho, snap.npart, fields);

    std::array<double, 6> subbox = {
        0, snap.BoxSize, 0, snap.BoxSize, 0, snap.BoxSize
    };
    PointCloud cloud;
    cloud.loadPoints(snap.pos.data(), snap.npart,
                     fields.data(), snap.npart, 4, subbox,
                     snap.vol.data(), snap.npart);
    cloud.buildTree();

    // Set up rays
    double L = 75.0;
    size_t npix = 128;
    double cx = snap.BoxSize / 2, cy = snap.BoxSize / 2;
    double dx = L / npix;

    std::vector<Float> starts(3 * npix * npix);
    std::vector<Float> ends(3 * npix * npix);

    for (size_t iy = 0; iy < npix; iy++) {
        for (size_t ix = 0; ix < npix; ix++) {
            size_t idx = 3 * (iy * npix + ix);
            double x = cx - L / 2 + (ix + 0.5) * dx;
            double y = cy - L / 2 + (iy + 0.5) * dx;
            starts[idx] = x;  starts[idx+1] = y;  starts[idx+2] = 0.0;
            ends[idx]   = x;  ends[idx+1]   = y;  ends[idx+2]   = snap.BoxSize;
        }
    }

    // Volume render
    Projection proj(starts.data(), ends.data(), npix * npix);
    proj.makeProjection(cloud, ReductionMode::VolumeRender);

    const auto& data = proj.getProjectionData();
    // data has 3 values per ray (RGB)
    std::cout << "Volume render complete: " << npix << "x" << npix
              << " pixels, " << data.size() << " output values\n";

    // Print RGB of the central pixel
    size_t center = (npix / 2) * npix + npix / 2;
    std::cout << "Central pixel RGB: ("
              << data[center * 3 + 0] << ", "
              << data[center * 3 + 1] << ", "
              << data[center * 3 + 2] << ")\n";

    return 0;
}
