// single_ray.cpp
// Trace a single ray through the mesh and inspect each segment.
//
// Usage: ./single_ray <snapshot.hdf5>

#include <vortrace/vortrace.hpp>
#include <vortrace/ray.hpp>
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

    // Cast a single ray through the centre of the box
    double cx = snap.BoxSize / 2;
    Point start = {cx + 3.0, cx + 10.5, 0.0};
    Point end   = {cx + 3.0, cx + 10.5, snap.BoxSize};

    Ray ray(start, end);
    ray.walk(cloud);

    const auto& segments = ray.get_segments();
    std::cout << "Ray crossed " << segments.size() << " cells\n\n";

    // Print the first 10 segments
    size_t nprint = std::min(segments.size(), size_t(10));
    std::cout << "  cell_id   s_enter     s_exit       ds       rho\n";
    std::cout << "  -------   -------     ------       --       ---\n";
    for (size_t i = 0; i < nprint; i++) {
        const auto& seg = segments[i];
        double rho_val = cloud.get_field(seg.cell_id, 0);
        std::cout << "  " << seg.cell_id
                  << "\t" << seg.s_enter
                  << "\t" << seg.s_exit
                  << "\t" << seg.ds()
                  << "\t" << rho_val << "\n";
    }
    if (segments.size() > nprint)
        std::cout << "  ... (" << segments.size() - nprint << " more)\n";

    // Also compute the integrated column density
    ray.integrate(cloud, ReductionMode::Sum);
    std::cout << "\nIntegrated column density: " << ray.get_col()[0] << "\n";

    return 0;
}
