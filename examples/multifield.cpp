// multifield.cpp
// Multi-field projection: density and density-weighted temperature.
//
// Usage: ./multifield <snapshot.hdf5>

#include <vortrace/vortrace.hpp>
#include "hdf5_reader.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <hdf5.h>

// Read temperature from the snapshot (simplified calculation)
static std::vector<double> read_temperature(const std::string& filename, size_t npart) {
    std::vector<double> temperature(npart);

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    std::vector<double> u(npart), xe(npart), rho(npart);
    {
        hid_t dset = H5Dopen2(file, "PartType0/InternalEnergy", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u.data());
        H5Dclose(dset);
    }
    {
        hid_t dset = H5Dopen2(file, "PartType0/ElectronAbundance", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, xe.data());
        H5Dclose(dset);
    }

    // Read unit system
    double UnitMass, UnitLength, UnitVelocity;
    {
        hid_t grp = H5Gopen2(file, "Parameters", H5P_DEFAULT);
        hid_t attr;
        attr = H5Aopen(grp, "UnitMass_in_g", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &UnitMass);
        H5Aclose(attr);
        attr = H5Aopen(grp, "UnitLength_in_cm", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &UnitLength);
        H5Aclose(attr);
        attr = H5Aopen(grp, "UnitVelocity_in_cm_per_s", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &UnitVelocity);
        H5Aclose(attr);
        H5Gclose(grp);
    }
    H5Fclose(file);

    const double HYDROGEN_MASSFRAC = 0.76;
    const double GAMMA = 5.0 / 3.0;
    const double PROTONMASS = 1.67262178e-24;
    const double BOLTZMANN = 1.38065e-16;
    const double UnitEnergy = UnitMass * UnitVelocity * UnitVelocity;

    for (size_t i = 0; i < npart; i++) {
        double mu = 4 * PROTONMASS / (1 + 3 * HYDROGEN_MASSFRAC
                                      + 4 * HYDROGEN_MASSFRAC * xe[i]);
        temperature[i] = (GAMMA - 1.0) * (u[i] / BOLTZMANN)
                         * (UnitEnergy / UnitMass) * mu;
    }

    return temperature;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <snapshot.hdf5>\n";
        return 1;
    }

    std::string filename = argv[1];
    auto snap = read_arepo_snap(filename);
    auto temperature = read_temperature(filename, snap.npart);
    std::cout << "Loaded " << snap.npart << " particles\n";

    // Build a 2-field array: [rho, rho * T]
    size_t nfields = 2;
    std::vector<double> fields(snap.npart * nfields);
    for (size_t i = 0; i < snap.npart; i++) {
        fields[i * 2 + 0] = snap.rho[i];
        fields[i * 2 + 1] = snap.rho[i] * temperature[i];
    }

    std::array<double, 6> subbox = {
        0, snap.BoxSize, 0, snap.BoxSize, 0, snap.BoxSize
    };
    PointCloud cloud;
    cloud.loadPoints(snap.pos.data(), snap.npart,
                     fields.data(), snap.npart, nfields, subbox,
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

    Projection proj(starts.data(), ends.data(), npix * npix);
    proj.makeProjection(cloud, ReductionMode::Sum);

    const auto& data = proj.getProjectionData();

    // Compute mass-weighted temperature for the central pixel
    size_t center = (npix / 2) * npix + npix / 2;
    double col_density = data[center * 2 + 0];
    double rho_T_integral = data[center * 2 + 1];
    double T_weighted = rho_T_integral / col_density;

    std::cout << "Central pixel: column density = " << col_density
              << ", mass-weighted T = " << T_weighted << "\n";

    return 0;
}
