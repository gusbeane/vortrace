#ifndef HDF5_READER_HPP
#define HDF5_READER_HPP

// Minimal HDF5 reader for Arepo snapshots used by the vortrace examples.
// Reads PartType0/Coordinates, Density, Masses, and Parameters/BoxSize.

#include <hdf5.h>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

struct ArepoSnapshot {
    std::vector<double> pos;     // flat: [x0,y0,z0, x1,y1,z1, ...]
    std::vector<double> rho;
    std::vector<double> vol;
    double BoxSize;
    size_t npart;
};

inline ArepoSnapshot read_arepo_snap(const std::string& filename) {
    ArepoSnapshot snap;

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
        throw std::runtime_error("Cannot open " + filename);

    // Read BoxSize
    {
        hid_t grp = H5Gopen2(file, "Parameters", H5P_DEFAULT);
        hid_t attr = H5Aopen(grp, "BoxSize", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &snap.BoxSize);
        H5Aclose(attr);
        H5Gclose(grp);
    }

    // Read Coordinates
    {
        hid_t dset = H5Dopen2(file, "PartType0/Coordinates", H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        snap.npart = dims[0];
        snap.pos.resize(snap.npart * 3);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, snap.pos.data());
        H5Sclose(space);
        H5Dclose(dset);
    }

    // Read Density
    {
        hid_t dset = H5Dopen2(file, "PartType0/Density", H5P_DEFAULT);
        snap.rho.resize(snap.npart);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, snap.rho.data());
        H5Dclose(dset);
    }

    // Read Masses and compute volumes
    {
        hid_t dset = H5Dopen2(file, "PartType0/Masses", H5P_DEFAULT);
        std::vector<double> mass(snap.npart);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mass.data());
        H5Dclose(dset);

        snap.vol.resize(snap.npart);
        for (size_t i = 0; i < snap.npart; i++)
            snap.vol[i] = mass[i] / snap.rho[i];
    }

    H5Fclose(file);
    return snap;
}

#endif // HDF5_READER_HPP
