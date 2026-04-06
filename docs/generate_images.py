#!/usr/bin/env python
"""Generate all tutorial images for the vortrace documentation.

Run from the repository root (or the docs/ directory):

    python docs/generate_images.py

Images are written to docs/images/.  This script is called automatically
by ``make html`` (via the Sphinx conf.py build-started hook), but you can
also run it standalone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # docs/
REPO_ROOT = SCRIPT_DIR.parent
IMAGE_DIR = SCRIPT_DIR / "images"
SNAP_GALAXY = REPO_ROOT / "tests" / "test_data" / "galaxy_interaction.hdf5"
SNAP_COSMO = REPO_ROOT / "tests" / "test_data" / "cosmo_box.hdf5"

# Make sure vortrace is importable even when running from docs/
sys.path.insert(0, str(REPO_ROOT))

import h5py as h5
import vortrace as vt

# ---------------------------------------------------------------------------
# Units (Arepo conventions for these snapshots)
# ---------------------------------------------------------------------------
MASS_UNIT = r"$10^{10}\;M_\odot$"
LENGTH_UNIT = "kpc"
DENSITY_UNIT = r"$10^{10}\;M_\odot\,\mathrm{kpc}^{-3}$"
COL_DENSITY_UNIT = r"$10^{10}\;M_\odot\,\mathrm{kpc}^{-2}$"

# ---------------------------------------------------------------------------
# Physical constants (cgs)
# ---------------------------------------------------------------------------
HYDROGEN_MASSFRAC = 0.76
GAMMA = 5.0 / 3.0
PROTONMASS = 1.67262178e-24
BOLTZMANN = 1.38065e-16


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _compute_temperature(f):
    """Compute gas temperature from Arepo snapshot fields."""
    UnitLength = float(f["Parameters"].attrs["UnitLength_in_cm"])
    UnitMass = float(f["Parameters"].attrs["UnitMass_in_g"])
    UnitVelocity = float(f["Parameters"].attrs["UnitVelocity_in_cm_per_s"])
    UnitEnergy = UnitMass * UnitVelocity ** 2

    u = np.array(f["PartType0/InternalEnergy"])
    xe = np.array(f["PartType0/ElectronAbundance"])

    mu = 4 * PROTONMASS / (1 + 3 * HYDROGEN_MASSFRAC
                           + 4 * HYDROGEN_MASSFRAC * xe)
    T = (GAMMA - 1.0) * (u / BOLTZMANN) * (UnitEnergy / UnitMass) * mu
    return T


def load_galaxy():
    """Load galaxy_interaction snapshot."""
    with h5.File(SNAP_GALAXY, "r") as f:
        pos = np.array(f["PartType0/Coordinates"])
        rho = np.array(f["PartType0/Density"])
        mass = np.array(f["PartType0/Masses"])
        box_size = float(f["Parameters"].attrs["BoxSize"])
        temperature = _compute_temperature(f)
        sfr = np.array(f["PartType0/StarFormationRate"])
    vol = mass / rho
    return pos, rho, vol, temperature, sfr, box_size


def load_cosmo():
    """Load cosmo_box snapshot."""
    with h5.File(SNAP_COSMO, "r") as f:
        pos = np.array(f["PartType0/Coordinates"])
        rho = np.array(f["PartType0/Density"])
        mass = np.array(f["PartType0/Masses"])
        box_size = float(f["Parameters"].attrs["BoxSize"])
    vol = mass / rho
    return pos, rho, vol, box_size


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

def make_quickstart(pc, box_size):
    """quickstart.png — single column-density projection."""
    L = 75.0
    extent = [box_size / 2 - L / 2, box_size / 2 + L / 2]
    bounds = [0, box_size]
    npix = 256

    image = pc.grid_projection(extent, npix, bounds, center=None)

    fig, ax, _ = vt.plot.plot_grid(
        image,
        extent=[-L / 2, L / 2, -L / 2, L / 2],
        label=f"Column density [{COL_DENSITY_UNIT}]",
    )
    ax.set_xlabel(f"x [{LENGTH_UNIT}]")
    ax.set_ylabel(f"y [{LENGTH_UNIT}]")
    _save(fig, "quickstart.png")


def make_grid_projection(pc, box_size):
    """grid_projection.png — side-by-side xy and xz projections."""
    L = 75.0
    extent = [box_size / 2 - L / 2, box_size / 2 + L / 2]
    bounds = [0, box_size]
    npix = 256

    image_xy = pc.grid_projection(extent, npix, bounds, center=None)

    center = [box_size / 2, box_size / 2, box_size / 2]
    image_xz = pc.grid_projection(extent, npix, bounds, center, proj="xz")

    ext = [-L / 2, L / 2, -L / 2, L / 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vt.plot.plot_grid(image_xy, extent=ext, ax=axes[0],
                      label=f"Column density [{COL_DENSITY_UNIT}]")
    axes[0].set_xlabel(f"x [{LENGTH_UNIT}]")
    axes[0].set_ylabel(f"y [{LENGTH_UNIT}]")
    axes[0].set_title("xy projection")

    vt.plot.plot_grid(image_xz, extent=ext, ax=axes[1],
                      label=f"Column density [{COL_DENSITY_UNIT}]")
    axes[1].set_xlabel(f"x [{LENGTH_UNIT}]")
    axes[1].set_ylabel(f"z [{LENGTH_UNIT}]")
    axes[1].set_title("xz projection")

    fig.tight_layout()
    _save(fig, "grid_projection.png")


def make_multifield(pos, rho, vol, temperature, box_size):
    """multifield.png — three-panel: column density, density-weighted T, mass-weighted T."""
    fields = np.column_stack([rho, rho * temperature])
    pc = vt.ProjectionCloud(
        pos, fields, vol=vol,
        boundbox=[0, box_size, 0, box_size, 0, box_size],
    )

    L = 75.0
    extent = [box_size / 2 - L / 2, box_size / 2 + L / 2]
    bounds = [0, box_size]
    npix = 256

    dat = pc.grid_projection(extent, npix, bounds, center=None)

    T_map = dat[:, :, 1] / dat[:, :, 0]

    ext = [-L / 2, L / 2, -L / 2, L / 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vt.plot.plot_grid(dat[:, :, 0], extent=ext, ax=axes[0],
                      label=f"Column density [{COL_DENSITY_UNIT}]")
    axes[0].set_xlabel(f"x [{LENGTH_UNIT}]")
    axes[0].set_ylabel(f"y [{LENGTH_UNIT}]")
    axes[0].set_title("Column density")

    vt.plot.plot_grid(dat[:, :, 1], extent=ext, ax=axes[1],
                      label=r"$\Sigma \cdot T$")
    axes[1].set_xlabel(f"x [{LENGTH_UNIT}]")
    axes[1].set_ylabel(f"y [{LENGTH_UNIT}]")
    axes[1].set_title(r"$\int \rho \, T \, dl$")

    vt.plot.plot_grid(np.log10(T_map), extent=ext, ax=axes[2],
                      log=False, label=r"$\log_{10}\,T$ [K]", cmap="bwr")
    axes[2].set_xlabel(f"x [{LENGTH_UNIT}]")
    axes[2].set_ylabel(f"y [{LENGTH_UNIT}]")
    axes[2].set_title("Mass-weighted T")

    fig.tight_layout()
    _save(fig, "multifield.png")


def make_volume_rendering(pos, rho, vol, temperature, sfr, box_size):
    """volume_rendering.png — four-panel volume rendering.

    Channels:  R = density,  G = star formation rate,  B = temperature.
    Opacity is driven by density.  Layout: large composite on top,
    three individual channels in a row below.
    """
    L = 75.0
    extent = [box_size / 2 - L / 2, box_size / 2 + L / 2]
    bounds = [0, box_size]
    npix = 256
    ext = [-L / 2, L / 2, -L / 2, L / 2]

    def _lognorm(q, plo=1, phi=99):
        """Log-scale and normalise *q* to [0, 1]."""
        lq = np.log10(np.clip(q, q[q > 0].min(), None))
        lo, hi = np.percentile(lq, plo), np.percentile(lq, phi)
        return np.clip((lq - lo) / (hi - lo), 0, 1)

    # --- Channels ---
    dens_norm = _lognorm(rho)
    R = 0.4 * dens_norm
    G = np.where(sfr > 0, _lognorm(np.where(sfr > 0, sfr, 1.0)), 0.0)
    hot = np.clip((np.log10(np.maximum(temperature, 1.0)) - np.log10(1.5e6)) / 1.5, 0, 1)
    B = 3.0 * hot * (1 - dens_norm) ** 2  # suppress in dense regions

    # --- Opacity: max of density-driven and temperature-driven ---
    alpha = np.maximum(0.08 * dens_norm ** 4,
                       0.01 * hot ** 2)

    fields_rgba = np.column_stack([R, G, B, alpha])
    pc_vol = vt.ProjectionCloud(
        pos, fields_rgba, vol=vol,
        boundbox=[0, box_size, 0, box_size, 0, box_size],
    )
    img_rgb = pc_vol.grid_projection(
        extent, npix, bounds, center=None, reduction="volume",
    )

    # Extract per-channel images
    z = np.zeros_like(img_rgb[:, :, 0])
    img_r = np.stack([img_rgb[:, :, 0], z, z], axis=-1)
    img_g = np.stack([z, img_rgb[:, :, 1], z], axis=-1)
    img_b = np.stack([z, z, img_rgb[:, :, 2]], axis=-1)

    def _prep(img, per_channel=False, gamma=1.0):
        """Normalise and transpose for imshow."""
        out = img.copy()
        if per_channel and out.ndim == 3:
            for c in range(out.shape[2]):
                mx = out[:, :, c].max()
                if mx > 0:
                    out[:, :, c] /= mx
        else:
            mx = out.max()
            if mx > 0:
                out /= mx
        if gamma != 1.0:
            out = np.clip(out, 0, None) ** (1.0 / gamma)
        return np.clip(np.swapaxes(out, 0, 1), 0, 1)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.25,
                          wspace=0.15)

    # Main composite image spanning all three columns
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(_prep(img_rgb, per_channel=True, gamma=2.0),
                   origin="lower", extent=ext)
    ax_main.set_xlabel(f"x [{LENGTH_UNIT}]")
    ax_main.set_ylabel(f"y [{LENGTH_UNIT}]")
    ax_main.set_title("Volume rendering")

    # Individual channel panels — each normalised independently
    for i, (img_ch, title) in enumerate([
        (img_r, "Density (R)"),
        (img_g, "SFR (G)"),
        (img_b, "Temperature (B)"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(_prep(img_ch), origin="lower", extent=ext)
        ax.set_xlabel(f"x [{LENGTH_UNIT}]")
        if i == 0:
            ax.set_ylabel(f"y [{LENGTH_UNIT}]")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=10)

    _save(fig, "volume_rendering.png")


def make_arbitrary_rays(pc, box_size):
    """arbitrary_rays.png — Mollweide sky map projection."""
    ntheta, nphi = 128, 256
    theta = np.linspace(0.01, np.pi - 0.01, ntheta)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")

    unitv = np.column_stack([
        np.sin(TT.ravel()) * np.cos(PP.ravel()),
        np.sin(TT.ravel()) * np.sin(PP.ravel()),
        np.cos(TT.ravel()),
    ])

    pts_end = 100 * unitv + box_size / 2
    pts_start = np.full_like(pts_end, box_size / 2)

    dens = pc.projection(pts_start, pts_end)
    sky = dens.reshape(ntheta, nphi)

    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": "mollweide"})
    lon = PP - np.pi  # mollweide needs [-pi, pi]
    lat = np.pi / 2 - TT
    pcm = ax.pcolormesh(lon, lat, np.log10(sky), cmap="inferno",
                        shading="auto")
    ax.set_title("Sky map projection")
    ax.grid(True, alpha=0.3)

    # Remove degree tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.colorbar(pcm, ax=ax, label=f"log Column density [{COL_DENSITY_UNIT}]",
                 orientation="horizontal", shrink=0.7, pad=0.05)

    _save(fig, "arbitrary_rays.png")


def make_single_ray(pc, rho, box_size):
    """single_ray.png — density profile along a single ray."""
    pt_start = np.array([box_size / 2 + 3, box_size / 2 + 10.5, 0])
    pt_end = np.array([box_size / 2 + 3, box_size / 2 + 10.5, box_size])

    _, cell_ids, s_vals, _ = pc.traced_projection(pt_start, pt_end)

    fig, ax = vt.plot.plot_ray(s_vals - box_size / 2, rho[cell_ids])
    ax.set_xlabel(f"z [{LENGTH_UNIT}]")
    ax.set_ylabel(f"Density [{DENSITY_UNIT}]")
    _save(fig, "single_ray.png")


def make_periodic(pos_cosmo, rho_cosmo, vol_cosmo, box_size_cosmo):
    """periodic.png — periodic-box projection."""
    pc = vt.ProjectionCloud(
        pos_cosmo, rho_cosmo, vol=vol_cosmo,
        boundbox=[0, box_size_cosmo, 0, box_size_cosmo, 0, box_size_cosmo],
        periodic=True,
    )

    image = pc.grid_projection(
        extent=[0, box_size_cosmo],
        nres=256,
        bounds=[0, box_size_cosmo],
        center=None,
    )

    fig, ax, _ = vt.plot.plot_grid(
        image,
        extent=[0, box_size_cosmo, 0, box_size_cosmo],
        label=f"Column density [{COL_DENSITY_UNIT}]",
    )
    ax.set_xlabel(f"x [{LENGTH_UNIT}]")
    ax.set_ylabel(f"y [{LENGTH_UNIT}]")
    _save(fig, "periodic.png")


def make_io(pc, box_size):
    """io_loaded.png — demonstrate save/load round-trip."""
    import tempfile

    L = 75.0
    extent = [box_size / 2 - L / 2, box_size / 2 + L / 2]
    bounds = [0, box_size]
    npix = 256

    image = pc.grid_projection(extent, npix, bounds, center=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "projection.npz")
        vt.io.save_grid(path, image,
                        extent=[-L / 2, L / 2, -L / 2, L / 2])
        data, meta = vt.io.load_grid(path)

    fig, ax, _ = vt.plot.plot_grid(
        data,
        extent=meta["extent"],
        label=f"Column density [{COL_DENSITY_UNIT}]",
    )
    ax.set_xlabel(f"x [{LENGTH_UNIT}]")
    ax.set_ylabel(f"y [{LENGTH_UNIT}]")
    _save(fig, "io_loaded.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name):
    """Save a figure and close it."""
    path = IMAGE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading galaxy_interaction snapshot...")
    pos, rho, vol, temperature, sfr, box_size = load_galaxy()

    # Build the single-field ProjectionCloud once and reuse it
    print("Building ProjectionCloud (single field)...")
    pc = vt.ProjectionCloud(
        pos, rho, vol=vol,
        boundbox=[0, box_size, 0, box_size, 0, box_size],
    )

    print("Generating images:")
    make_quickstart(pc, box_size)
    make_grid_projection(pc, box_size)
    make_single_ray(pc, rho, box_size)
    make_arbitrary_rays(pc, box_size)
    make_io(pc, box_size)

    # These need different fields, so they build their own ProjectionCloud
    make_multifield(pos, rho, vol, temperature, box_size)
    make_volume_rendering(pos, rho, vol, temperature, sfr, box_size)

    print("Loading cosmo_box snapshot...")
    pos_cosmo, rho_cosmo, vol_cosmo, box_size_cosmo = load_cosmo()

    print("Generating images:")
    make_periodic(pos_cosmo, rho_cosmo, vol_cosmo, box_size_cosmo)

    print("Done.")


if __name__ == "__main__":
    main()
