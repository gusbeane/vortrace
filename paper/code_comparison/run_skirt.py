"""Time the same projection with SKIRT.  Usage: python run_skirt.py

SKIRT builds the Voronoi mesh with Voro++ and a "probe" then integrates the
density along one ray per pixel.  SKIRT works in pc and Msun, and its rays
must start outside the box, so the box is shifted to be centred on z = 0.
"""
import re
import struct
import time
from datetime import datetime

import numpy as np

import common
from common import HI, LO, NPIX

work = common.workdir("skirt")
pos, rho, vol = common.load_cells()
start = time.perf_counter()

# Input file in SKIRT's binary column format: x, y, z [pc], rho [Msun/pc^3].
z_mid = (LO[2] + HI[2]) / 2
table = np.column_stack([(pos - [0, 0, z_mid]) * 1e3, rho * 10.0])
with open(work / "cells.scol", "wb") as f:
    f.write(b"SKIRT X\n")
    f.write(struct.pack("<QQQQ", 0x010203040A0BFEFF, 0, *table.shape))
    for label in ["x", "y", "z", "rho", "pc", "pc", "pc", "Msun/pc3"]:
        f.write(label.encode().ljust(8))
    table.astype("<f8").tofile(f)
    f.write(b"SCOLEND\n")

# Parameter file: no photons, just the mesh and a column-density probe
# looking down z (roll 90 deg makes the image axes the model x and y).
lo, hi = (LO - [0, 0, z_mid]) * 1e3, (HI - [0, 0, z_mid]) * 1e3
box = (f'minX="{lo[0]} pc" maxX="{hi[0]} pc" minY="{lo[1]} pc" '
       f'maxY="{hi[1]} pc" minZ="{lo[2]} pc" maxZ="{hi[2]} pc"')
width, center = hi[0] - lo[0], (lo + hi) / 2
(work / "proj.ski").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9">
  <MonteCarloSimulation userLevel="Expert" simulationMode="ExtinctionOnly"
                        numPackets="0">
    <random type="Random"><Random seed="0"/></random>
    <units type="Units"><ExtragalacticUnits/></units>
    <cosmology type="Cosmology"><LocalUniverseCosmology/></cosmology>
    <sourceSystem type="SourceSystem">
      <SourceSystem minWavelength="0.09 micron" maxWavelength="100 micron"
                    wavelengths="0.55 micron" sourceBias="0.5"/>
    </sourceSystem>
    <mediumSystem type="MediumSystem">
      <MediumSystem>
        <media type="Medium">
          <VoronoiMeshMedium filename="cells.scol" {box}
                             massType="MassDensity" massFraction="1"
                             importMetallicity="false"
                             importTemperature="false"
                             importVelocity="false"
                             importMagneticField="false"
                             importVariableMixParams="false" useColumns="">
            <materialMix type="MaterialMix"><MeanInterstellarDustMix/></materialMix>
          </VoronoiMeshMedium>
        </media>
        <samplingOptions type="SamplingOptions">
          <SamplingOptions numDensitySamples="1" numPropertySamples="1"/>
        </samplingOptions>
        <grid type="SpatialGrid">
          <VoronoiMeshSpatialGrid {box} policy="ImportedMesh"/>
        </grid>
      </MediumSystem>
    </mediumSystem>
    <instrumentSystem type="InstrumentSystem"><InstrumentSystem/></instrumentSystem>
    <probeSystem type="ProbeSystem">
      <ProbeSystem>
        <probes type="Probe">
          <DensityProbe probeName="coldens" aggregation="Type"
                        probeAfter="Setup">
            <form type="Form">
              <ParallelProjectionForm inclination="0 deg" azimuth="0 deg"
                  roll="90 deg" numSampling="1"
                  fieldOfViewX="{width} pc" numPixelsX="{NPIX}"
                  centerX="{center[0]} pc"
                  fieldOfViewY="{width} pc" numPixelsY="{NPIX}"
                  centerY="{center[1]} pc"/>
            </form>
          </DensityProbe>
        </probes>
      </ProbeSystem>
    </probeSystem>
  </MonteCarloSimulation>
</skirt-simulation-hierarchy>
""")

skirt = common.EXTERNAL / "SKIRT9_build" / "SKIRT" / "main" / "skirt"
run_s, peak_gib = common.run(
    [skirt, "-t", common.NTHREADS, "-b", "-i", work, "-o", work, "proj.ski"],
    work / "log.txt")

# Output: a FITS image of 32-bit floats in Msun/pc^2, stored [y, x].  The
# header is a whole number of 2880-byte blocks ending with an END card.
raw = (work / "proj_coldens_dust_Sigma.fits").read_bytes()
header_size = (raw.index(b"END" + b" " * 77) // 2880 + 1) * 2880
assert b"Msun/pc2" in raw[:header_size]
image = np.frombuffer(raw, ">f4", NPIX * NPIX, header_size)
image = image.reshape(NPIX, NPIX).T.astype(float) / 1e4
total_s = time.perf_counter() - start


def seconds_between(first, last):
    """Time between the first log lines containing ``first`` and ``last``.

    SKIRT does not report phase timings, but every line of its log starts
    with a timestamp such as '12/11/2026 16:20:28.135'.
    """
    stamps = []
    for text in (first, last):
        line = next(l for l in open(work / "proj_log.txt") if text in l)
        stamps.append(datetime.strptime(line[:23], "%d/%m/%Y %H:%M:%S.%f"))
    return (stamps[1] - stamps[0]).total_seconds()


common.save("skirt", image, ncells=len(rho),
            build_s=seconds_between("Constructing Voronoi tessellation",
                                    "Done computing Voronoi tessellation"),
            trace_s=seconds_between("Starting setup output",
                                    "Finished setup output"),
            run_s=run_s, total_s=total_s, peak_gib=peak_gib)
