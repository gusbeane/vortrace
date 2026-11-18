"""Setup shared by the yt comparison scripts.

The snapshot is an isolated dwarf galaxy (27 million cells, 180 kpc box).
Both codes project the gas density along z through a 4 kpc slab.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
SNAPSHOT = DATA / "snap_500.0.hdf5"          # first of four parts
WORK = DATA / "external" / "runs" / "yt"     # images and logs, not in git

BOX = 180.0                                  # kpc
ZRANGE = [88.0, 92.0]                        # kpc

# Images to make: name -> (centre x, centre y, width) in kpc, and pixels.
# "main" is the image that is timed.  The other two are the figure insets:
# an underdense region and a dense knot.
REGIONS = {
    "main": (90.0, 90.0, 6.0, 1024),
    "void": (89.75, 92.0, 0.25, 256),
    "dense": (89.5283203125, 89.3701171875, 0.015, 256),
}
