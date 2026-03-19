"""Save and load projection grids and point clouds.

Supports two formats:
- **npz** (default) -- zero extra dependencies, fast.
- **hdf5** -- requires ``h5py``; natural for astronomy workflows.

The format is chosen via the *fmt* parameter on save, and auto-detected
from the file extension on load.
"""

import json
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_h5py():
    """Lazy import of h5py with a clear error message."""
    try:
        import h5py  # noqa: F811
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 I/O. "
            "Install it with:  pip install h5py"
        ) from exc
    return h5py


def _detect_format(filename):
    """Return 'hdf5' or 'npz' based on file extension."""
    fname = str(filename)
    if fname.endswith(('.hdf5', '.h5')):
        return 'hdf5'
    return 'npz'


# ---------------------------------------------------------------------------
# Grid I/O
# ---------------------------------------------------------------------------

def save_grid(filename, data, *, extent=None, metadata=None, fmt="npz"):
    """Save a 2-D projection array to disk.

    Parameters
    ----------
    filename : str or path-like
        Destination path.
    data : array_like
        2-D projection grid.
    extent : array_like, optional
        Spatial extent, e.g. ``[xmin, xmax, ymin, ymax]``.
    metadata : dict, optional
        Arbitrary metadata stored alongside the grid.
    fmt : {'npz', 'hdf5'}
        File format.
    """
    data = np.asarray(data)

    if fmt == "npz":
        arrays = {"data": data}
        if extent is not None:
            arrays["extent"] = np.asarray(extent)
        if metadata is not None:
            arrays["_metadata"] = np.array(json.dumps(metadata))
        np.savez(filename, **arrays)

    elif fmt == "hdf5":
        h5py = _import_h5py()
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=data)
            if extent is not None:
                f.create_dataset("extent", data=np.asarray(extent))
            if metadata is not None:
                f.attrs["metadata"] = json.dumps(metadata)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'npz' or 'hdf5'.")


def load_grid(filename):
    """Load a projection grid saved by :func:`save_grid`.

    The format is auto-detected from the file extension.

    Parameters
    ----------
    filename : str or path-like
        Path to a ``.npz`` or ``.hdf5`` / ``.h5`` file.

    Returns
    -------
    data : np.ndarray
        The 2-D projection array.
    metadata : dict
        Metadata dictionary (empty if none was stored).
    """
    fmt = _detect_format(filename)

    if fmt == "npz":
        with np.load(filename, allow_pickle=False) as npz:
            data = npz["data"]
            metadata = {}
            if "_metadata" in npz:
                metadata = json.loads(str(npz["_metadata"]))
            if "extent" in npz:
                metadata["extent"] = npz["extent"]
        return data, metadata

    # hdf5
    h5py = _import_h5py()
    with h5py.File(filename, "r") as f:
        data = f["data"][:]
        metadata = {}
        if "metadata" in f.attrs:
            metadata = json.loads(f.attrs["metadata"])
        if "extent" in f:
            metadata["extent"] = f["extent"][:]
    return data, metadata


# ---------------------------------------------------------------------------
# Cloud I/O
# ---------------------------------------------------------------------------

def save_cloud(filename, cloud, *, fmt="npz"):
    """Save a :class:`~vortrace.vortrace.ProjectionCloud` to disk.

    Only the data needed to reconstruct the cloud (positions, fields,
    and bounding box) are stored.  The KD-tree is rebuilt on load.

    Parameters
    ----------
    filename : str or path-like
        Destination path.
    cloud : ProjectionCloud
        The cloud to save.
    fmt : {'npz', 'hdf5'}
        File format.
    """
    pos = np.asarray(cloud.pos_orig)
    fields = np.asarray(cloud.fields_orig)
    boundbox = np.asarray(cloud.boundbox)

    if fmt == "npz":
        np.savez(filename, pos=pos, fields=fields, boundbox=boundbox)

    elif fmt == "hdf5":
        h5py = _import_h5py()
        with h5py.File(filename, "w") as f:
            f.create_dataset("pos", data=pos)
            f.create_dataset("fields", data=fields)
            f.attrs["boundbox"] = boundbox
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'npz' or 'hdf5'.")


def load_cloud(filename):
    """Load a ProjectionCloud saved by :func:`save_cloud`.

    The KD-tree is rebuilt automatically during construction.

    Parameters
    ----------
    filename : str or path-like
        Path to a ``.npz`` or ``.hdf5`` / ``.h5`` file.

    Returns
    -------
    ProjectionCloud
        A fully usable cloud with a rebuilt KD-tree.
    """
    from vortrace.vortrace import ProjectionCloud

    fmt = _detect_format(filename)

    if fmt == "npz":
        with np.load(filename, allow_pickle=False) as npz:
            pos = npz["pos"]
            fields = npz["fields"]
            boundbox = npz["boundbox"]
    else:
        h5py = _import_h5py()
        with h5py.File(filename, "r") as f:
            pos = f["pos"][:]
            fields = f["fields"][:]
            boundbox = f.attrs["boundbox"]

    return ProjectionCloud(pos, fields, boundbox=list(boundbox))
