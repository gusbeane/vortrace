"""Quick plotting helpers for vortrace projections.

``matplotlib`` is imported lazily inside each function, so
``import vortrace.plot`` succeeds even without matplotlib installed.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike


def plot_grid(data: ArrayLike, *, extent: ArrayLike | None = None,
              ax: Any = None, log: bool = True, cmap: str = "inferno",
              colorbar: bool = True, label: str | None = None,
              **imshow_kwargs: Any) -> tuple[Any, Any, Any]:
    """Display a 2-D projection grid as an image.

    Parameters
    ----------
    data : array_like
        2-D projection array (will be transposed for display so that the
        first axis corresponds to *x* and the second to *y*).
    extent : array_like, optional
        ``[xmin, xmax, ymin, ymax]`` passed to ``imshow``.
    ax : matplotlib Axes, optional
        Axes to plot into.  A new figure is created if *None*.
    log : bool
        If *True* (default), use ``LogNorm``.
    cmap : str
        Colormap name (default ``"inferno"``).
    colorbar : bool
        Whether to add a colorbar.
    label : str, optional
        Colorbar label.
    **imshow_kwargs
        Extra keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    im : matplotlib AxesImage
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    data = np.asarray(data)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    kwargs = {"origin": "lower", "cmap": cmap}
    if extent is not None:
        kwargs["extent"] = extent
    if log:
        kwargs["norm"] = mcolors.LogNorm()
    kwargs.update(imshow_kwargs)

    im = ax.imshow(data.T, **kwargs)

    if colorbar:
        fig.colorbar(im, ax=ax, label=label or "")

    return fig, ax, im


def plot_ray(s_vals: ArrayLike, dens: ArrayLike, *, ax: Any = None,
             log: bool = True, **plot_kwargs: Any) -> tuple[Any, Any]:
    """Plot a 1-D ray density profile.

    Parameters
    ----------
    s_vals : array_like
        Position along the ray.
    dens : array_like
        Density values at each position.
    ax : matplotlib Axes, optional
        Axes to plot into.  A new figure is created if *None*.
    log : bool
        If *True* (default), use a log scale for the y-axis.
    **plot_kwargs
        Extra keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    ax.plot(s_vals, dens, **plot_kwargs)

    if log:
        ax.set_yscale("log")

    return fig, ax
