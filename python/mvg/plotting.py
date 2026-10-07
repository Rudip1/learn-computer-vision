"""Matplotlib helpers shared by the notebooks and the figure scripts. Drawing only, no geometry."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def setup(dpi=80):
    """Notebook defaults: modest resolution (keeps committed outputs small), light grid."""
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "image.cmap": "gray",
    })


def draw_line(ax, l, xlim=None, ylim=None, **kwargs):
    """Draw the homogeneous line ``l = (a, b, c)``, a*u + b*v + c = 0, clipped to the axes limits."""
    a, b, c = np.asarray(l, dtype=float)
    xlim = ax.get_xlim() if xlim is None else xlim
    ylim = ax.get_ylim() if ylim is None else ylim
    if abs(b) >= abs(a):
        u = np.array(xlim, dtype=float)
        v = -(a * u + c) / b
    else:
        v = np.array(ylim, dtype=float)
        u = -(b * v + c) / a
    return ax.plot(u, v, **kwargs)


def draw_polygon(ax, pts, closed=True, **kwargs):
    """Draw an N x 2 polygon."""
    pts = np.asarray(pts, dtype=float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    return ax.plot(pts[:, 0], pts[:, 1], **kwargs)


def grid_lines(n=5, lo=-1.0, hi=1.0, samples=30):
    """Polylines of a square n x n grid, as a list of (samples x 2) arrays (useful to visualise a warp)."""
    t = np.linspace(lo, hi, samples)
    lines = []
    for c in np.linspace(lo, hi, n):
        lines.append(np.column_stack([np.full_like(t, c), t]))
        lines.append(np.column_stack([t, np.full_like(t, c)]))
    return lines


def image_axes(ax, width, height, title=None):
    """Pixel axes: origin top-left, v pointing down, equal aspect."""
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    if title:
        ax.set_title(title)
    return ax


def new_figure(ncols=1, nrows=1, size=4.0, **kwargs):
    """A figure with ``nrows x ncols`` square-ish axes."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), **kwargs)
    return fig, axes


def frustum_lines(camera, depth=1.0):
    """World-frame polylines of a camera's viewing frustum (apex at the centre, base at the given depth)."""
    w, h = camera.width, camera.height
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    base = camera.backproject(corners, np.full(4, depth))
    c = camera.center()
    lines = [np.vstack([c, b]) for b in base]
    lines.append(np.vstack([base, base[:1]]))
    return lines


def draw_camera(ax3d, camera, depth=1.0, color="C0", label=None, axes=True):
    """Draw a camera frustum and (optionally) its x (red), y (green), z (blue) axes in a 3-D matplotlib axis."""
    for i, line in enumerate(frustum_lines(camera, depth)):
        ax3d.plot(*line.T, color=color, lw=1, label=label if i == 0 else None)
    if axes:
        c = camera.center()
        for k, col in enumerate(["r", "g", "b"]):
            d = camera.R[k] * depth * 0.6  # rows of R_cw are the camera axes in world coordinates
            ax3d.plot(*np.vstack([c, c + d]).T, color=col, lw=2)


def set_axes_equal(ax3d):
    """Equal scale on the three axes of a 3-D plot."""
    lims = np.array([ax3d.get_xlim3d(), ax3d.get_ylim3d(), ax3d.get_zlim3d()])
    centre, radius = lims.mean(axis=1), 0.5 * np.max(lims[:, 1] - lims[:, 0])
    ax3d.set_xlim3d(centre[0] - radius, centre[0] + radius)
    ax3d.set_ylim3d(centre[1] - radius, centre[1] + radius)
    ax3d.set_zlim3d(centre[2] - radius, centre[2] + radius)
