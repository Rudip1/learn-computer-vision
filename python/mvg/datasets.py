"""Synthetic data for the notebooks: scenes, cameras, images and sequences with exact ground truth.

Generating test data is not an algorithm of the module, so it lives here in Python; every estimator that the
notebooks run on these data is in the C++ core.
"""

from __future__ import annotations

import numpy as np


def cube(size=1.0, origin=(0.0, 0.0, 0.0)):
    """Corners (8 x 3) and edges (list of index pairs) of an axis-aligned cube."""
    corners = np.array([[(i & 1), (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float) * size
    corners += np.asarray(origin, dtype=float)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return corners, edges


def random_points(n, lo, hi, rng=None):
    """n points uniformly distributed in the box [lo, hi] (each a 3-vector)."""
    rng = np.random.default_rng(rng)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return lo + (hi - lo) * rng.random((n, 3))


def checkerboard_points(rows, cols, square):
    """Inner-corner grid of a planar target on Z = 0, (rows*cols) x 3, row-major from the origin."""
    j, i = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([j.ravel() * square, i.ravel() * square, np.zeros(rows * cols)])


# --------------------------------------------------------------------------------------------------------------
# Rendering of textured planes (used for calibration targets, markers and image pairs)
# --------------------------------------------------------------------------------------------------------------

def checkerboard_texture(rows, cols, square):
    """Texture function of a checkerboard with rows x cols squares of side `square`, top-left corner at the
    origin of the plane. Returns f(XY) -> intensity in [0, 1], NaN outside the board (with a white margin of one
    square). Its inner corners are checkerboard_points(rows - 1, cols - 1, square) shifted by one square."""
    def texture(xy):
        i = np.floor(xy[:, 1] / square).astype(int)
        j = np.floor(xy[:, 0] / square).astype(int)
        val = np.where((i + j) % 2 == 0, 0.08, 0.92)
        inside = (i >= 0) & (i < rows) & (j >= 0) & (j < cols)
        margin = (i >= -1) & (i <= rows) & (j >= -1) & (j <= cols)
        out = np.where(inside, val, np.where(margin, 0.92, np.nan))
        return out
    return texture


def render_plane(camera, texture, background=0.35, supersample=2, noise=0.0, rng=None):
    """Render the plane Z = 0 of the world, painted with `texture(XY) -> intensity`, as seen by `camera`
    (an mvg.PinholeCamera, distortion included). Returns a float image in [0, 1] of shape (height, width).

    Each pixel averages supersample x supersample rays (anti-aliasing). Rays that miss the plane, hit it behind
    the camera, or where the texture returns NaN, get the background value."""
    w, h, s = camera.width, camera.height, supersample
    offs = (np.arange(s) + 0.5) / s - 0.5
    u = (np.arange(w)[None, :, None, None] + offs[None, None, None, :]) * np.ones((h, 1, s, 1))
    v = (np.arange(h)[:, None, None, None] + offs[None, None, :, None]) * np.ones((1, w, 1, s))
    px = np.column_stack([u.ravel(), v.ravel()])
    d = camera.rays(px)
    c = camera.center()
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -c[2] / d[:, 2]
    hit = lam > 0
    xy = c[:2] + lam[:, None] * d[:, :2]
    val = np.full(len(px), np.nan)
    val[hit] = texture(xy[hit])
    val = np.where(np.isnan(val), background, val)
    img = val.reshape(h, w, s, s).mean(axis=(2, 3))
    if noise > 0:
        img = img + np.random.default_rng(rng).normal(scale=noise, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def orbit_cameras(K, n, target=(0.0, 0.0, 0.0), distance=0.6, tilt=(0.35, 0.6), roll=0.3, distortion=None,
                  width=640, height=480, rng=None):
    """n cameras looking at `target` on the plane Z = 0 from tilted viewpoints spread around it (the
    configuration recommended for planar calibration, section 3.6). The board's +Y axis appears as image-down."""
    import mvg

    rng = np.random.default_rng(rng)
    target = np.asarray(target, dtype=float)
    cams = []
    for k in range(n):
        az = 2 * np.pi * k / n + rng.uniform(-0.2, 0.2)
        t = rng.uniform(*tilt)
        dist = distance * rng.uniform(0.85, 1.15)
        C = target + dist * np.array([np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), -np.cos(t)])
        aim = target + np.r_[rng.uniform(-0.02, 0.02, 2), 0.0]
        R = mvg.so3_exp([0.0, 0.0, rng.uniform(-roll, roll)]) @ mvg.look_at(C, aim, [0.0, -1.0, 0.0])
        cams.append(mvg.PinholeCamera(K, width, height, R, -R @ C,
                                      distortion if distortion is not None else mvg.Distortion()))
    return cams
