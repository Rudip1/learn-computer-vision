#!/usr/bin/env python3
"""Generate the figures of 1_theory/figures/ from the mvg module. Figures are never edited by hand.

    python tools/make_figures.py          # all figures
    python tools/make_figures.py 01 05    # figures of chapters 1 and 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import mvg  # noqa: E402
from mvg.plotting import grid_lines, setup  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "1_theory" / "figures"
FIGURES = {}


def figure(chapter: str):
    def register(fn):
        FIGURES.setdefault(chapter, []).append(fn)
        return fn
    return register


def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote 1_theory/figures/{name}")


@figure("01")
def hierarchy():
    A = np.array([[1.0, 0.6], [0.0, 0.8]])
    examples = {
        "translation (2 dof)": mvg.make_affine(np.eye(2), [0.5, 0.3]),
        "Euclidean (3 dof)": mvg.make_euclidean(0.5, 0.5, 0.3),
        "similarity (4 dof)": mvg.make_similarity(0.6, 0.5, 0.5, 0.3),
        "affine (6 dof)": mvg.make_affine(A, [0.5, 0.3]),
        "projective (8 dof)": mvg.make_affine(A, [0.0, 0.0]) @ np.array([[1, 0, 0], [0, 1, 0], [0.2, 0.15, 1.0]]),
    }
    t = np.linspace(0, 2 * np.pi, 100)
    circle = 0.5 * np.c_[np.cos(t), np.sin(t)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    for ax, (name, H) in zip(axes, examples.items()):
        for g in grid_lines(5):
            ax.plot(*g.T, color="0.8", lw=0.8)
            ax.plot(*mvg.apply_homography(H, g).T, color="C0", lw=1.2)
        ax.plot(*mvg.apply_homography(H, circle).T, color="C1", lw=1.5)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("The same grid and circle (grey: original) under one transformation of each class", y=1.02)
    save(fig, "01_hierarchy.png")


@figure("02")
def pinhole():
    from mvg.datasets import cube, random_points
    from mvg.plotting import draw_camera, image_axes, set_axes_equal

    corners, edges = cube(1.0, origin=(-0.5, -0.5, 0.0))
    cloud = random_points(40, [-1.5, -1.5, 0.0], [1.5, 1.5, 1.5], rng=2)
    C = np.array([3.0, -2.5, 1.8])
    R = mvg.look_at(C, [0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
    cam = mvg.PinholeCamera(mvg.make_intrinsics(520.0, 520.0, 320.0, 240.0), 640, 480, R, -R @ C)
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for a, b in edges:
        ax.plot(*corners[[a, b]].T, color="C1")
    ax.scatter(*cloud.T, s=6, color="0.4")
    draw_camera(ax, cam, depth=0.8)
    for r in cam.rays(cam.project(corners)):
        ax.plot(*np.vstack([C, C + 5.5 * r]).T, color="C2", lw=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    ax.set_title("world: camera frustum (x red, y green, z blue)")
    ax = fig.add_subplot(1, 2, 2)
    px = cam.project(corners)
    for a, b in edges:
        ax.plot(*px[[a, b]].T, color="C1")
    ax.scatter(*cam.project(cloud[cam.is_visible(cloud)]).T, s=8, color="0.4")
    image_axes(ax, 640, 480, "image: x ~ K [R | t] X")
    plt.tight_layout()
    save(fig, "02_pinhole.png")


@figure("02")
def distortion():
    g = np.linspace(-0.6, 0.6, 13)
    lines = [np.column_stack([np.full(60, c), np.linspace(-0.6, 0.6, 60)]) for c in g]
    lines += [ln[:, ::-1] for ln in lines]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, k1, title in zip(axes, [0.0, -0.3, 0.3], ["no distortion", "barrel, k1 = -0.3", "pincushion, k1 = 0.3"]):
        d = mvg.Distortion(k1=k1)
        for ln in lines:
            ax.plot(*mvg.distort(ln, d).T, color="C0", lw=0.8)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(0.75, -0.75)
        ax.set_xlabel("x_d")
    axes[0].set_ylabel("y_d")
    plt.tight_layout()
    save(fig, "02_distortion.png")


@figure("03")
def calibration_points():
    from mvg.datasets import random_points

    rng = np.random.default_rng(3)
    K = mvg.make_intrinsics(560.0, 710.0, 326.0, 299.0)
    R = mvg.rotation_from_euler("XYX", [0.3, -0.5, 0.2])
    cam = mvg.PinholeCamera(K, 640, 480, R, -R @ np.array([0.4, -0.3, -3.0]))
    box = ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    X_test = random_points(500, *box, rng=99)
    x_test = cam.project(X_test)
    sigma, Ns, train, test = 0.5, [6, 7, 8, 10, 15, 20, 30, 50, 100], [], []
    for n in Ns:
        tr, te = [], []
        for _ in range(200):
            X = random_points(n, *box, rng=rng)
            x = cam.project(X) + rng.normal(scale=sigma, size=(n, 2))
            P = mvg.estimate_projection_dlt(X, x)
            tr.append(np.sqrt(np.mean(np.sum((mvg.project(P, X) - x) ** 2, axis=1))))
            te.append(np.sqrt(np.mean(np.sum((mvg.project(P, X_test) - x_test) ** 2, axis=1))))
        train.append(np.median(tr))
        test.append(np.median(te))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.loglog(Ns, train, "o-", label="on the N fitted points")
    ax.loglog(Ns, test, "s-", label="on 500 held-out points")
    ax.axhline(sigma * np.sqrt(2), color="k", ls="--", label=r"$\sigma\sqrt{2}$")
    ax.set_xlabel("number of points N")
    ax.set_ylabel("median RMS reprojection error [px]")
    ax.set_title(r"DLT with $\sigma$ = 0.5 px noise, 200 trials per N")
    ax.legend()
    save(fig, "03_points_vs_error.png")


@figure("03")
def calibration_residuals():
    from mvg.datasets import checkerboard_points, orbit_cameras
    from mvg.plotting import image_axes

    rng = np.random.default_rng(0)
    K = mvg.make_intrinsics(820.0, 805.0, 322.0, 236.0)
    dist = mvg.Distortion(k1=-0.25, k2=0.08, p1=6e-4, p2=-4e-4)
    board = checkerboard_points(6, 9, 0.04) + [0.04, 0.04, 0.0]
    cams = orbit_cameras(K, 12, target=(0.2, 0.14, 0.0), distance=0.9, distortion=dist, rng=0)
    obs = [c.project(board) + rng.normal(scale=0.1, size=(len(board), 2)) for c in cams]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fix in zip(axes, [False, True]):
        opt = mvg.CalibrationOptions()
        opt.fix_distortion = fix
        r = mvg.calibrate_planar(board, obs, opt)
        for v, o in enumerate(obs):
            c = mvg.PinholeCamera(r.K, 640, 480, r.R[v], r.t[v], r.distortion)
            e = o - c.project(board)
            ax.quiver(*o.T, *e.T, angles="xy", scale_units="xy", scale=0.02, width=0.003)
        title = "distortion fixed to zero" if fix else "full model"
        image_axes(ax, 640, 480, f"{title}: RMS {r.rms:.2f} px (residuals x 50)")
    plt.tight_layout()
    save(fig, "03_residuals.png")


def main(argv: list[str]) -> int:
    setup()
    chapters = argv or sorted(FIGURES)
    for ch in chapters:
        for fn in FIGURES.get(ch, []):
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
