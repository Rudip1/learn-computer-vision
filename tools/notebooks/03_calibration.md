```python
# Install the module when it is not available (e.g. on Colab). Locally: `pip install -e .` from the repository root.
try:
    import mvg
except ImportError:
    %pip install -q git+https://github.com/Rudip1/learn-computer-vision
    import mvg
```

# 3 · Camera calibration

**Recap** ([theory](../../1_theory/03_calibration.md)).

* Normalize first: centroid at the origin, mean distance $\sqrt{2}$ (image) or $\sqrt{3}$ (scene), eq. (3.1).
* DLT: $\mathbf{x}_i \times P\mathbf{X}_i = \mathbf{0}$ gives two rows per point; $P$ is the last right singular vector.
* RQ decomposition splits $M = KR$; fix signs so that $\det R = 1$ and $K$ has a positive diagonal.
* Zhang: each view of a plane gives a homography and two linear constraints on $B = K^{-\top}K^{-1}$.
* The final answer minimizes the reprojection error with Levenberg–Marquardt, distortion included.

```python
import numpy as np
import matplotlib.pyplot as plt
import mvg
from mvg.datasets import random_points, checkerboard_points, checkerboard_texture, render_plane, orbit_cameras
from mvg.plotting import setup, image_axes

setup()
rng = np.random.default_rng(3)
np.set_printoptions(precision=4, suppress=True)
```

## 1. Calibrating from known 3-D points

A simulated camera looks at random points inside a cube of side 1 m. With exact pixels the DLT returns the true
camera; with 0.5 px of noise per coordinate (95 % of the errors within ±1 px) it does not.

```python
K_true = mvg.make_intrinsics(560.0, 710.0, 326.0, 299.0)
R_true = mvg.rotation_from_euler("XYX", [0.3, -0.5, 0.2])
C_true = np.array([0.4, -0.3, -3.0])
cam = mvg.PinholeCamera(K_true, 640, 480, R_true, -R_true @ C_true)

X = random_points(6, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], rng=rng)
x = cam.project(X)

for name, P in [("DLT", mvg.estimate_projection_dlt(X, x)), ("Hall", mvg.estimate_projection_hall(X, x))]:
    d = mvg.decompose_projection(P)
    print(f"{name}: K =\n{d.K}\n  rotation error = {np.degrees(mvg.rotation_angle_between(d.R, R_true)):.2e} deg, "
          f"centre = {d.center}")
```

```python
sigma = 0.5
x_noisy = x + rng.normal(scale=sigma, size=x.shape)
d = mvg.decompose_projection(mvg.estimate_projection_dlt(X, x_noisy))
print("K from 6 noisy points:\n", d.K)
print("skew s =", d.K[0, 1], " (the true camera has s = 0)")
```

The skew is not zero because the DLT estimates all 11 parameters freely; noise leaks into every one of them.

### ✏️ Exercise 3.1 — Hartley normalization

Return the $3\times 3$ similarity $T$ of eq. (3.1) for an $N \times 2$ array.

```python
def normalization_2d(x):
    # BEGIN SOLUTION
    c = x.mean(axis=0)
    s = np.sqrt(2) / np.linalg.norm(x - c, axis=1).mean()
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    # END SOLUTION
```

```python
pts = rng.uniform(100, 600, size=(40, 2))
T = normalization_2d(pts)
assert np.allclose(T, mvg.normalization_transform(pts))
xn = (T @ np.c_[pts, np.ones(40)].T).T[:, :2]
assert np.allclose(xn.mean(axis=0), 0) and np.isclose(np.linalg.norm(xn, axis=1).mean(), np.sqrt(2))
print("Exercise 3.1 passed")
```

### ✏️ Exercise 3.2 — the DLT in NumPy

Build the two rows of eq. (3.3) for each correspondence, solve with the SVD and denormalize with eq. (3.2).

```python
def dlt_projection(X, x):
    T = mvg.normalization_transform(x)       # 3 x 3
    U = mvg.normalization_transform(X)       # 4 x 4
    xh = (T @ np.c_[x, np.ones(len(x))].T).T
    Xh = (U @ np.c_[X, np.ones(len(X))].T).T
    # BEGIN SOLUTION
    rows = []
    for (u, v, w), Xi in zip(xh, Xh):
        rows.append(np.r_[np.zeros(4), -w * Xi, v * Xi])
        rows.append(np.r_[w * Xi, np.zeros(4), -u * Xi])
    _, _, Vt = np.linalg.svd(np.array(rows))
    P = np.linalg.inv(T) @ Vt[-1].reshape(3, 4) @ U
    # END SOLUTION
    return P / np.linalg.norm(P)
```

```python
P_mine, P_ref = dlt_projection(X, x_noisy), mvg.estimate_projection_dlt(X, x_noisy)
assert np.allclose(P_mine * np.sign(P_mine[2, 3]), P_ref * np.sign(P_ref[2, 3]), atol=1e-9)
print("Exercise 3.2 passed")
```

## 2. How many points?

Repeat the noisy estimate many times for each number of points $N$ and measure two different things:

* the **training error** — reprojection error on the $N$ points used for the fit;
* the **test error** — reprojection error on 500 other points seen by the same camera.

```python
X_test = random_points(500, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], rng=99)
x_test = cam.project(X_test)
Ns = [6, 7, 8, 10, 15, 20, 30, 50, 100]
train, test, focal = [], [], []
for n in Ns:
    tr, te, fe = [], [], []
    for _ in range(200):
        Xn = random_points(n, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], rng=rng)
        xn = cam.project(Xn) + rng.normal(scale=sigma, size=(n, 2))
        P = mvg.estimate_projection_dlt(Xn, xn)
        tr.append(np.sqrt(np.mean(np.sum((mvg.project(P, Xn) - xn) ** 2, axis=1))))
        te.append(np.sqrt(np.mean(np.sum((mvg.project(P, X_test) - x_test) ** 2, axis=1))))
        fe.append(abs(mvg.decompose_projection(P).K[0, 0] - K_true[0, 0]))
    train.append(np.median(tr)); test.append(np.median(te)); focal.append(np.median(fe))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(Ns, train, "o-", label="training points")
axes[0].plot(Ns, test, "s-", label="held-out points (true accuracy)")
axes[0].axhline(sigma * np.sqrt(2), color="k", ls="--", label=r"$\sigma\sqrt{2}$")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("number of points N")
axes[0].set_ylabel("median RMS reprojection error [px]"); axes[0].legend()
axes[1].plot(Ns, focal, "o-"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("number of points N"); axes[1].set_ylabel(r"median $|f_x - f_x^{true}|$ [px]")
plt.tight_layout()
```

With 6 points the fit passes through the data (training error ≈ 0) but generalizes badly. As $N$ grows the
training error *rises* towards $\sigma\sqrt{2}$ while the test error falls: only held-out data measure accuracy.

## 3. Planar calibration with a checkerboard

Twelve rendered images of a 10 × 7 checkerboard (40 mm squares) seen through a lens with barrel distortion. The
corners are found with OpenCV's detector when it is installed (detection is not the topic of this chapter).

```python
K_cal = mvg.make_intrinsics(820.0, 805.0, 322.0, 236.0)
dist_true = mvg.Distortion(k1=-0.25, k2=0.08, p1=6e-4, p2=-4e-4)
square = 0.04
board_obj = checkerboard_points(6, 9, square) + [square, square, 0.0]  # 9 x 6 inner corners
cams = orbit_cameras(K_cal, 12, target=(0.2, 0.14, 0.0), distance=0.9, distortion=dist_true, rng=0)
texture = checkerboard_texture(7, 10, square)
images = [render_plane(c, texture, noise=0.01, rng=k) for k, c in enumerate(cams)]

try:
    import cv2

    def detect(img):
        g = (img * 255).astype(np.uint8)
        ok, corners = cv2.findChessboardCorners(g, (9, 6))
        assert ok, "board not found"
        corners = cv2.cornerSubPix(g, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        return np.asarray(corners, dtype=float).reshape(-1, 2)
except ImportError:
    cv2 = None

    def detect(img, k=[0]):  # fallback: true projections plus detector-like noise
        k[0] += 1
        return cams[k[0] - 1].project(board_obj) + rng.normal(scale=0.1, size=(len(board_obj), 2))

observations = []
for c, img in zip(cams, images):
    corners = detect(img)
    truth = c.project(board_obj)
    # A checkerboard is symmetric: make the detected order match the object points (first corner = corner 0).
    if np.linalg.norm(corners[0] - truth[0]) > np.linalg.norm(corners[-1] - truth[0]):
        corners = corners[::-1]
    observations.append(corners)

errors = np.concatenate([np.linalg.norm(o - c.project(board_obj), axis=1) for o, c in zip(observations, cams)])
print(f"corner detector error: RMS {np.sqrt(np.mean(errors**2)):.3f} px, max {errors.max():.3f} px")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, k in zip(axes, [0, 4, 8]):
    ax.imshow(images[k], vmin=0, vmax=1)
    ax.plot(*observations[k].T, "r+", ms=6)
    ax.set_title(f"view {k}"); ax.axis("off")
plt.tight_layout()
```

**Closed form, then refinement.**

```python
res = mvg.calibrate_planar(board_obj, observations)
print("closed-form K (zero distortion):\n", res.K_linear, f"\n  RMS = {res.rms_linear:.3f} px")
print("refined K:\n", res.K, f"\n  RMS = {res.rms:.3f} px after {res.lm.iterations} LM iterations")
print("distortion  estimated:", res.distortion.as_vector(), "\n            true:     ", dist_true.as_vector())
print("Image radius covered by the corners (normalized):",
      max(np.linalg.norm(mvg.pixels_to_normalized(res.K, o), axis=1).max() for o in observations).round(3))

fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogy(np.sqrt(np.array(res.lm.cost_history) / (len(board_obj) * len(cams))), "o-")
ax.set_xlabel("LM iteration"); ax.set_ylabel("RMS reprojection error [px]");
```

$k_1$ is recovered well, $k_2$ much less: the corners reach a normalized radius of only about 0.3 (printed above), where
$k_2 r^4$ is small and strongly correlated with $k_1 r^2$. Higher-order terms need data near the image corners.

**Reference: OpenCV.** `cv2.calibrateCamera` implements the same method (Zhang + LM).

```python
if cv2 is not None:
    obj = [board_obj.astype(np.float32)] * len(observations)
    img_pts = [o.astype(np.float32).reshape(-1, 1, 2) for o in observations]
    rms_cv, K_cv, dist_cv, _, _ = cv2.calibrateCamera(obj, img_pts, (640, 480), None, None,
                                                      flags=cv2.CALIB_FIX_K3)
    print(f"{'':10}{'fx':>10}{'fy':>10}{'cx':>10}{'cy':>10}{'k1':>10}{'k2':>10}")
    for name, Km, dv in [("true", K_cal, dist_true.as_vector()), ("mvg", res.K, res.distortion.as_vector()),
                         ("OpenCV", K_cv, dist_cv.ravel())]:
        print(f"{name:10}" + "".join(f"{v:10.3f}" for v in [Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]])
              + "".join(f"{v:10.4f}" for v in dv[:2]))
    print(f"OpenCV RMS = {rms_cv:.3f} px, mvg RMS = {res.rms:.3f} px")
```

### ✏️ Exercise 3.3 — Zhang's constraints

Write the vector $\mathbf{v}_{ij}$ of eq. (3.9) and return the two rows of eq. (3.10) for one homography.

```python
def zhang_rows(H):
    # BEGIN SOLUTION
    def v(i, j):
        hi, hj = H[:, i], H[:, j]
        return np.array([hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1],
                         hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]])
    return np.vstack([v(0, 1), v(0, 0) - v(1, 1)])
    # END SOLUTION
```

```python
Hs = [mvg.estimate_homography_dlt(board_obj[:, :2], o) for o in observations]
for H in Hs:
    assert np.allclose(zhang_rows(H), mvg.zhang_constraints(H))
V = np.vstack([zhang_rows(H / np.linalg.norm(H)) for H in Hs])
print("singular values of V:", np.linalg.svd(V, compute_uv=False))
print("Exercise 3.3 passed")
```

### ✏️ Exercise 3.4 — the gold-standard $P$

The DLT of section 1 minimizes an algebraic error. Refine its result by minimizing the reprojection error with
`mvg.levenberg_marquardt`: parametrize $P$ by its first 11 entries with $p_{34}$ fixed, and write the residual
function (one $(u, v)$ difference per point).

```python
X30 = random_points(30, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], rng=5)
x30 = cam.project(X30) + rng.normal(scale=sigma, size=(30, 2))
P_dlt = mvg.estimate_projection_dlt(X30, x30)
P_dlt = P_dlt / P_dlt[2, 3]

def residual(p):
    # BEGIN SOLUTION
    P = np.append(p, 1.0).reshape(3, 4)
    return (mvg.project(P, X30) - x30).ravel()
    # END SOLUTION

result = mvg.levenberg_marquardt(residual, P_dlt.ravel()[:11])
P_gold = np.append(result.x, 1.0).reshape(3, 4)
```

```python
rms = lambda P: np.sqrt(np.mean(np.sum((mvg.project(P, X30) - x30) ** 2, axis=1)))
print(f"RMS reprojection error: DLT {rms(P_dlt):.4f} px  ->  refined {rms(P_gold):.4f} px")
assert rms(P_gold) <= rms(P_dlt) + 1e-12
print("Exercise 3.4 passed")
```

## 🔨 Break it

**No normalization.** Estimate a homography (eq. 3.7) from 30 noisy correspondences whose pixel coordinates are
shifted further and further from the origin, as happens with large images or cropped regions. The normalized DLT
does not notice; the unnormalized one breaks down once the offset is large compared to the spread of the points.

```python
H_ref = np.array([[1.05, 0.2, 30.0], [-0.1, 0.95, -20.0], [2e-5, -1e-5, 1.0]])
offsets = [0, 300, 1e3, 3e3, 1e4, 3e4]
med = {True: [], False: []}
for off in offsets:
    e = {True: [], False: []}
    for _ in range(50):
        xa = rng.uniform(-500, 500, size=(30, 2)) + off
        xb = mvg.apply_homography(H_ref, xa) + rng.normal(scale=sigma, size=(30, 2))
        xt = rng.uniform(-500, 500, size=(300, 2)) + off
        for norm in (True, False):
            He = mvg.estimate_homography_dlt(xa, xb, normalize=norm)
            e[norm].append(np.sqrt(np.mean(np.sum((mvg.apply_homography(He, xt)
                                                   - mvg.apply_homography(H_ref, xt)) ** 2, axis=1))))
    for norm in (True, False):
        med[norm].append(np.median(e[norm]))
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.loglog(np.array(offsets[1:]), med[True][1:], "o-", label="normalized")
ax.loglog(np.array(offsets[1:]), med[False][1:], "s-", label="unnormalized")
ax.set_xlabel("offset of the pixel coordinates [px]"); ax.set_ylabel("median held-out error [px]"); ax.legend()
print("offset 0: normalized %.3f px, unnormalized %.3f px" % (med[True][0], med[False][0]))
```

**Coplanar points.** The 3-D DLT needs points off a plane. On a plane, the matrix $A$ loses rank and $P$ is not
determined.

```python
X_plane = np.c_[rng.uniform(-0.5, 0.5, size=(30, 2)), np.zeros(30)]
x_plane = cam.project(X_plane)
for name, XX, xx in [("general position", X30, cam.project(X30)), ("coplanar", X_plane, x_plane)]:
    T, U = mvg.normalization_transform(xx), mvg.normalization_transform(XX)
    rows = []
    for (u, v, w), Xi in zip((T @ np.c_[xx, np.ones(len(xx))].T).T, (U @ np.c_[XX, np.ones(len(XX))].T).T):
        rows += [np.r_[np.zeros(4), -w * Xi, v * Xi], np.r_[w * Xi, np.zeros(4), -u * Xi]]
    print(f"{name:17} smallest singular values of A: {np.linalg.svd(np.array(rows), compute_uv=False)[-5:]}")
```

**Fronto-parallel views.** Zhang's method needs the target tilted. With the board always parallel to the image
plane, the constraints (3.10) become dependent: two singular values collapse and $K$ cannot be recovered.

```python
def zhang_singular_values(cameras):
    Hs = [mvg.estimate_homography_dlt(board_obj[:, :2], c.project(board_obj)) for c in cameras]
    return np.linalg.svd(np.vstack([mvg.zhang_constraints(H / np.linalg.norm(H)) for H in Hs]), compute_uv=False)

tilted = orbit_cameras(K_cal, 6, target=(0.2, 0.14, 0.0), distance=0.9, rng=1)
flat = orbit_cameras(K_cal, 6, target=(0.2, 0.14, 0.0), distance=0.9, tilt=(0.0, 1e-3), roll=1.5, rng=1)
print("tilted views:          ", zhang_singular_values(tilted))
print("fronto-parallel views: ", zhang_singular_values(flat))
for name, views in [("tilted", tilted), ("fronto-parallel", flat)]:
    Hs_n = [mvg.estimate_homography_dlt(board_obj[:, :2], c.project(board_obj) + rng.normal(scale=0.1, size=(54, 2)))
            for c in views]
    try:
        print(f"K from {name} views with 0.1 px noise:\n", mvg.intrinsics_from_homographies(Hs_n))
    except RuntimeError as e:
        print(f"{name}: intrinsics_from_homographies failed: {e}")
```

**Ignoring distortion.** Calibrate the same images with the distortion coefficients fixed at zero. The RMS error
grows and the residuals are no longer random: they point radially, the signature of an unmodelled lens.

```python
opt = mvg.CalibrationOptions()
opt.fix_distortion = True
res_nd = mvg.calibrate_planar(board_obj, observations, opt)
print(f"RMS with distortion model: {res.rms:.3f} px, without: {res_nd.rms:.3f} px")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, r, title in [(axes[0], res, "with distortion"), (axes[1], res_nd, "distortion fixed to 0")]:
    for v, o in enumerate(observations):
        c = mvg.PinholeCamera(r.K, 640, 480, r.R[v], r.t[v], r.distortion)
        e = o - c.project(board_obj)
        ax.quiver(*o.T, *e.T, angles="xy", scale_units="xy", scale=0.02, width=0.003)
    image_axes(ax, 640, 480, f"{title}: residuals x 50")
plt.tight_layout()
```

## What to remember

* Normalize before every DLT; it costs nothing and fixes the conditioning.
* The DLT needs non-coplanar points; Zhang's method needs tilted views of a plane.
* Linear estimates are starting points; the answer is the minimizer of the reprojection error.
* Training error is not accuracy — check on held-out data.
* Structured residuals mean a missing model term (here: distortion); random residuals at the detector noise
  level mean a good calibration.
