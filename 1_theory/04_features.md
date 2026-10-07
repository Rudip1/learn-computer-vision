# 4 · Features and matching

Every multi-view algorithm in the following chapters starts from **correspondences**: pairs of pixels, one in
each image, that are projections of the same scene point. This chapter explains how they are found: detect
distinctive points (features), describe their neighbourhood by a vector (descriptor), and pair descriptors across
images (matching), throwing away pairs that are likely wrong.

Notation: [00_notation.md](00_notation.md). Code: [`cpp/include/mvg/features.hpp`](../cpp/include/mvg/features.hpp),
[`cpp/include/mvg/image.hpp`](../cpp/include/mvg/image.hpp).
Notebook: [`2_notebooks/exercises/04_features.ipynb`](../2_notebooks/exercises/04_features.ipynb).

## 4.1 What makes a good feature

A feature is useful if it is

* **repeatable** — found again at the same scene point in another image, despite changes of viewpoint, scale,
  illumination and noise;
* **distinctive** — its neighbourhood looks different from the neighbourhoods of other features, so that it can be
  matched unambiguously;
* **local** — small enough to be unaffected by occlusion and by the non-planarity of the scene;
* **accurately localized** — its position is defined to a fraction of a pixel.

A point in a uniform region fails distinctiveness in every direction; a point on a straight edge can slide along
the edge (the *aperture problem*); a **corner** is pinned in both directions. Blobs (bright or dark regions of a
characteristic size) are the other classic choice, because their size gives a scale.

## 4.2 Corners: the structure tensor

Shift a window $w$ around a pixel by $\boldsymbol{\Delta} = (\Delta u, \Delta v)$ and measure how much the image
changes:

$$
E(\boldsymbol{\Delta}) = \sum_{\mathbf{p}} w(\mathbf{p})\,\left[I(\mathbf{p} + \boldsymbol{\Delta}) - I(\mathbf{p})\right]^2
\approx \boldsymbol{\Delta}^\top M\, \boldsymbol{\Delta}, \tag{4.1}
$$

using the first-order Taylor expansion $I(\mathbf{p} + \boldsymbol{\Delta}) \approx I(\mathbf{p}) + \nabla I^\top\boldsymbol{\Delta}$.
The $2\times 2$ **structure tensor** (second-moment matrix) is

$$
M = \sum_{\mathbf{p}} w(\mathbf{p}) \begin{pmatrix} I_u^2 & I_u I_v \\ I_u I_v & I_v^2 \end{pmatrix} , \tag{4.2}
$$

with $I_u = \partial I/\partial u$, $I_v = \partial I/\partial v$ (Sobel derivatives here) and a Gaussian window $w$ of
standard deviation $\sigma_w$. Its eigenvalues $\lambda_1 \ge \lambda_2 \ge 0$ are the largest and smallest change
over all shift directions:

| $\lambda_1$ | $\lambda_2$ | region |
|---|---|---|
| small | small | flat |
| large | small | edge (the change is zero along the edge) |
| large | large | corner |

**Shi–Tomasi** score: $\lambda_2 = \min(\lambda_1, \lambda_2)$. **Harris** avoids the eigen-decomposition with

$$
R = \det M - k\,(\operatorname{tr} M)^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2,\qquad k \in [0.04, 0.06] , \tag{4.3}
$$

positive at corners, negative at edges, small in flat regions.

**Algorithm** (`detect_corners`):

1. Smooth the image slightly ($\sigma = 1$) and compute $I_u, I_v$.
2. Form the three images $I_u^2$, $I_uI_v$, $I_v^2$ and smooth each with the window, $\sigma_w = 1.5$: this is (4.2)
   at every pixel.
3. Compute the score (4.3) or the smaller eigenvalue at every pixel.
4. **Non-maximum suppression**: keep pixels that are the maximum of the score in a $(2r+1)^2$ neighbourhood and
   exceed a fraction (e.g. 1 %) of the largest score in the image.
5. Sort by score and keep the best $N$.
6. **Sub-pixel refinement**: fit a parabola to the score in $u$ and in $v$ through the maximum and its two
   neighbours; the vertex of $s(-1), s(0), s(1)$ is at offset $\frac{1}{2}\frac{s(-1) - s(1)}{s(-1) - 2s(0) + s(1)}$.

Harris corners are invariant to rotation (eigenvalues do not change when the image rotates) and to additive
brightness changes (only derivatives enter), but **not to scale**: a corner seen from twice as far may fit inside
the window and look like a point, or a rounded corner seen closely may look like an edge.

## 4.3 Scale (overview)

Scale invariance needs a scale selection rule. Lindeberg's principle: search for extrema of a scale-normalized
derivative operator over position *and* scale. The Laplacian of Gaussian $\sigma^2\nabla^2 G_\sigma * I$ responds
maximally to a blob of radius $\sqrt{2}\sigma$; SIFT approximates it by the **difference of Gaussians**
$G_{k\sigma} * I - G_\sigma * I$ computed on an image pyramid, keeps extrema in $3\times 3\times 3$ neighbourhoods of
$(u, v, \sigma)$, and rejects edge responses with a Harris-like ratio test on the Hessian (Lowe 2004). FAST (Rosten
& Drummond 2006) tests a circle of 16 pixels for a contiguous arc brighter or darker than the centre and is used
with a pyramid in ORB. These detectors are not re-implemented in this module; the notebook uses OpenCV's SIFT and
ORB as references.

## 4.4 Descriptors

A descriptor turns the neighbourhood of a feature into a vector $\mathbf{d}$ such that the neighbourhoods of the
same scene point in two images give nearby vectors.

**Normalized patch.** Take the $(2r+1)^2$ intensities around the feature, subtract their mean and divide by their
norm. Then $\lVert\mathbf{d}_1 - \mathbf{d}_2\rVert^2 = 2(1 - \text{NCC})$, where NCC is the normalized
cross-correlation of the two patches: the descriptor is invariant to affine brightness changes $I \mapsto aI + b$,
$a > 0$, but not to rotation or scale.

**SIFT** (overview). A $16\times 16$ neighbourhood, rotated to the dominant gradient orientation and scaled by the
detected scale, is divided into $4\times 4$ cells; each cell contributes an 8-bin histogram of gradient
orientations weighted by magnitude: 128 numbers, normalized, clipped at 0.2 and renormalized.

**BRIEF / ORB** (binary). Compare the smoothed intensity at $n_b$ pre-defined pairs of positions
$(\mathbf{a}_k, \mathbf{b}_k)$ inside the patch:

$$
d_k = \begin{cases} 1 & I_\sigma(\mathbf{p} + \mathbf{a}_k) < I_\sigma(\mathbf{p} + \mathbf{b}_k) \\ 0 & \text{otherwise} \end{cases},
\qquad k = 1 \dots n_b . \tag{4.4}
$$

With $n_b = 256$ the descriptor is 32 bytes and two descriptors are compared with the **Hamming distance** (number
of differing bits), one XOR and a population count per 64 bits. BRIEF is not rotation invariant. ORB makes it so
by measuring an orientation with the **intensity centroid** (Rosin 1999): with the patch moments
$m_{pq} = \sum u^p v^q I(u, v)$ taken relative to the feature,

$$
\theta = \operatorname{atan2}(m_{01}, m_{10}) , \tag{4.5}
$$

and rotating the sampling pattern by $\theta$ before evaluating (4.4) ("steered BRIEF").

## 4.5 Matching

Given descriptors $\{\mathbf{d}_i\}$ in image 1 and $\{\mathbf{d}'_j\}$ in image 2, the **nearest neighbour** of
$\mathbf{d}_i$ is $j^\ast = \arg\min_j \operatorname{dist}(\mathbf{d}_i, \mathbf{d}'_j)$ (Euclidean distance for
float descriptors, Hamming for binary ones). Every feature has a nearest neighbour, including features whose true
match was not detected or is occluded — so nearest-neighbour matching alone produces many wrong pairs. Three filters
reduce them.

**Absolute threshold.** Reject if $\operatorname{dist}_1 > \tau$. Simple, but a single $\tau$ does not fit all
features: some descriptors are much more distinctive than others.

**Ratio test** (Lowe 2004). Compare the distance to the nearest neighbour with the distance to the second nearest:

$$
\frac{\operatorname{dist}_1}{\operatorname{dist}_2} < \rho . \tag{4.6}
$$

For a correct match the nearest neighbour is much closer than any other; for a wrong match the nearest is just
one of many similar-looking descriptors and the ratio is close to 1. Lowe measured that $\rho = 0.8$ removes
90 % of the wrong matches while losing fewer than 5 % of the correct ones. Small $\rho$ keeps few, reliable matches;
large $\rho$ many, less reliable ones.

**Cross-check** (mutual nearest neighbours). Keep $(i, j)$ only if $j$ is the nearest neighbour of $i$ in image 2
*and* $i$ is the nearest neighbour of $j$ in image 1.

**Evaluating a matcher.** With a known ground-truth mapping (a homography in the notebook), a match is correct if
the transferred point lies within a few pixels of its partner. Report the number of matches and the **precision**
(fraction correct); plotting both against $\rho$ shows the trade-off. No setting of $\rho$ guarantees zero wrong
matches with a useful number of correct ones — the remaining outliers are removed by geometric verification with
RANSAC (chapter 5).

**Algorithm** (`match_descriptors`):

1. For each $\mathbf{d}_i$ find the nearest and second-nearest $\mathbf{d}'_j$ (brute force here; kd-trees or
   approximate methods such as FLANN for large sets).
2. Discard $i$ if $\operatorname{dist}_1 > \tau$ or (4.6) fails.
3. If cross-checking, find the nearest neighbour of each $\mathbf{d}'_j$ in image 1 and discard $(i, j)$ unless
   they are mutual.
4. Return the surviving pairs with their distances.

## 4.6 Common mistakes

* Matching without any filtering and passing the result straight to a least-squares estimator.
* Using Euclidean distance on binary descriptors, or Hamming distance on float ones.
* Expecting rotation or scale invariance from a detector/descriptor that has none (Harris + patch, plain BRIEF).
* Computing gradients on an unsmoothed image: noise dominates the structure tensor.
* Thresholding Harris scores with an absolute value: the score scales with the fourth power of the contrast. Use a
  fraction of the maximum, or keep the top $N$.

## References

* C. Harris, M. Stephens. "A combined corner and edge detector." *Alvey Vision Conference*, 1988.
* J. Shi, C. Tomasi. "Good features to track." *CVPR*, 1994.
* D. G. Lowe. "Distinctive image features from scale-invariant keypoints." *IJCV* 60(2), 2004.
* M. Calonder, V. Lepetit, C. Strecha, P. Fua. "BRIEF: Binary robust independent elementary features." *ECCV*, 2010.
* E. Rublee, V. Rabaud, K. Konolige, G. Bradski. "ORB: An efficient alternative to SIFT or SURF." *ICCV*, 2011.
* P. L. Rosin. "Measuring corner properties." *CVIU* 73(2), 1999.
* T. Lindeberg. "Feature detection with automatic scale selection." *IJCV* 30(2), 1998.
* R. Szeliski. *Computer Vision: Algorithms and Applications*, 2nd ed., 2022. Chapter 7.
