# Notation

One notation is used in every chapter, the code and the notebooks. It follows Hartley & Zisserman,
*Multiple View Geometry in Computer Vision* (2nd ed., 2004), abbreviated **HZ** below, except where a column says
otherwise.

## Conventions

| Convention | Meaning |
|---|---|
| $\mathbf{x} \simeq \mathbf{y}$ | equality up to a non-zero scale factor: $\mathbf{x} = \lambda \mathbf{y}$ for some $\lambda \neq 0$ |
| bold lower case $\mathbf{x}, \mathbf{l}$ | column vectors |
| upper case $H, P, K$ | matrices |
| $\tilde{\mathbf{x}}$ | the inhomogeneous (Euclidean) version of a homogeneous vector $\mathbf{x}$ |
| $[\mathbf{a}]_\times$ | skew-symmetric matrix with $[\mathbf{a}]_\times \mathbf{b} = \mathbf{a} \times \mathbf{b}$ |
| $\lVert \cdot \rVert$ | Euclidean norm (Frobenius norm for matrices) |
| $A^{-\top}$ | $(A^{-1})^\top = (A^\top)^{-1}$ |
| point sets | stored as columns: $N$ image points form a $2\times N$ matrix (C++ `mvg::Points2`); the Python bindings take and return $N \times 2$ arrays (rows are points) as NumPy and OpenCV do |
| angles | radians everywhere in code; degrees only in plots and text when stated |

## Symbols

| Symbol | Meaning | Chapter |
|---|---|---|
| $\mathbb{P}^2, \mathbb{P}^3$ | projective plane, projective space | 1 |
| $\mathbf{x} = (x_1, x_2, x_3)^\top$ | homogeneous image point; $\tilde{\mathbf{x}} = (x_1/x_3,\, x_2/x_3)^\top = (u, v)^\top$ | 1 |
| $\mathbf{l} = (a, b, c)^\top$ | image line $a u + b v + c = 0$ | 1 |
| $\mathbf{l}_\infty = (0,0,1)^\top$ | line at infinity of $\mathbb{P}^2$ | 1 |
| $H$ | $3\times 3$ plane projective transformation (homography), $\mathbf{x}' \simeq H \mathbf{x}$ | 1, 5 |
| $\mathbf{X} = (X_1, X_2, X_3, X_4)^\top$ | homogeneous scene point; $\tilde{\mathbf{X}} = (X, Y, Z)^\top$ | 2 |
| $R, \mathbf{t}$ | rotation and translation **world → camera**: $\tilde{\mathbf{X}}_c = R \tilde{\mathbf{X}}_w + \mathbf{t}$ | 2 |
| $\mathbf{C}$ | camera centre in world coordinates, $\tilde{\mathbf{C}} = -R^\top \mathbf{t}$ | 2 |
| $K$ | intrinsic matrix with $f_x, f_y$ (focal lengths in pixels), $s$ (skew), $(c_x, c_y)$ (principal point) | 2 |
| $P = K [R \mid \mathbf{t}]$ | $3\times 4$ camera projection matrix, $\mathbf{x} \simeq P \mathbf{X}$ | 2 |
| $(x_n, y_n)$ | normalized image coordinates, $(X_c/Z_c,\, Y_c/Z_c)$ | 2 |
| $k_1, k_2, k_3, p_1, p_2$ | radial and tangential (Brown–Conrady) distortion coefficients | 2 |
| $T_{ab} \in SE(3)$ | rigid transform mapping coordinates in frame $b$ to frame $a$: $\mathbf{p}_a = T_{ab}\, \mathbf{p}_b$ | 2, 9 |
| $\boldsymbol{\omega} \in \mathbb{R}^3$ | rotation vector (axis × angle), $R = \exp([\boldsymbol{\omega}]_\times)$ | 2, 3 |
| $T$ | similarity transform used for data normalization (Hartley normalization) | 3, 5, 6 |
| $\boldsymbol{\theta}, J, \lambda$ | parameter vector, Jacobian, Levenberg–Marquardt damping | 3 |
| $\mathbf{d}$ | feature descriptor | 4 |
| $\rho$ | Lowe ratio-test threshold | 4 |
| $\epsilon, s, p$ | RANSAC outlier fraction, sample size, success probability | 5 |
| $F, E$ | fundamental and essential matrices, $\mathbf{x}'^\top F \mathbf{x} = 0$ | 6 |
| $\mathbf{e}, \mathbf{e}'$ | epipoles in the first and second image, $F\mathbf{e} = 0$, $F^\top \mathbf{e}' = 0$ | 6 |
| $b, d, Z$ | stereo baseline, disparity, depth: $Z = f b / d$ | 7 |
| $\mathbf{c}_j$ | EPnP control points | 8 |
| ATE, RPE | absolute trajectory error, relative pose error | 9 |
| $e_k = (u_k, v_k, t_k, p_k)$ | event: pixel, timestamp, polarity $p_k \in \{-1, +1\}$ | 10 |
| $L = \log I$, $C$ | log intensity, contrast threshold | 10 |
| $\mathbf{v}$ | image-plane velocity (optical flow) in pixels per second | 10 |
