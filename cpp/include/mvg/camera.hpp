#pragma once
/// @file camera.hpp
/// The pinhole camera with Brown-Conrady distortion: intrinsics, extrinsics, projection, back-projection,
/// distortion and its Newton inverse, and the Jacobian of the projection.
///
/// Theory: 1_theory/02_pinhole_camera.md. Equation numbers below refer to that file.

#include <Eigen/Core>

#include "mvg/types.hpp"

namespace mvg {

/// Brown-Conrady coefficients in OpenCV order (k1, k2, p1, p2, k3). Eq. (2.12).
struct Distortion {
    double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0;

    /// (k1, k2, p1, p2, k3) as a vector.
    Eigen::Matrix<double, 5, 1> as_vector() const;
    static Distortion from_vector(const Eigen::Matrix<double, 5, 1>& v);
    bool is_zero() const { return k1 == 0.0 && k2 == 0.0 && p1 == 0.0 && p2 == 0.0 && k3 == 0.0; }
};

/// K = [fx s cx; 0 fy cy; 0 0 1], eq. (2.9).
Eigen::Matrix3d make_intrinsics(double fx, double fy, double cx, double cy, double skew = 0.0);

/// P = K [R | t], eq. (2.10).
Matrix34 projection_matrix(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

/// Camera centre (inhomogeneous) as the right null vector of P (section 2.3).
Eigen::Vector3d camera_center(const Matrix34& P);

/// Project 3 x N world points with a projection matrix: x ~ P X, then divide. Eq. (2.10).
Points2 project(const Matrix34& P, const Points3& X);

/// Apply the distortion model to normalized coordinates (2 x N). Eq. (2.12).
Points2 distort(const Points2& xn, const Distortion& d);

/// Jacobian of the distortion map at one normalized point. Eq. (2.14).
Eigen::Matrix2d distortion_jacobian(const Eigen::Vector2d& xn, const Distortion& d);

/// Invert the distortion with Newton's method (2.13), starting from the distorted point.
Points2 undistort(const Points2& xd, const Distortion& d, int max_iterations = 20, double tol = 1e-14);

/// Pixels -> normalized (distorted) coordinates: K^-1 applied to each point.
Points2 pixels_to_normalized(const Eigen::Matrix3d& K, const Points2& px);

/// Normalized coordinates -> pixels: K applied to each point.
Points2 normalized_to_pixels(const Eigen::Matrix3d& K, const Points2& xn);

/// d(u, v) / d(X_c) for a camera-frame point, no distortion. Eq. (2.16).
Eigen::Matrix<double, 2, 3> projection_jacobian(const Eigen::Matrix3d& K, const Eigen::Vector3d& Xc);

/// Field of view (rad) covered by `size_px` pixels with focal length `focal_px` pixels. Eq. (2.11).
double field_of_view(double size_px, double focal_px);

/// A complete camera: intrinsics, distortion, extrinsics (world -> camera) and image size.
struct PinholeCamera {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Distortion distortion;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  ///< R_cw, eq. (2.5)
    Eigen::Vector3d t = Eigen::Vector3d::Zero();      ///< t_cw, eq. (2.5)
    int width = 640;
    int height = 480;

    /// Camera-frame coordinates of world points, eq. (2.5).
    Points3 to_camera(const Points3& Xw) const;

    /// Full projection (algorithm of section 2.7, steps 1-4). Points behind the camera are projected too; use
    /// is_visible() to filter.
    Points2 project(const Points3& Xw) const;

    /// Z_c > 0 and the projection falls inside [0, width) x [0, height). Section 2.7, steps 1 and 5.
    Eigen::Array<bool, Eigen::Dynamic, 1> is_visible(const Points3& Xw) const;

    /// Unit viewing directions (world frame) of pixels, eq. (2.15).
    Points3 rays(const Points2& px) const;

    /// World points of pixels with known camera-frame depth Z_c (one per pixel), section 2.5.
    Points3 backproject(const Points2& px, const Eigen::VectorXd& depth) const;

    Eigen::Vector3d center() const;  ///< eq. (2.6)
    Matrix34 P() const;              ///< eq. (2.10), ignores distortion
    Eigen::Matrix4d pose() const;    ///< T_wc, the camera pose in the world
};

}  // namespace mvg
