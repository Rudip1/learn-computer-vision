#pragma once
/// @file rotation.hpp
/// Rotations and rigid transforms: elementary rotations, Euler sequences, the exponential and logarithm maps
/// of SO(3), the nearest rotation to a matrix, look-at, and 4x4 rigid transforms.
///
/// Theory: 1_theory/02_pinhole_camera.md, section 2.1. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <string>

namespace mvg {

/// Rotation about the x, y or z axis by `angle` (rad). Eq. (2.1).
Eigen::Matrix3d rot_x(double angle);
Eigen::Matrix3d rot_y(double angle);
Eigen::Matrix3d rot_z(double angle);

/// Euler sequence: `axes` is three letters from {X, Y, Z} with no two consecutive letters equal ("XYZ",
/// "ZYX", "XYX", ...), and R = R_a1(angles[0]) * R_a2(angles[1]) * R_a3(angles[2]). Section 2.1.
Eigen::Matrix3d rotation_from_euler(const std::string& axes, const Eigen::Vector3d& angles);

/// Rodrigues' formula R = exp([w]_x), eq. (2.2). Exact for small angles (Taylor branch).
Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w);

/// Rotation vector of R (angle in [0, pi]), eq. (2.3), with the special branch near pi.
Eigen::Vector3d so3_log(const Eigen::Matrix3d& R);

/// True if R^T R = I and det R = 1 within `tol`.
bool is_rotation(const Eigen::Matrix3d& R, double tol = 1e-9);

/// Closest rotation to M in the Frobenius norm, eq. (2.4).
Eigen::Matrix3d nearest_rotation(const Eigen::Matrix3d& M);

/// World-to-camera rotation R_cw of a camera at `center` looking at `target`, image "up" along `up`. Eq.
/// (2.7). Camera axes: x right, y down, z forward.
Eigen::Matrix3d look_at(const Eigen::Vector3d& center, const Eigen::Vector3d& target,
                        const Eigen::Vector3d& up);

/// 4x4 rigid transform [R t; 0 1].
Eigen::Matrix4d make_transform(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

/// Inverse of a rigid transform: [R^T  -R^T t; 0 1].
Eigen::Matrix4d invert_transform(const Eigen::Matrix4d& T);

/// Angle (rad) of the relative rotation R_a^T R_b: the geodesic distance on SO(3).
double rotation_angle_between(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb);

}  // namespace mvg
