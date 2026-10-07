#pragma once
/// @file calibration.hpp
/// Camera calibration: DLT and Hall's method for P, RQ decomposition of P, Zhang's planar method and the
/// maximum-likelihood refinement of all parameters with Levenberg-Marquardt.
///
/// Theory: 1_theory/03_calibration.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <utility>
#include <vector>

#include "mvg/camera.hpp"
#include "mvg/optimization.hpp"
#include "mvg/types.hpp"

namespace mvg {

/// Normalized DLT for P from N >= 6 non-coplanar correspondences X_i (3 x N) <-> x_i (2 x N). Section 3.3.
/// The result is scaled to unit Frobenius norm.
Matrix34 estimate_projection_dlt(const Points3& X, const Points2& x, bool normalize = true);

/// Hall's inhomogeneous method: linear least squares with p34 = 1, eq. (3.4).
Matrix34 estimate_projection_hall(const Points3& X, const Points2& x);

/// RQ decomposition M = K R with K upper triangular (positive diagonal) and R orthogonal, eq. (3.5).
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq_decomposition(const Eigen::Matrix3d& M);

struct ProjectionDecomposition {
    Eigen::Matrix3d K;       ///< intrinsics, K(2,2) = 1
    Eigen::Matrix3d R;       ///< rotation world -> camera, det R = +1
    Eigen::Vector3d t;       ///< translation world -> camera
    Eigen::Vector3d center;  ///< camera centre in world coordinates
};

/// P -> K, R, t (algorithm of section 3.4).
ProjectionDecomposition decompose_projection(const Matrix34& P);

/// The two rows (3.10) that one plane-to-image homography contributes to Zhang's system V b = 0.
Eigen::Matrix<double, 2, 6> zhang_constraints(const Eigen::Matrix3d& H);

/// Intrinsics from >= 3 homographies (>= 2 with zero_skew), eqs. (3.8)-(3.11).
Eigen::Matrix3d intrinsics_from_homographies(const std::vector<Eigen::Matrix3d>& Hs, bool zero_skew = false);

/// Rotation and translation of a planar target (Z = 0) from its homography, eq. (3.12).
std::pair<Eigen::Matrix3d, Eigen::Vector3d> pose_from_homography(const Eigen::Matrix3d& K,
                                                                 const Eigen::Matrix3d& H);

/// Which parameters the refinement of section 3.7 keeps fixed.
struct CalibrationOptions {
    bool fix_skew = true;              ///< s = 0
    bool fix_principal_point = false;  ///< keep (cx, cy) at the linear estimate
    bool fix_k3 = true;                ///< k3 = 0
    bool fix_tangential = false;       ///< p1 = p2 = 0
    bool fix_distortion = false;       ///< all distortion coefficients = 0
    LMOptions lm;
};

struct CalibrationResult {
    Eigen::Matrix3d K;               ///< refined intrinsics
    Distortion distortion;           ///< refined distortion
    std::vector<Eigen::Matrix3d> R;  ///< per-view rotation world (target) -> camera
    std::vector<Eigen::Vector3d> t;  ///< per-view translation
    double rms = 0.0;                ///< RMS reprojection error after refinement [px]
    Eigen::VectorXd per_view_rms;    ///< RMS per view [px]
    Eigen::Matrix3d K_linear;        ///< closed-form estimate of section 3.6
    double rms_linear = 0.0;         ///< RMS of the closed-form estimate (zero distortion) [px]
    LMResult lm;                     ///< optimizer summary
};

/// Planar calibration: homographies (3.7), Zhang's closed form (3.11)-(3.12), then Levenberg-Marquardt on the
/// reprojection error (3.13). `object_points` are the target points (3 x N, Z = 0), `image_points[v]` their
/// observed pixels in view v (2 x N, same order).
CalibrationResult calibrate_planar(const Points3& object_points, const std::vector<Points2>& image_points,
                                   const CalibrationOptions& options = {});

/// Reprojection error norms ||x_i - pi(X_i)|| of a camera on correspondences (one value per point).
Eigen::VectorXd reprojection_errors(const PinholeCamera& camera, const Points3& X, const Points2& x);

}  // namespace mvg
