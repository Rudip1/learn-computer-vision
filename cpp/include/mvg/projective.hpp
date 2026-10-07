#pragma once
/// @file projective.hpp
/// Projective geometry of the plane: homogeneous coordinates, points and lines of P^2, the hierarchy of plane
/// transformations, closed-form fitting of the affine-or-lower classes, and the decomposition of a
/// homography.
///
/// Theory: 1_theory/01_projective_geometry.md. Equation numbers below refer to that file.

#include <Eigen/Core>
#include <string>

#include "mvg/types.hpp"

namespace mvg {

/// Append a row of ones: d x N inhomogeneous points -> (d+1) x N homogeneous points. Eq. (1.1).
Eigen::MatrixXd to_homogeneous(const Eigen::MatrixXd& points);

/// Divide by the last row: (d+1) x N homogeneous points -> d x N inhomogeneous points. Eq. (1.2).
/// Ideal points (last coordinate exactly zero) produce +-inf / nan, as the division would; check first if
/// needed.
Eigen::MatrixXd from_homogeneous(const Eigen::MatrixXd& points);

/// Skew-symmetric matrix with skew(a) * b == a.cross(b). Eq. (1.6).
Eigen::Matrix3d skew(const Eigen::Vector3d& a);

/// Line through two homogeneous points: l = x cross y. Eq. (1.4).
Eigen::Vector3d line_through(const Eigen::Vector3d& x, const Eigen::Vector3d& y);

/// Intersection of two homogeneous lines: x = l cross m. Eq. (1.5). Parallel lines give an ideal point.
Eigen::Vector3d intersect_lines(const Eigen::Vector3d& l, const Eigen::Vector3d& m);

/// Scale a line (a, b, c) so that a^2 + b^2 = 1, which makes l^T x a signed distance for x = (u, v, 1).
/// Throws std::invalid_argument for the line at infinity (a = b = 0).
Eigen::Vector3d normalize_line(const Eigen::Vector3d& l);

/// Scale a homogeneous vector to unit norm with its largest-magnitude entry positive, so that two vectors
/// that represent the same projective point compare equal entry by entry.
Eigen::VectorXd normalize_projective(const Eigen::VectorXd& x);

/// Euclidean distance from the inhomogeneous point (u, v) to the line l. Eq. (1.8).
double point_line_distance(const Eigen::Vector2d& x, const Eigen::Vector3d& l);

/// Apply a homography to inhomogeneous 2 x N points: x' ~ H x, then divide. Eq. (1.9).
Points2 apply_homography(const Eigen::Matrix3d& H, const Points2& x);

/// Image of a line under the homography H: l' ~ H^{-T} l. Eq. (1.10).
Eigen::Vector3d transform_line(const Eigen::Matrix3d& H, const Eigen::Vector3d& l);

/// The classes of the transformation hierarchy, table in section 1.4.
enum class TransformClass { Translation, Euclidean, Similarity, Affine, Projective };

/// Name of a class ("translation", "euclidean", ...).
std::string to_string(TransformClass c);

/// Smallest class of the hierarchy that contains H (section 1.4). `tol` is the tolerance on the entries of
/// the normalized matrix (H scaled to H(2,2) = 1).
TransformClass classify_transform(const Eigen::Matrix3d& H, double tol = 1e-9);

/// H for a rotation by `theta` (rad) followed by a translation (tx, ty).
Eigen::Matrix3d make_euclidean(double theta, double tx, double ty);

/// H for a scaling by s > 0, a rotation by `theta` and a translation (tx, ty).
Eigen::Matrix3d make_similarity(double s, double theta, double tx, double ty);

/// H for x' = A x + t.
Eigen::Matrix3d make_affine(const Eigen::Matrix2d& A, const Eigen::Vector2d& t);

/// Least-squares translation mapping x to xp (2 x N each): difference of centroids. Section 1.7.
Eigen::Matrix3d fit_translation(const Points2& x, const Points2& xp);

/// Least-squares rigid motion (rotation + translation) mapping x to xp, eq. (1.18). Needs N >= 2.
Eigen::Matrix3d fit_euclidean(const Points2& x, const Points2& xp);

/// Least-squares similarity mapping x to xp, linear system (1.17). Needs N >= 2.
Eigen::Matrix3d fit_similarity(const Points2& x, const Points2& xp);

/// Least-squares affine map from x to xp, section 1.7. Needs N >= 3 non-collinear points.
Eigen::Matrix3d fit_affine(const Points2& x, const Points2& xp);

/// Fit the given class (translation ... affine) by least squares. Projective fitting is estimate_homography()
/// (chapter 5); passing TransformClass::Projective throws std::invalid_argument.
Eigen::Matrix3d fit_transform(TransformClass c, const Points2& x, const Points2& xp);

/// Root-mean-square transfer error sqrt(mean ||xp_i - H(x_i)||^2) in the second image.
double rms_transfer_error(const Eigen::Matrix3d& H, const Points2& x, const Points2& xp);

/// Cross ratio of four collinear points (2 x 4, finite), eq. (1.12), using signed positions along the line.
double cross_ratio(const Points2& x4);

/// Factors of H = H_S H_A H_P, eq. (1.13).
struct ProjectiveDecomposition {
    Eigen::Matrix3d similarity;  ///< H_S = [sR t; 0 1]
    Eigen::Matrix3d affine;      ///< H_A = [U 0; 0 1], U upper triangular, det U = 1
    Eigen::Matrix3d projective;  ///< H_P = [I 0; v^T 1]
    double scale;                ///< s
    double angle;                ///< rotation angle of R (rad)
};

/// Decompose H into similarity, affine and projective factors, algorithm of section 1.5.
/// Throws std::invalid_argument if H(2,2) == 0 or the map reverses orientation.
ProjectiveDecomposition decompose_projective(const Eigen::Matrix3d& H);

/// Homography that maps the vanishing line l = (l1, l2, l3), l3 != 0, back to the line at infinity, eq.
/// (1.15).
Eigen::Matrix3d affine_rectification(const Eigen::Vector3d& vanishing_line);

}  // namespace mvg
