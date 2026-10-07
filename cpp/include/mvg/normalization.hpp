#pragma once
/// @file normalization.hpp
/// Hartley normalization of point sets: translate the centroid to the origin and scale the mean distance to
/// sqrt(2) (image points) or sqrt(3) (scene points).
///
/// Theory: 1_theory/03_calibration.md, section 3.2, eq. (3.1).

#include <Eigen/Core>

#include "mvg/types.hpp"

namespace mvg {

/// Similarity T (3x3) such that the points T x have centroid 0 and mean distance sqrt(2) from it. Eq. (3.1).
Eigen::Matrix3d normalization_transform(const Points2& x);

/// Similarity U (4x4) such that the points U X have centroid 0 and mean distance sqrt(3) from it.
Eigen::Matrix4d normalization_transform(const Points3& X);

}  // namespace mvg
