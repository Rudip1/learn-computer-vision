#pragma once
/// @file types.hpp
/// Common Eigen aliases used throughout the library.
///
/// Point sets are stored column-wise: a set of N image points is a 2 x N matrix, a set of N homogeneous image
/// points is 3 x N, a set of N scene points is 3 x N. This is the layout of 1_theory/00_notation.md.

#include <Eigen/Core>

namespace mvg {

using Points2 = Eigen::Matrix<double, 2, Eigen::Dynamic>;  ///< N inhomogeneous image points, one per column.
using Points3 = Eigen::Matrix<double, 3, Eigen::Dynamic>;  ///< N scene points (or homogeneous image points).
using Points4 = Eigen::Matrix<double, 4, Eigen::Dynamic>;  ///< N homogeneous scene points.
using Matrix34 = Eigen::Matrix<double, 3, 4>;              ///< Camera projection matrix P.

}  // namespace mvg
