#pragma once
/// @file homography.hpp
/// Estimation of plane projective transformations x' ~ H x.
///
/// Theory: the DLT is derived in 1_theory/03_calibration.md, section 3.5, eq. (3.7); robust estimation,
/// refinement and image warping are in 1_theory/05_homographies.md.

#include <Eigen/Core>

#include "mvg/types.hpp"

namespace mvg {

/// Direct linear transform for H from N >= 4 correspondences x_i <-> xp_i (2 x N each), eq. (3.7).
/// With `normalize` both sets are first normalized with eq. (3.1) and the result denormalized with eq. (3.2).
/// The returned H is scaled to unit Frobenius norm.
Eigen::Matrix3d estimate_homography_dlt(const Points2& x, const Points2& xp, bool normalize = true);

}  // namespace mvg
