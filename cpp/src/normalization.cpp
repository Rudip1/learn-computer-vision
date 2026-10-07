#include "mvg/normalization.hpp"

#include <cmath>
#include <stdexcept>

namespace mvg {

namespace {

template <int D>
Eigen::Matrix<double, D + 1, D + 1> normalization(const Eigen::Matrix<double, D, Eigen::Dynamic>& x) {
    if (x.cols() < 2) throw std::invalid_argument("normalization_transform: need at least 2 points");
    const Eigen::Matrix<double, D, 1> c = x.rowwise().mean();
    const double mean_dist = (x.colwise() - c).colwise().norm().mean();
    if (mean_dist <= 0.0) throw std::invalid_argument("normalization_transform: all points coincide");
    const double s = std::sqrt(static_cast<double>(D)) / mean_dist;  // eq. (3.1)
    Eigen::Matrix<double, D + 1, D + 1> T = Eigen::Matrix<double, D + 1, D + 1>::Identity();
    T.template topLeftCorner<D, D>() *= s;
    T.template topRightCorner<D, 1>() = -s * c;
    return T;
}

}  // namespace

Eigen::Matrix3d normalization_transform(const Points2& x) { return normalization<2>(x); }

Eigen::Matrix4d normalization_transform(const Points3& X) { return normalization<3>(X); }

}  // namespace mvg
