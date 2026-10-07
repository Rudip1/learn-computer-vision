#pragma once
// Random test data shared by the test files. Deterministic: every generator takes the RNG explicitly.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <random>

#include "mvg/types.hpp"

namespace mvg::test {

template <int N>
Eigen::Matrix<double, N, 1> random_vector(std::mt19937& rng, double scale = 1.0) {
    std::normal_distribution<double> g(0.0, scale);
    Eigen::Matrix<double, N, 1> v;
    for (int i = 0; i < N; ++i) v(i) = g(rng);
    return v;
}

inline Points2 random_points2(std::mt19937& rng, int n, double half_width) {
    std::uniform_real_distribution<double> u(-half_width, half_width);
    Points2 x(2, n);
    for (int j = 0; j < n; ++j) x.col(j) << u(rng), u(rng);
    return x;
}

inline Points3 random_points3(std::mt19937& rng, int n, const Eigen::Vector3d& lo,
                              const Eigen::Vector3d& hi) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Points3 X(3, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < 3; ++i) X(i, j) = lo(i) + (hi(i) - lo(i)) * u(rng);
    return X;
}

/// A well-conditioned homography: similarity + mild affine + mild perspective, applied to coordinates of
/// order 100.
inline Eigen::Matrix3d random_homography(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::Matrix3d H;
    H << 1.0 + 0.3 * u(rng), 0.3 * u(rng), 20.0 * u(rng),  //
        0.3 * u(rng), 1.0 + 0.3 * u(rng), 20.0 * u(rng),   //
        1e-3 * u(rng), 1e-3 * u(rng), 1.0;
    return H * (1.0 + std::abs(u(rng)));
}

inline Eigen::Matrix3d random_rotation(std::mt19937& rng, double max_angle = M_PI) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const Eigen::Vector3d axis = random_vector<3>(rng).normalized();
    return Eigen::AngleAxisd(max_angle * u(rng), axis).toRotationMatrix();
}

}  // namespace mvg::test
