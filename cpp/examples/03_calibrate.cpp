// Simulates views of a planar checkerboard through a distorting lens, then calibrates: Zhang's closed form
// followed by Levenberg-Marquardt on the reprojection error (chapter 3).
//
//   ./03_calibrate [noise_px] [views]      defaults: 0.2 px, 10 views

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "mvg/calibration.hpp"
#include "mvg/rotation.hpp"

int main(int argc, char** argv) {
    const double noise = argc > 1 ? std::atof(argv[1]) : 0.2;
    const int views = argc > 2 ? std::atoi(argv[2]) : 10;

    // 9 x 6 inner corners, 40 mm squares, on the plane Z = 0.
    mvg::Points3 board(3, 54);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 9; ++j) board.col(i * 9 + j) << 0.04 * j, 0.04 * i, 0.0;
    const Eigen::Vector3d centre(0.16, 0.10, 0.0);

    const Eigen::Matrix3d K = mvg::make_intrinsics(820.0, 805.0, 322.0, 236.0);
    const mvg::Distortion dist{-0.25, 0.08, 6e-4, -4e-4, 0.0};
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::normal_distribution<double> g(0.0, noise);

    std::vector<mvg::Points2> observations;
    for (int k = 0; k < views; ++k) {
        const double az = 2.0 * M_PI * k / views, tilt = 0.35 + 0.2 * std::abs(u(rng)), d = 0.6;
        const Eigen::Vector3d C =
            centre + d * Eigen::Vector3d(std::sin(tilt) * std::cos(az), std::sin(tilt) * std::sin(az),
                                         -std::cos(tilt));
        mvg::PinholeCamera cam;
        cam.K = K;
        cam.distortion = dist;
        cam.R = mvg::so3_exp(Eigen::Vector3d(0, 0, 0.3 * u(rng))) * mvg::look_at(C, centre, {0, -1, 0});
        cam.t = -cam.R * C;
        mvg::Points2 x = cam.project(board);
        for (int j = 0; j < x.cols(); ++j) x.col(j) += Eigen::Vector2d(g(rng), g(rng));
        observations.push_back(x);
    }

    const mvg::CalibrationResult r = mvg::calibrate_planar(board, observations);
    std::printf("%d views, %.2f px noise\n", views, noise);
    std::printf("%-14s %9s %9s %9s %9s %9s %9s   rms [px]\n", "", "fx", "fy", "cx", "cy", "k1", "k2");
    std::printf("%-14s %9.2f %9.2f %9.2f %9.2f %9.4f %9.4f\n", "true", K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                dist.k1, dist.k2);
    std::printf("%-14s %9.2f %9.2f %9.2f %9.2f %9s %9s   %.3f\n", "closed form", r.K_linear(0, 0),
                r.K_linear(1, 1), r.K_linear(0, 2), r.K_linear(1, 2), "-", "-", r.rms_linear);
    std::printf("%-14s %9.2f %9.2f %9.2f %9.2f %9.4f %9.4f   %.3f  (%d LM iterations)\n", "refined",
                r.K(0, 0), r.K(1, 1), r.K(0, 2), r.K(1, 2), r.distortion.k1, r.distortion.k2, r.rms,
                r.lm.iterations);
    return 0;
}
