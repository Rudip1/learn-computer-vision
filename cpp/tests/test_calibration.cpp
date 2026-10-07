// Chapter 3 tests: normalization, DLT / Hall, RQ decomposition, Zhang's method, Levenberg-Marquardt and the
// full planar calibration, all against synthetic cameras with known parameters.

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>

#include "mvg/calibration.hpp"
#include "mvg/homography.hpp"
#include "mvg/normalization.hpp"
#include "mvg/optimization.hpp"
#include "mvg/projective.hpp"
#include "mvg/rotation.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mvg;

namespace {

PinholeCamera test_camera(std::mt19937& rng) {
    PinholeCamera cam;
    cam.K = make_intrinsics(800.0, 790.0, 330.0, 245.0);
    cam.R = test::random_rotation(rng, 0.4);
    cam.t = Eigen::Vector3d(0.1, -0.2, 3.0);
    return cam;
}

// A 9 x 6 target with 40 mm squares, centred near the origin, on Z = 0.
Points3 board() {
    Points3 X(3, 54);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 9; ++j) X.col(i * 9 + j) << 0.04 * j - 0.16, 0.04 * i - 0.10, 0.0;
    return X;
}

// Cameras looking at the board from tilted viewpoints.
std::vector<PinholeCamera> board_views(const Eigen::Matrix3d& K, const Distortion& d, int n,
                                       std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<PinholeCamera> cams;
    for (int k = 0; k < n; ++k) {
        const double az = 2.0 * M_PI * k / n, tilt = 0.35 + 0.25 * std::abs(u(rng));
        const double dist = 0.55 + 0.1 * u(rng);
        const Eigen::Vector3d C(dist * std::sin(tilt) * std::cos(az), dist * std::sin(tilt) * std::sin(az),
                                -dist * std::cos(tilt));
        PinholeCamera c;
        c.K = K;
        c.distortion = d;
        const Eigen::Vector3d target(0.02 * u(rng), 0.02 * u(rng), 0.0);
        c.R = so3_exp(Eigen::Vector3d(0, 0, 0.3 * u(rng))) * look_at(C, target, Eigen::Vector3d(0, -1, 0));
        c.t = -c.R * C;
        cams.push_back(c);
    }
    return cams;
}

}  // namespace

TEST_CASE("Hartley normalization, eq. (3.1)", "[ch03]") {
    std::mt19937 rng(1);
    const Points2 x = (test::random_points2(rng, 50, 300.0).array() + 400.0).matrix();
    const Eigen::Matrix3d T = normalization_transform(x);
    const Points2 xn = (T * x.colwise().homogeneous()).colwise().hnormalized();
    CHECK(xn.rowwise().mean().norm() < 1e-12);
    CHECK_THAT(xn.colwise().norm().mean(), WithinRel(std::sqrt(2.0), 1e-12));
    const Points3 X = test::random_points3(rng, 50, {-1, 2, 5}, {3, 4, 9});
    const Eigen::Matrix4d U = normalization_transform(X);
    const Points3 Xn = (U * X.colwise().homogeneous()).colwise().hnormalized();
    CHECK(Xn.rowwise().mean().norm() < 1e-12);
    CHECK_THAT(Xn.colwise().norm().mean(), WithinRel(std::sqrt(3.0), 1e-12));
}

TEST_CASE("DLT and Hall recover P from noise-free data", "[ch03]") {
    std::mt19937 rng(2);
    const PinholeCamera cam = test_camera(rng);
    const Points3 X = test::random_points3(rng, 20, {-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5});
    const Points2 x = cam.project(X);
    const Matrix34 P = cam.P();
    for (const Matrix34& Pe : {estimate_projection_dlt(X, x), estimate_projection_dlt(X, x, false),
                               estimate_projection_hall(X, x)}) {
        const Matrix34 a = Pe / Pe(2, 3), b = P / P(2, 3);
        CHECK(a.isApprox(b, 1e-8));
    }
}

TEST_CASE("Hall's method fails when the world origin is on the principal plane", "[ch03]") {
    std::mt19937 rng(3);
    PinholeCamera cam = test_camera(rng);
    cam.t.z() = 0.0;  // p34 = depth of the world origin = 0
    const Points3 X = test::random_points3(rng, 30, {-0.5, -0.5, 2.0}, {0.5, 0.5, 3.0});
    std::normal_distribution<double> g(0.0, 0.5);
    Points2 x = cam.project(X);
    for (int j = 0; j < x.cols(); ++j) x.col(j) += Eigen::Vector2d(g(rng), g(rng));
    PinholeCamera dlt = cam, hall = cam;
    const auto d1 = decompose_projection(estimate_projection_dlt(X, x));
    dlt.K = d1.K, dlt.R = d1.R, dlt.t = d1.t;
    const auto d2 = decompose_projection(estimate_projection_hall(X, x));
    hall.K = d2.K, hall.R = d2.R, hall.t = d2.t;
    const double e_dlt = reprojection_errors(dlt, X, x).mean();
    const double e_hall = reprojection_errors(hall, X, x).mean();
    CHECK(e_dlt < 1.0);
    CHECK(e_hall > 10.0 * e_dlt);
}

TEST_CASE("RQ decomposition and P -> K, R, t", "[ch03]") {
    std::mt19937 rng(4);
    for (int trial = 0; trial < 10; ++trial) {
        const Eigen::Matrix3d K = make_intrinsics(700 + trial, 650, 310, 250, 3.0);
        const Eigen::Matrix3d R = test::random_rotation(rng);
        const auto [Ke, Re] = rq_decomposition(K * R);
        CHECK((Ke * Re).isApprox(K * R, 1e-12));
        CHECK(Ke.isApprox(K, 1e-10));
        CHECK(Re.isApprox(R, 1e-10));
        const Eigen::Vector3d t = test::random_vector<3>(rng);
        const Matrix34 P = projection_matrix(K, R, t);
        for (double scale : {1.0, -2.5, 1e-3}) {
            const ProjectionDecomposition d = decompose_projection(scale * P);
            CHECK(d.K.isApprox(K, 1e-9));
            CHECK(d.R.isApprox(R, 1e-9));
            CHECK(d.t.isApprox(t, 1e-9));
            CHECK(d.center.isApprox(-R.transpose() * t, 1e-9));
        }
    }
}

TEST_CASE("homography DLT is exact on noise-free data", "[ch03]") {
    std::mt19937 rng(5);
    const Eigen::Matrix3d H = test::random_homography(rng);
    const Points2 x = test::random_points2(rng, 10, 200.0);
    const Points2 xp = apply_homography(H, x);
    for (bool norm : {true, false}) {
        const Eigen::Matrix3d He = estimate_homography_dlt(x, xp, norm);
        CHECK((He / He(2, 2)).isApprox(H / H(2, 2), 1e-8));
    }
    CHECK_THROWS(estimate_homography_dlt(x.leftCols(3), xp.leftCols(3)));
}

TEST_CASE("Zhang's closed form recovers K and the poses exactly", "[ch03]") {
    std::mt19937 rng(6);
    const Points3 X = board();
    for (double skew : {0.0, 2.0}) {
        const Eigen::Matrix3d K = make_intrinsics(820.0, 800.0, 315.0, 238.0, skew);
        const auto cams = board_views(K, {}, 5, rng);
        std::vector<Eigen::Matrix3d> Hs;
        for (const auto& c : cams) Hs.push_back(estimate_homography_dlt(X.topRows<2>(), c.project(X)));
        CHECK(intrinsics_from_homographies(Hs).isApprox(K, 1e-7));
        if (skew == 0.0) {
            const std::vector<Eigen::Matrix3d> two(Hs.begin(), Hs.begin() + 2);
            CHECK(intrinsics_from_homographies(two, true).isApprox(K, 1e-7));
        }
        for (size_t v = 0; v < cams.size(); ++v) {
            const auto [R, t] = pose_from_homography(K, Hs[v]);
            CHECK(R.isApprox(cams[v].R, 1e-8));
            CHECK(t.isApprox(cams[v].t, 1e-8));
        }
    }
}

TEST_CASE("Levenberg-Marquardt minimizes the Rosenbrock function", "[ch03]") {
    const ResidualFunction f = [](const Eigen::VectorXd& x) {
        Eigen::VectorXd r(2);
        r << 10.0 * (x(1) - x(0) * x(0)), 1.0 - x(0);
        return r;
    };
    const JacobianFunction J = [](const Eigen::VectorXd& x) {
        Eigen::MatrixXd j(2, 2);
        j << -20.0 * x(0), 10.0, -1.0, 0.0;
        return j;
    };
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;
    for (const auto& res : {levenberg_marquardt(f, x0), levenberg_marquardt(f, x0, {}, J)}) {
        CHECK(res.x.isApprox(Eigen::Vector2d(1.0, 1.0), 1e-8));
        CHECK(res.final_cost < 1e-20);
        CHECK(res.cost_history.front() == res.initial_cost);
        for (size_t k = 1; k < res.cost_history.size(); ++k)
            CHECK(res.cost_history[k] < res.cost_history[k - 1]);
    }
    CHECK((numeric_jacobian(f, x0) - J(x0)).norm() < 1e-6);
}

TEST_CASE("Levenberg-Marquardt agrees with linear least squares on a linear problem", "[ch03]") {
    std::mt19937 rng(7);
    const Eigen::MatrixXd A = Eigen::MatrixXd::Random(30, 4);
    const Eigen::VectorXd b = Eigen::VectorXd::Random(30);
    const auto res = levenberg_marquardt([&](const Eigen::VectorXd& x) { return Eigen::VectorXd(A * x - b); },
                                         Eigen::VectorXd::Zero(4));
    const Eigen::VectorXd ls = A.colPivHouseholderQr().solve(b);
    CHECK((res.x - ls).norm() < 1e-8);
}

TEST_CASE("planar calibration recovers intrinsics and distortion", "[ch03]") {
    std::mt19937 rng(8);
    const Points3 X = board();
    const Eigen::Matrix3d K = make_intrinsics(820.0, 805.0, 322.0, 236.0);
    const Distortion d{-0.25, 0.08, 6e-4, -4e-4, 0.0};
    const auto cams = board_views(K, d, 8, rng);

    SECTION("noise-free") {
        std::vector<Points2> obs;
        for (const auto& c : cams) obs.push_back(c.project(X));
        const CalibrationResult r = calibrate_planar(X, obs);
        CHECK(r.K.isApprox(K, 1e-6));
        CHECK((r.distortion.as_vector() - d.as_vector()).norm() < 1e-6);
        CHECK(r.rms < 1e-6);
        CHECK(r.rms_linear > r.rms);
        for (size_t v = 0; v < cams.size(); ++v) CHECK(rotation_angle_between(r.R[v], cams[v].R) < 1e-7);
    }
    SECTION("with 0.2 px noise") {
        std::normal_distribution<double> g(0.0, 0.2);
        std::vector<Points2> obs;
        for (const auto& c : cams) {
            Points2 x = c.project(X);
            for (int j = 0; j < x.cols(); ++j) x.col(j) += Eigen::Vector2d(g(rng), g(rng));
            obs.push_back(x);
        }
        const CalibrationResult r = calibrate_planar(X, obs);
        CHECK_THAT(r.K(0, 0), WithinRel(K(0, 0), 0.01));
        CHECK_THAT(r.K(1, 1), WithinRel(K(1, 1), 0.01));
        CHECK_THAT(r.K(0, 2), WithinAbs(K(0, 2), 3.0));
        CHECK_THAT(r.K(1, 2), WithinAbs(K(1, 2), 3.0));
        CHECK_THAT(r.distortion.k1, WithinAbs(d.k1, 0.02));
        // RMS of 2-D errors with sigma per coordinate is sigma * sqrt(2) (section 3.7).
        CHECK_THAT(r.rms, WithinAbs(0.2 * std::sqrt(2.0), 0.03));
    }
    SECTION("the plane must be Z = 0") {
        Points3 Y = X;
        Y(2, 0) = 0.1;
        CHECK_THROWS(calibrate_planar(Y, {cams[0].project(Y), cams[1].project(Y)}));
    }
}
