// Chapter 2 tests: rotations, the pinhole model, distortion and back-projection.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>

#include "mvg/camera.hpp"
#include "mvg/rotation.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mvg;

TEST_CASE("elementary rotations match Eigen's angle-axis", "[ch02]") {
    const double a = 0.7;
    CHECK(rot_x(a).isApprox(Eigen::AngleAxisd(a, Eigen::Vector3d::UnitX()).toRotationMatrix()));
    CHECK(rot_y(a).isApprox(Eigen::AngleAxisd(a, Eigen::Vector3d::UnitY()).toRotationMatrix()));
    CHECK(rot_z(a).isApprox(Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix()));
}

TEST_CASE("Euler sequences are products in the stated order", "[ch02]") {
    const Eigen::Vector3d ang(0.3, -1.1, 2.0);
    CHECK(rotation_from_euler("XYZ", ang).isApprox(rot_x(0.3) * rot_y(-1.1) * rot_z(2.0)));
    CHECK(rotation_from_euler("XYX", ang).isApprox(rot_x(0.3) * rot_y(-1.1) * rot_x(2.0)));
    CHECK(rotation_from_euler("zyx", ang).isApprox(rot_z(0.3) * rot_y(-1.1) * rot_x(2.0)));
    CHECK_THROWS(rotation_from_euler("XXY", ang));
    CHECK_THROWS(rotation_from_euler("XQ", ang));
}

TEST_CASE("so3 exp agrees with Eigen and log inverts it", "[ch02]") {
    std::mt19937 rng(2);
    for (double angle : {0.0, 1e-9, 1e-5, 0.3, 1.0, 2.5, M_PI - 1e-3, M_PI - 1e-7, M_PI}) {
        const Eigen::Vector3d n = test::random_vector<3>(rng).normalized();
        const Eigen::Vector3d w = angle * n;
        const Eigen::Matrix3d R = so3_exp(w);
        CHECK(R.isApprox(Eigen::AngleAxisd(angle, n).toRotationMatrix(), 1e-12));
        CHECK(is_rotation(R));
        const Eigen::Vector3d w2 = so3_log(R);
        CHECK((so3_exp(w2) - R).norm() < 1e-9);
        if (angle < M_PI - 1e-3) CHECK((w2 - w).norm() < 1e-9);
    }
}

TEST_CASE("nearest rotation projects a perturbed rotation back onto SO(3)", "[ch02]") {
    std::mt19937 rng(4);
    const Eigen::Matrix3d R = test::random_rotation(rng);
    Eigen::Matrix3d M = R;
    M += 1e-3 * Eigen::Matrix3d::Random();
    const Eigen::Matrix3d Rn = nearest_rotation(M);
    CHECK(is_rotation(Rn, 1e-12));
    CHECK(rotation_angle_between(R, Rn) < 5e-3);
    CHECK(nearest_rotation(R).isApprox(R, 1e-12));
    CHECK(is_rotation(nearest_rotation(-R), 1e-12));  // det(-R) = -1 is fixed by the guard
}

TEST_CASE("look-at points the optical axis at the target with image up along up", "[ch02]") {
    PinholeCamera cam;
    cam.K = make_intrinsics(500, 500, 320, 240);
    const Eigen::Vector3d C(4, -3, 2), target(0, 1, 0.5), up(0, 0, 1);
    cam.R = look_at(C, target, up);
    cam.t = -cam.R * C;
    CHECK(is_rotation(cam.R));
    CHECK(cam.center().isApprox(C));
    const Points2 px = cam.project(target);
    CHECK_THAT(px(0, 0), WithinAbs(320.0, 1e-9));
    CHECK_THAT(px(1, 0), WithinAbs(240.0, 1e-9));
    const Points2 above = cam.project(target + up);
    CHECK(above(1, 0) < 240.0);  // v points down, so "up" in the world is smaller v
    CHECK_THROWS(look_at(C, C + up, up));
}

TEST_CASE("worked example of section 2.3", "[ch02]") {
    PinholeCamera cam;
    cam.K = make_intrinsics(500, 500, 320, 240);
    const Points2 px = cam.project(Eigen::Vector3d(0.2, -0.1, 2.0));
    CHECK(px.col(0).isApprox(Eigen::Vector2d(370.0, 215.0)));
    CHECK_THAT(field_of_view(640, 320), WithinRel(M_PI / 2, 1e-12));
}

TEST_CASE("projection matrix, camera centre and the full model agree", "[ch02]") {
    std::mt19937 rng(5);
    PinholeCamera cam;
    cam.K = make_intrinsics(800, 780, 330, 250, 2.0);
    cam.R = test::random_rotation(rng);
    const Eigen::Vector3d C(1, 2, -3);
    cam.t = -cam.R * C;
    const Matrix34 P = cam.P();
    CHECK(camera_center(P).isApprox(C, 1e-10));
    CHECK((P * C.homogeneous()).norm() < 1e-9);
    const Points3 X =
        (cam.R.transpose() * test::random_points3(rng, 20, {-1, -1, 4}, {1, 1, 8})).colwise() + C;
    CHECK(project(P, X).isApprox(cam.project(X), 1e-12));
    // Third row of P gives the depth (section 2.3).
    const Points3 Xc = cam.to_camera(X);
    for (int j = 0; j < X.cols(); ++j)
        CHECK_THAT(P.row(2).dot(X.col(j).homogeneous()), WithinRel(Xc(2, j), 1e-12));
}

TEST_CASE("cheirality: points behind the camera are not visible", "[ch02]") {
    PinholeCamera cam;
    cam.K = make_intrinsics(500, 500, 320, 240);
    Points3 X(3, 2);
    X << 0.1, -0.1, 0.0, 0.0, 2.0, -2.0;
    const auto vis = cam.is_visible(X);
    CHECK(vis(0));
    CHECK_FALSE(vis(1));
    const Points2 px = cam.project(X);  // the point behind is mirrored through the principal point
    CHECK(px.col(1).isApprox(Eigen::Vector2d(345.0, 240.0)));
    CHECK(px.col(0).isApprox(Eigen::Vector2d(345.0, 240.0)));
}

TEST_CASE("radial distortion hand value and Jacobian", "[ch02]") {
    Distortion d;
    d.k1 = -0.2;
    Points2 x(2, 1);
    x << 0.5, 0.0;
    CHECK_THAT(distort(x, d)(0, 0), WithinRel(0.475, 1e-15));  // 0.5 (1 - 0.2 * 0.25)

    Distortion full{-0.28, 0.07, 1e-3, -2e-3, -0.01};
    std::mt19937 rng(8);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector2d p = test::random_vector<2>(rng, 0.3);
        const Eigen::Matrix2d J = distortion_jacobian(p, full);
        Eigen::Matrix2d Jn;
        const double h = 1e-7;
        for (int k = 0; k < 2; ++k) {
            Points2 a(2, 1), b(2, 1);
            a.col(0) = p;
            b.col(0) = p;
            a(k, 0) += h;
            b(k, 0) -= h;
            Jn.col(k) = (distort(a, full) - distort(b, full)) / (2 * h);
        }
        CHECK((J - Jn).norm() < 1e-7);
    }
}

TEST_CASE("undistort inverts distort inside the invertible region", "[ch02]") {
    const Distortion d{-0.28, 0.07, 1e-3, -2e-3, -0.01};
    std::mt19937 rng(9);
    const Points2 xn = test::random_points2(rng, 200, 0.6);
    const Points2 back = undistort(distort(xn, d), d);
    CHECK((back - xn).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("back-projection inverts projection", "[ch02]") {
    std::mt19937 rng(10);
    PinholeCamera cam;
    cam.K = make_intrinsics(600, 610, 320, 240);
    cam.distortion = {-0.2, 0.05, 0.0, 0.0, 0.0};
    cam.R = test::random_rotation(rng);
    cam.t = Eigen::Vector3d(0.3, -0.2, 1.0);
    const Points3 Xc = test::random_points3(rng, 30, {-1, -1, 3}, {1, 1, 6});
    const Points3 Xw = cam.R.transpose() * (Xc.colwise() - cam.t);
    const Points2 px = cam.project(Xw);
    const Points3 X2 = cam.backproject(px, Xc.row(2).transpose());
    CHECK((X2 - Xw).cwiseAbs().maxCoeff() < 1e-9);
    const Points3 dirs = cam.rays(px);
    for (int j = 0; j < Xw.cols(); ++j)
        CHECK(dirs.col(j).isApprox((Xw.col(j) - cam.center()).normalized(), 1e-9));
}

TEST_CASE("projection Jacobian matches finite differences, eq. (2.16)", "[ch02]") {
    const Eigen::Matrix3d K = make_intrinsics(700, 650, 300, 200, 1.5);
    const Eigen::Vector3d Xc(0.4, -0.3, 2.5);
    const Eigen::Matrix<double, 2, 3> J = projection_jacobian(K, Xc);
    const Matrix34 P = projection_matrix(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const double h = 1e-6;
    for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d a = Xc, b = Xc;
        a(k) += h;
        b(k) -= h;
        const Eigen::Vector2d col = (project(P, a) - project(P, b)) / (2 * h);
        CHECK((J.col(k) - col).norm() < 1e-5);
    }
}
