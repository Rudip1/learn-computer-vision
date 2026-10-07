// Chapter 1 tests: hand-worked examples from 1_theory/01_projective_geometry.md and closed-form invariants.

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>

#include "mvg/projective.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mvg;

TEST_CASE("homogeneous round trip", "[ch01]") {
    Points2 x(2, 3);
    x << 1, -2, 3.5, 4, 0, -7;
    const Eigen::MatrixXd h = to_homogeneous(x);
    REQUIRE(h.rows() == 3);
    CHECK(h.row(2).isOnes());
    CHECK(from_homogeneous(3.0 * h).isApprox(x));
}

TEST_CASE("worked example of section 1.2", "[ch01]") {
    const Eigen::Vector3d l = line_through({1, 2, 1}, {3, 4, 1});
    CHECK(l.isApprox(Eigen::Vector3d(-2, 2, -2)));
    const Eigen::Vector3d x = intersect_lines({1, -1, 1}, {1, 0, -2});
    CHECK(x.isApprox(Eigen::Vector3d(2, 3, 1)));
    CHECK(normalize_projective(intersect_lines(l, {1, 0, -2})).isApprox(normalize_projective(x)));
    CHECK_THAT(point_line_distance({0, 0}, l), WithinAbs(1.0 / std::sqrt(2.0), 1e-15));
}

TEST_CASE("skew matrix reproduces the cross product", "[ch01]") {
    const Eigen::Vector3d a(0.3, -1.2, 2.0), b(-0.7, 0.4, 1.1);
    CHECK((skew(a) * b).isApprox(a.cross(b)));
    CHECK((skew(a) + skew(a).transpose()).isZero());
}

TEST_CASE("parallel lines meet at an ideal point, eq. (1.7)", "[ch01]") {
    const double a = 2.0, b = -3.0, c = 1.0, cp = 5.0;
    const Eigen::Vector3d x = intersect_lines({a, b, c}, {a, b, cp});
    CHECK(x.isApprox((cp - c) * Eigen::Vector3d(b, -a, 0.0)));
    CHECK_THAT(Eigen::Vector3d(0, 0, 1).dot(x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("lines transform with the inverse transpose, eq. (1.10)", "[ch01]") {
    std::mt19937 rng(1);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Matrix3d H = test::random_homography(rng);
        const Eigen::Vector3d x = test::random_vector<3>(rng), y = test::random_vector<3>(rng);
        const Eigen::Vector3d l = line_through(x, y);
        const Eigen::Vector3d lp = transform_line(H, l);
        CHECK_THAT(lp.normalized().dot((H * x).normalized()), WithinAbs(0.0, 1e-10));
        CHECK_THAT(lp.normalized().dot((H * y).normalized()), WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("normalize_projective identifies equal points", "[ch01]") {
    const Eigen::Vector3d x(1.0, -4.0, 2.0);
    CHECK(normalize_projective(x).isApprox(normalize_projective(-3.5 * x)));
    CHECK_THROWS(normalize_projective(Eigen::Vector3d::Zero()));
}

TEST_CASE("classification of the hierarchy", "[ch01]") {
    CHECK(classify_transform(make_affine(Eigen::Matrix2d::Identity(), {3, 4})) ==
          TransformClass::Translation);
    CHECK(classify_transform(make_euclidean(0.3, 1, 2)) == TransformClass::Euclidean);
    CHECK(classify_transform(2.0 * make_euclidean(0.3, 1, 2)) == TransformClass::Euclidean);  // scale-free
    CHECK(classify_transform(make_similarity(1.7, -0.4, 1, 2)) == TransformClass::Similarity);
    Eigen::Matrix2d A;
    A << 1.0, 0.5, 0.0, 2.0;
    CHECK(classify_transform(make_affine(A, {0, 0})) == TransformClass::Affine);
    CHECK(classify_transform(make_affine(1e-6 * A, {0, 0})) ==
          TransformClass::Affine);  // tolerance is relative
    Eigen::Matrix2d reflection;
    reflection << 1.0, 0.0, 0.0, -1.0;
    CHECK(classify_transform(make_affine(reflection, {0, 0})) == TransformClass::Affine);
    Eigen::Matrix3d H = make_euclidean(0.1, 0, 0);
    H(2, 0) = 1e-3;
    CHECK(classify_transform(H) == TransformClass::Projective);
    CHECK(to_string(TransformClass::Similarity) == "similarity");
}

TEST_CASE("each class is recovered exactly from noise-free data", "[ch01]") {
    std::mt19937 rng(7);
    const Points2 x = test::random_points2(rng, 30, 100.0);
    Eigen::Matrix2d A;
    A << 1.2, 0.3, -0.2, 0.8;
    const std::pair<TransformClass, Eigen::Matrix3d> cases[] = {
        {TransformClass::Translation, make_affine(Eigen::Matrix2d::Identity(), {5, -3})},
        {TransformClass::Euclidean, make_euclidean(2.5, -10, 4)},
        {TransformClass::Similarity, make_similarity(0.6, -1.0, 3, 8)},
        {TransformClass::Affine, make_affine(A, {-2, 1})},
    };
    for (const auto& [cls, H] : cases) {
        const Points2 xp = apply_homography(H, x);
        const Eigen::Matrix3d Hhat = fit_transform(cls, x, xp);
        CHECK(Hhat.isApprox(H, 1e-10));
        CHECK(rms_transfer_error(Hhat, x, xp) < 1e-9);
        CHECK(classify_transform(Hhat, 1e-8) == cls);
    }
    CHECK_THROWS(fit_transform(TransformClass::Projective, x, x));
}

TEST_CASE("residuals shrink as the class grows", "[ch01]") {
    std::mt19937 rng(3);
    const Points2 x = test::random_points2(rng, 50, 100.0);
    const Points2 xp = apply_homography(test::random_homography(rng), x);
    double previous = std::numeric_limits<double>::infinity();
    for (auto cls : {TransformClass::Translation, TransformClass::Euclidean, TransformClass::Similarity,
                     TransformClass::Affine}) {
        // Each class contains the one before it; Euclidean and translation are not nested under
        // least squares in general, but similarity contains Euclidean and affine contains similarity.
        const double r = rms_transfer_error(fit_transform(cls, x, xp), x, xp);
        if (cls != TransformClass::Euclidean) CHECK(r <= previous + 1e-9);
        previous = std::min(previous, r);
    }
}

TEST_CASE("Procrustes never returns a reflection", "[ch01]") {
    Points2 x(2, 4), xp(2, 4);
    x << 0, 1, 0, 1, 0, 0, 1, 1;
    xp << 0, 1, 0, 1, 0, 0, -1, -1;  // a mirror image of x
    const Eigen::Matrix3d H = fit_euclidean(x, xp);
    const double det = H.topLeftCorner<2, 2>().determinant();
    CHECK_THAT(det, WithinAbs(1.0, 1e-12));
}

TEST_CASE("cross ratio is projectively invariant, eq. (1.12)", "[ch01]") {
    Points2 x(2, 4);
    // Positions 0, 1, 3, 4 along a line: cr = (3 * 3) / (4 * 2) = 9/8.
    const Eigen::Vector2d a(2, 1), d = Eigen::Vector2d(3, 4) / 5.0;
    const double mu[4] = {0, 1, 3, 4};
    for (int i = 0; i < 4; ++i) x.col(i) = a + mu[i] * d;
    CHECK_THAT(cross_ratio(x), WithinRel(9.0 / 8.0, 1e-12));
    std::mt19937 rng(11);
    for (int trial = 0; trial < 10; ++trial) {
        const Points2 xp = apply_homography(test::random_homography(rng), x);
        CHECK_THAT(cross_ratio(xp), WithinRel(9.0 / 8.0, 1e-8));
    }
}

TEST_CASE("decomposition H = H_S H_A H_P, eq. (1.13)", "[ch01]") {
    std::mt19937 rng(5);
    for (int trial = 0; trial < 20; ++trial) {
        Eigen::Matrix3d H = test::random_homography(rng);
        if ((H.topLeftCorner<2, 2>() / H(2, 2) -
             H.block<2, 1>(0, 2) * H.block<1, 2>(2, 0) / (H(2, 2) * H(2, 2)))
                .determinant() <= 0.0)
            H.row(0) *= -1.0;  // make it orientation preserving
        const ProjectiveDecomposition dec = decompose_projective(H);
        CHECK((dec.similarity * dec.affine * dec.projective).isApprox(H / H(2, 2), 1e-10));
        CHECK(classify_transform(dec.similarity, 1e-9) != TransformClass::Affine);
        CHECK_THAT(dec.affine.determinant(), WithinAbs(1.0, 1e-10));
        CHECK(dec.affine(1, 0) == 0.0);
        CHECK(dec.affine(0, 0) > 0.0);
        CHECK(dec.affine(1, 1) > 0.0);
        CHECK(dec.scale > 0.0);
    }
    const ProjectiveDecomposition sim = decompose_projective(make_similarity(2.0, 0.7, 1, 1));
    CHECK_THAT(sim.scale, WithinRel(2.0, 1e-12));
    CHECK_THAT(sim.angle, WithinRel(0.7, 1e-12));
    CHECK(sim.affine.isApprox(Eigen::Matrix3d::Identity()));
}

TEST_CASE("affine rectification restores parallelism, eq. (1.15)", "[ch01]") {
    std::mt19937 rng(9);
    const Eigen::Matrix3d H = test::random_homography(rng);
    // Two pairs of parallel world lines, imaged by H.
    const Eigen::Vector3d l1(1, 0, -1), l2(1, 0, -3), l3(0, 1, -1), l4(0, 1, 2);
    const Eigen::Vector3d v1 = intersect_lines(transform_line(H, l1), transform_line(H, l2));
    const Eigen::Vector3d v2 = intersect_lines(transform_line(H, l3), transform_line(H, l4));
    const Eigen::Vector3d vanishing = line_through(v1, v2);
    CHECK(normalize_projective(vanishing).isApprox(normalize_projective(transform_line(H, {0, 0, 1})), 1e-8));
    const Eigen::Matrix3d Ha = affine_rectification(vanishing);
    const Eigen::Matrix3d G = Ha * H;  // world -> rectified image: must be affine
    CHECK(classify_transform(G, 1e-8) == TransformClass::Affine);
}
