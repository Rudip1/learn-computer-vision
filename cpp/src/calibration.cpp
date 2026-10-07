#include "mvg/calibration.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "mvg/homography.hpp"
#include "mvg/normalization.hpp"
#include "mvg/rotation.hpp"

namespace mvg {

namespace {

void check_correspondences(const Points3& X, const Points2& x, Eigen::Index min_n, const char* who) {
    if (X.cols() != x.cols()) throw std::invalid_argument(std::string(who) + ": point sets differ in size");
    if (X.cols() < min_n)
        throw std::invalid_argument(std::string(who) + ": need at least " + std::to_string(min_n) +
                                    " points");
}

}  // namespace

Matrix34 estimate_projection_dlt(const Points3& X, const Points2& x, bool normalize) {
    check_correspondences(X, x, 6, "estimate_projection_dlt");
    const Eigen::Index n = X.cols();
    const Eigen::Matrix3d T = normalize ? normalization_transform(x) : Eigen::Matrix3d::Identity();
    const Eigen::Matrix4d U = normalize ? normalization_transform(X) : Eigen::Matrix4d::Identity();
    const Points3 xh = T * x.colwise().homogeneous();
    const Points4 Xh = U * X.colwise().homogeneous();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 12);
    for (Eigen::Index i = 0; i < n; ++i) {  // eq. (3.3)
        const Eigen::RowVector4d Xi = Xh.col(i).transpose();
        const double u = xh(0, i), v = xh(1, i), w = xh(2, i);
        A.block<1, 4>(2 * i, 4) = -w * Xi;
        A.block<1, 4>(2 * i, 8) = v * Xi;
        A.block<1, 4>(2 * i + 1, 0) = w * Xi;
        A.block<1, 4>(2 * i + 1, 8) = -u * Xi;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd p = svd.matrixV().col(11);
    Matrix34 Pn;
    Pn << p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11);
    const Matrix34 P = T.inverse() * Pn * U;  // eq. (3.2)
    return P / P.norm();
}

Matrix34 estimate_projection_hall(const Points3& X, const Points2& x) {
    check_correspondences(X, x, 6, "estimate_projection_hall");
    const Eigen::Index n = X.cols();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2 * n, 11);
    Eigen::VectorXd b(2 * n);
    for (Eigen::Index i = 0; i < n; ++i) {  // eq. (3.4)
        const Eigen::RowVector3d Xi = X.col(i).transpose();
        const double u = x(0, i), v = x(1, i);
        Q.block<1, 3>(2 * i, 0) = Xi;
        Q(2 * i, 3) = 1.0;
        Q.block<1, 3>(2 * i, 8) = -u * Xi;
        Q.block<1, 3>(2 * i + 1, 4) = Xi;
        Q(2 * i + 1, 7) = 1.0;
        Q.block<1, 3>(2 * i + 1, 8) = -v * Xi;
        b(2 * i) = u;
        b(2 * i + 1) = v;
    }
    const Eigen::VectorXd a = Q.colPivHouseholderQr().solve(b);
    Matrix34 P;
    P << a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10), 1.0;
    return P;
}

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq_decomposition(const Eigen::Matrix3d& M) {
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();  // reversal permutation
    E(0, 2) = E(1, 1) = E(2, 0) = 1.0;
    Eigen::HouseholderQR<Eigen::Matrix3d> qr((E * M).transpose());  // eq. (3.5)
    const Eigen::Matrix3d Qp = qr.householderQ();
    const Eigen::Matrix3d Rp = qr.matrixQR().triangularView<Eigen::Upper>();
    Eigen::Matrix3d K = E * Rp.transpose() * E;
    Eigen::Matrix3d R = E * Qp.transpose();
    for (int i = 0; i < 3; ++i) {  // positive diagonal (section 3.4, step 3)
        if (K(i, i) < 0.0) {
            K.col(i) *= -1.0;
            R.row(i) *= -1.0;
        }
    }
    return {K, R};
}

ProjectionDecomposition decompose_projection(const Matrix34& P_in) {
    Matrix34 P = P_in;
    if (P.leftCols<3>().determinant() < 0.0) P = -P;  // step 1
    auto [K, R] = rq_decomposition(P.leftCols<3>());  // steps 2-3
    ProjectionDecomposition out;
    out.t = K.partialPivLu().solve(P.col(3));  // step 4
    out.K = K / K(2, 2);
    out.R = R;
    out.center = -R.transpose() * out.t;  // step 5
    return out;
}

Eigen::Matrix<double, 2, 6> zhang_constraints(const Eigen::Matrix3d& H) {
    const auto v = [&H](int i, int j) {  // eq. (3.9), 0-based columns
        Eigen::Matrix<double, 1, 6> r;
        r << H(0, i) * H(0, j), H(0, i) * H(1, j) + H(1, i) * H(0, j), H(1, i) * H(1, j),
            H(2, i) * H(0, j) + H(0, i) * H(2, j), H(2, i) * H(1, j) + H(1, i) * H(2, j), H(2, i) * H(2, j);
        return r;
    };
    Eigen::Matrix<double, 2, 6> V;
    V.row(0) = v(0, 1);            // h1^T B h2 = 0
    V.row(1) = v(0, 0) - v(1, 1);  // h1^T B h1 = h2^T B h2, eq. (3.10)
    return V;
}

Eigen::Matrix3d intrinsics_from_homographies(const std::vector<Eigen::Matrix3d>& Hs, bool zero_skew) {
    const Eigen::Index views = static_cast<Eigen::Index>(Hs.size());
    if (views < (zero_skew ? 2 : 3))
        throw std::invalid_argument("intrinsics_from_homographies: need >= 3 views (>= 2 with zero skew)");
    Eigen::MatrixXd V(2 * views + (zero_skew ? 1 : 0), 6);
    for (Eigen::Index k = 0; k < views; ++k) {
        // Scale each H to unit norm so that every view has the same weight.
        V.block<2, 6>(2 * k, 0) = zhang_constraints(Hs[k] / Hs[k].norm());
    }
    if (zero_skew) V.row(2 * views) << 0, 1, 0, 0, 0, 0;  // B12 = 0
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 6, 1> b = svd.matrixV().col(5);
    Eigen::Matrix3d B;
    B << b(0), b(1), b(3), b(1), b(2), b(4), b(3), b(4), b(5);
    if (B(0, 0) < 0.0) B = -B;  // the SVD sign is arbitrary
    Eigen::LLT<Eigen::Matrix3d> llt(B);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error(
            "intrinsics_from_homographies: B is not positive definite (degenerate views?)");
    const Eigen::Matrix3d U = llt.matrixU();  // B = U^T U, U ~ K^-1
    const Eigen::Matrix3d K = U.inverse();
    return K / K(2, 2);  // eq. (3.11)
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> pose_from_homography(const Eigen::Matrix3d& K,
                                                                 const Eigen::Matrix3d& H) {
    const Eigen::Matrix3d G = K.partialPivLu().solve(H);  // K^-1 H
    double lambda = 1.0 / G.col(0).norm();                // eq. (3.12)
    if (lambda * G(2, 2) < 0.0) lambda = -lambda;         // target in front: t_z > 0
    const Eigen::Vector3d r1 = lambda * G.col(0), r2 = lambda * G.col(1);
    Eigen::Matrix3d R;
    R << r1, r2, r1.cross(r2);
    return {nearest_rotation(R), lambda * G.col(2)};
}

Eigen::VectorXd reprojection_errors(const PinholeCamera& camera, const Points3& X, const Points2& x) {
    check_correspondences(X, x, 1, "reprojection_errors");
    return (camera.project(X) - x).colwise().norm().transpose();
}

namespace {

constexpr int kIntrinsicParams = 10;  // fx fy cx cy s k1 k2 p1 p2 k3

PinholeCamera camera_from_params(const Eigen::VectorXd& full, int view) {
    PinholeCamera c;
    c.K = make_intrinsics(full(0), full(1), full(2), full(3), full(4));
    c.distortion = Distortion{full(5), full(6), full(7), full(8), full(9)};
    const int o = kIntrinsicParams + 6 * view;
    c.R = so3_exp(full.segment<3>(o));
    c.t = full.segment<3>(o + 3);
    return c;
}

}  // namespace

CalibrationResult calibrate_planar(const Points3& object_points, const std::vector<Points2>& image_points,
                                   const CalibrationOptions& opt) {
    const int views = static_cast<int>(image_points.size());
    if (views < 2) throw std::invalid_argument("calibrate_planar: need at least 2 views");
    if (object_points.row(2).cwiseAbs().maxCoeff() > 1e-9)
        throw std::invalid_argument("calibrate_planar: object points must lie on the plane Z = 0");
    for (const auto& x : image_points) check_correspondences(object_points, x, 4, "calibrate_planar");
    const Points2 plane = object_points.topRows<2>();

    // Closed-form initialization, section 3.6.
    std::vector<Eigen::Matrix3d> Hs;
    for (const auto& x : image_points) Hs.push_back(estimate_homography_dlt(plane, x));
    Eigen::Matrix3d K0 = intrinsics_from_homographies(Hs, opt.fix_skew || views < 3);
    if (opt.fix_skew) K0(0, 1) = 0.0;

    Eigen::VectorXd full = Eigen::VectorXd::Zero(kIntrinsicParams + 6 * views);
    full.head<5>() << K0(0, 0), K0(1, 1), K0(0, 2), K0(1, 2), K0(0, 1);
    for (int v = 0; v < views; ++v) {
        const auto [R, t] = pose_from_homography(K0, Hs[v]);
        full.segment<3>(kIntrinsicParams + 6 * v) = so3_log(R);
        full.segment<3>(kIntrinsicParams + 6 * v + 3) = t;
    }

    const auto residuals_full = [&](const Eigen::VectorXd& p) {
        Eigen::VectorXd r(2 * object_points.cols() * views);
        for (int v = 0; v < views; ++v) {
            const Points2 e = camera_from_params(p, v).project(object_points) - image_points[v];
            r.segment(2 * object_points.cols() * v, 2 * object_points.cols()) =
                Eigen::Map<const Eigen::VectorXd>(e.data(), e.size());
        }
        return r;
    };

    CalibrationResult out;
    out.K_linear = K0;
    out.rms_linear = std::sqrt(residuals_full(full).squaredNorm() / (object_points.cols() * views));

    // Free parameters (section 3.7): everything except the fixed ones.
    std::vector<int> free;
    for (int k = 0; k < static_cast<int>(full.size()); ++k) {
        const bool fixed = (k == 4 && opt.fix_skew) || ((k == 2 || k == 3) && opt.fix_principal_point) ||
                           (k >= 5 && k <= 9 && opt.fix_distortion) ||
                           ((k == 7 || k == 8) && opt.fix_tangential) || (k == 9 && opt.fix_k3);
        if (!fixed) free.push_back(k);
    }
    const auto expand = [&](const Eigen::VectorXd& z) {
        Eigen::VectorXd p = full;
        for (size_t i = 0; i < free.size(); ++i) p(free[i]) = z(static_cast<Eigen::Index>(i));
        return p;
    };
    Eigen::VectorXd z0(free.size());
    for (size_t i = 0; i < free.size(); ++i) z0(static_cast<Eigen::Index>(i)) = full(free[i]);

    out.lm =
        levenberg_marquardt([&](const Eigen::VectorXd& z) { return residuals_full(expand(z)); }, z0, opt.lm);
    const Eigen::VectorXd p = expand(out.lm.x);

    const PinholeCamera c0 = camera_from_params(p, 0);
    out.K = c0.K;
    out.distortion = c0.distortion;
    out.per_view_rms.resize(views);
    for (int v = 0; v < views; ++v) {
        const PinholeCamera c = camera_from_params(p, v);
        out.R.push_back(c.R);
        out.t.push_back(c.t);
        out.per_view_rms(v) = std::sqrt(reprojection_errors(c, object_points, image_points[v]).squaredNorm() /
                                        static_cast<double>(object_points.cols()));
    }
    out.rms = std::sqrt(out.lm.final_cost / (object_points.cols() * views));
    return out;
}

}  // namespace mvg
