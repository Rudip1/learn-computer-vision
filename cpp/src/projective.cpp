#include "mvg/projective.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace mvg {

Eigen::MatrixXd to_homogeneous(const Eigen::MatrixXd& points) {
    Eigen::MatrixXd h(points.rows() + 1, points.cols());
    h.topRows(points.rows()) = points;
    h.bottomRows(1).setOnes();
    return h;
}

Eigen::MatrixXd from_homogeneous(const Eigen::MatrixXd& points) {
    if (points.rows() < 2) throw std::invalid_argument("from_homogeneous: need at least 2 rows");
    const Eigen::Index d = points.rows() - 1;
    Eigen::MatrixXd x(d, points.cols());
    for (Eigen::Index j = 0; j < points.cols(); ++j) x.col(j) = points.col(j).head(d) / points(d, j);
    return x;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& a) {
    Eigen::Matrix3d S;
    S << 0.0, -a.z(), a.y(),  //
        a.z(), 0.0, -a.x(),   //
        -a.y(), a.x(), 0.0;
    return S;
}

Eigen::Vector3d line_through(const Eigen::Vector3d& x, const Eigen::Vector3d& y) { return x.cross(y); }

Eigen::Vector3d intersect_lines(const Eigen::Vector3d& l, const Eigen::Vector3d& m) { return l.cross(m); }

Eigen::Vector3d normalize_line(const Eigen::Vector3d& l) {
    const double n = l.head<2>().norm();
    if (n == 0.0) throw std::invalid_argument("normalize_line: the line at infinity has no normal direction");
    return l / n;
}

Eigen::VectorXd normalize_projective(const Eigen::VectorXd& x) {
    const double n = x.norm();
    if (n == 0.0) throw std::invalid_argument("normalize_projective: zero vector is not a projective point");
    Eigen::Index imax = 0;
    x.cwiseAbs().maxCoeff(&imax);
    return (x(imax) < 0.0 ? -1.0 : 1.0) * x / n;
}

double point_line_distance(const Eigen::Vector2d& x, const Eigen::Vector3d& l) {
    return std::abs(normalize_line(l).dot(x.homogeneous()));
}

Points2 apply_homography(const Eigen::Matrix3d& H, const Points2& x) {
    return (H * x.colwise().homogeneous()).colwise().hnormalized();
}

Eigen::Vector3d transform_line(const Eigen::Matrix3d& H, const Eigen::Vector3d& l) {
    // eq. (1.10): solve H^T l' = l instead of forming the inverse explicitly.
    return H.transpose().partialPivLu().solve(l);
}

std::string to_string(TransformClass c) {
    switch (c) {
        case TransformClass::Translation:
            return "translation";
        case TransformClass::Euclidean:
            return "euclidean";
        case TransformClass::Similarity:
            return "similarity";
        case TransformClass::Affine:
            return "affine";
        case TransformClass::Projective:
            return "projective";
    }
    return "unknown";
}

TransformClass classify_transform(const Eigen::Matrix3d& H, double tol) {
    if (std::abs(H(2, 2)) <= tol * H.norm()) return TransformClass::Projective;
    const Eigen::Matrix3d Hn = H / H(2, 2);
    if (std::abs(Hn(2, 0)) > tol || std::abs(Hn(2, 1)) > tol) return TransformClass::Projective;
    const Eigen::Matrix2d A = Hn.topLeftCorner<2, 2>();
    if ((A - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff() <= tol) return TransformClass::Translation;
    // A^T A = s^2 I and det A > 0  <=>  A = s R(theta).
    const Eigen::Matrix2d AtA = A.transpose() * A;
    const double s2 = 0.5 * AtA.trace();
    const bool conformal = std::abs(AtA(0, 1)) <= tol * s2 && std::abs(AtA(0, 0) - AtA(1, 1)) <= tol * s2 &&
                           A.determinant() > 0.0;
    if (!conformal) return TransformClass::Affine;
    if (std::abs(s2 - 1.0) <= tol) return TransformClass::Euclidean;
    return TransformClass::Similarity;
}

Eigen::Matrix3d make_euclidean(double theta, double tx, double ty) {
    return make_similarity(1.0, theta, tx, ty);
}

Eigen::Matrix3d make_similarity(double s, double theta, double tx, double ty) {
    if (s <= 0.0) throw std::invalid_argument("make_similarity: scale must be positive");
    const double c = std::cos(theta), sn = std::sin(theta);
    Eigen::Matrix3d H;
    H << s * c, -s * sn, tx,  //
        s * sn, s * c, ty,    //
        0.0, 0.0, 1.0;
    return H;
}

Eigen::Matrix3d make_affine(const Eigen::Matrix2d& A, const Eigen::Vector2d& t) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    H.topLeftCorner<2, 2>() = A;
    H.topRightCorner<2, 1>() = t;
    return H;
}

namespace {

void check_pairs(const Points2& x, const Points2& xp, Eigen::Index min_n, const char* who) {
    if (x.cols() != xp.cols()) throw std::invalid_argument(std::string(who) + ": point sets differ in size");
    if (x.cols() < min_n)
        throw std::invalid_argument(std::string(who) + ": need at least " + std::to_string(min_n) +
                                    " points");
}

}  // namespace

Eigen::Matrix3d fit_translation(const Points2& x, const Points2& xp) {
    check_pairs(x, xp, 1, "fit_translation");
    const Eigen::Vector2d t = xp.rowwise().mean() - x.rowwise().mean();
    return make_affine(Eigen::Matrix2d::Identity(), t);
}

Eigen::Matrix3d fit_euclidean(const Points2& x, const Points2& xp) {
    check_pairs(x, xp, 2, "fit_euclidean");
    const Eigen::Vector2d mx = x.rowwise().mean(), mxp = xp.rowwise().mean();
    // eq. (1.18): cross-covariance of the centred sets, then SVD with the reflection guard.
    const Eigen::Matrix2d Sigma = (xp.colwise() - mxp) * (x.colwise() - mx).transpose();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d D = Eigen::Matrix2d::Identity();
    D(1, 1) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
    const Eigen::Matrix2d R = svd.matrixU() * D * svd.matrixV().transpose();
    return make_affine(R, mxp - R * mx);
}

Eigen::Matrix3d fit_similarity(const Points2& x, const Points2& xp) {
    check_pairs(x, xp, 2, "fit_similarity");
    const Eigen::Index n = x.cols();
    Eigen::MatrixXd M(2 * n, 4);
    Eigen::VectorXd b(2 * n);
    for (Eigen::Index i = 0; i < n; ++i) {  // eq. (1.17)
        M.row(2 * i) << x(0, i), -x(1, i), 1.0, 0.0;
        M.row(2 * i + 1) << x(1, i), x(0, i), 0.0, 1.0;
        b(2 * i) = xp(0, i);
        b(2 * i + 1) = xp(1, i);
    }
    const Eigen::Vector4d p = M.colPivHouseholderQr().solve(b);
    Eigen::Matrix2d A;
    A << p(0), -p(1), p(1), p(0);
    return make_affine(A, p.tail<2>());
}

Eigen::Matrix3d fit_affine(const Points2& x, const Points2& xp) {
    check_pairs(x, xp, 3, "fit_affine");
    const Eigen::Index n = x.cols();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(2 * n, 6);
    Eigen::VectorXd b(2 * n);
    for (Eigen::Index i = 0; i < n; ++i) {
        M.row(2 * i).head<3>() << x(0, i), x(1, i), 1.0;
        M.row(2 * i + 1).tail<3>() << x(0, i), x(1, i), 1.0;
        b(2 * i) = xp(0, i);
        b(2 * i + 1) = xp(1, i);
    }
    const Eigen::Matrix<double, 6, 1> p = M.colPivHouseholderQr().solve(b);
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    H.row(0) = p.head<3>().transpose();
    H.row(1) = p.tail<3>().transpose();
    return H;
}

Eigen::Matrix3d fit_transform(TransformClass c, const Points2& x, const Points2& xp) {
    switch (c) {
        case TransformClass::Translation:
            return fit_translation(x, xp);
        case TransformClass::Euclidean:
            return fit_euclidean(x, xp);
        case TransformClass::Similarity:
            return fit_similarity(x, xp);
        case TransformClass::Affine:
            return fit_affine(x, xp);
        case TransformClass::Projective:
            break;
    }
    throw std::invalid_argument("fit_transform: use estimate_homography() for the projective class");
}

double rms_transfer_error(const Eigen::Matrix3d& H, const Points2& x, const Points2& xp) {
    check_pairs(x, xp, 1, "rms_transfer_error");
    return std::sqrt((apply_homography(H, x) - xp).colwise().squaredNorm().mean());
}

double cross_ratio(const Points2& x4) {
    if (x4.cols() != 4) throw std::invalid_argument("cross_ratio: need exactly 4 points");
    // Signed positions mu_i along the line through x_1 with direction x_4 - x_1 (any direction works).
    const Eigen::Vector2d dir = (x4.col(3) - x4.col(0)).normalized();
    Eigen::Vector4d mu;
    for (int i = 0; i < 4; ++i) mu(i) = dir.dot(x4.col(i) - x4.col(0));
    const auto d = [&](int i, int j) { return mu(j) - mu(i); };
    return (d(0, 2) * d(1, 3)) / (d(0, 3) * d(1, 2));  // eq. (1.12)
}

ProjectiveDecomposition decompose_projective(const Eigen::Matrix3d& H) {
    if (H(2, 2) == 0.0) throw std::invalid_argument("decompose_projective: H(2,2) must be non-zero");
    const Eigen::Matrix3d Hn = H / H(2, 2);                      // step 1
    const Eigen::Vector2d v = Hn.block<1, 2>(2, 0).transpose();  // step 2
    const Eigen::Vector2d t = Hn.block<2, 1>(0, 2);
    const Eigen::Matrix2d M = Hn.topLeftCorner<2, 2>() - t * v.transpose();  // step 3
    if (M.determinant() <= 0.0) throw std::invalid_argument("decompose_projective: H reverses orientation");

    // Step 4: M = Q U' with Q a rotation and U' upper triangular with positive diagonal.
    Eigen::HouseholderQR<Eigen::Matrix2d> qr(M);
    Eigen::Matrix2d Q = qr.householderQ();
    Eigen::Matrix2d Up = qr.matrixQR().triangularView<Eigen::Upper>();
    for (int i = 0; i < 2; ++i) {
        if (Up(i, i) < 0.0) {
            Up.row(i) *= -1.0;
            Q.col(i) *= -1.0;
        }
    }
    // det M > 0 and det U' > 0 imply det Q = +1, so Q is a rotation.

    const double s = std::sqrt(Up.determinant());  // step 5
    ProjectiveDecomposition out;
    out.scale = s;
    out.angle = std::atan2(Q(1, 0), Q(0, 0));
    out.similarity = make_affine(s * Q, t);
    out.affine = make_affine(Up / s, Eigen::Vector2d::Zero());
    out.projective = Eigen::Matrix3d::Identity();
    out.projective.block<1, 2>(2, 0) = v.transpose();
    return out;
}

Eigen::Matrix3d affine_rectification(const Eigen::Vector3d& vanishing_line) {
    if (vanishing_line.z() == 0.0)
        throw std::invalid_argument(
            "affine_rectification: the vanishing line must not pass through the origin");
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    H.row(2) = vanishing_line.transpose();  // eq. (1.15)
    return H;
}

}  // namespace mvg
