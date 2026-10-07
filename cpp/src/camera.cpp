#include "mvg/camera.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "mvg/rotation.hpp"

namespace mvg {

Eigen::Matrix<double, 5, 1> Distortion::as_vector() const {
    Eigen::Matrix<double, 5, 1> v;
    v << k1, k2, p1, p2, k3;
    return v;
}

Distortion Distortion::from_vector(const Eigen::Matrix<double, 5, 1>& v) {
    return {v(0), v(1), v(2), v(3), v(4)};
}

Eigen::Matrix3d make_intrinsics(double fx, double fy, double cx, double cy, double skew) {
    Eigen::Matrix3d K;
    K << fx, skew, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    return K;
}

Matrix34 projection_matrix(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Matrix34 Rt;
    Rt << R, t;
    return K * Rt;
}

Eigen::Vector3d camera_center(const Matrix34& P) {
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(P, Eigen::ComputeFullV);
    const Eigen::Vector4d C = svd.matrixV().col(3);
    if (std::abs(C(3)) < 1e-15 * C.norm()) throw std::invalid_argument("camera_center: camera at infinity");
    return C.head<3>() / C(3);
}

Points2 project(const Matrix34& P, const Points3& X) {
    return (P * X.colwise().homogeneous()).colwise().hnormalized();
}

namespace {

Eigen::Vector2d distort_one(const Eigen::Vector2d& p, const Distortion& d) {
    const double x = p.x(), y = p.y();
    const double r2 = x * x + y * y;
    const double radial = 1.0 + r2 * (d.k1 + r2 * (d.k2 + r2 * d.k3));
    return {x * radial + 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x),  // eq. (2.12)
            y * radial + d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y};
}

}  // namespace

Points2 distort(const Points2& xn, const Distortion& d) {
    if (d.is_zero()) return xn;
    Points2 xd(2, xn.cols());
    for (Eigen::Index j = 0; j < xn.cols(); ++j) xd.col(j) = distort_one(xn.col(j), d);
    return xd;
}

Eigen::Matrix2d distortion_jacobian(const Eigen::Vector2d& p, const Distortion& d) {
    const double x = p.x(), y = p.y();
    const double r2 = x * x + y * y;
    const double k = 1.0 + r2 * (d.k1 + r2 * (d.k2 + r2 * d.k3));
    const double kp = d.k1 + r2 * (2.0 * d.k2 + 3.0 * r2 * d.k3);  // dk / d(r^2)
    Eigen::Matrix2d J;                                             // eq. (2.14)
    J(0, 0) = k + 2.0 * x * x * kp + 2.0 * d.p1 * y + 6.0 * d.p2 * x;
    J(0, 1) = 2.0 * x * y * kp + 2.0 * d.p1 * x + 2.0 * d.p2 * y;
    J(1, 0) = J(0, 1);
    J(1, 1) = k + 2.0 * y * y * kp + 6.0 * d.p1 * y + 2.0 * d.p2 * x;
    return J;
}

Points2 undistort(const Points2& xd, const Distortion& d, int max_iterations, double tol) {
    if (d.is_zero()) return xd;
    Points2 xn = xd;  // initial guess x_n^(0) = x_d
    for (Eigen::Index j = 0; j < xd.cols(); ++j) {
        Eigen::Vector2d p = xd.col(j);
        for (int it = 0; it < max_iterations; ++it) {  // Newton, eq. (2.13)
            const Eigen::Vector2d r = distort_one(p, d) - xd.col(j);
            if (r.squaredNorm() < tol * tol) break;
            p -= distortion_jacobian(p, d).partialPivLu().solve(r);
        }
        xn.col(j) = p;
    }
    return xn;
}

Points2 pixels_to_normalized(const Eigen::Matrix3d& K, const Points2& px) {
    return (K.partialPivLu().solve(px.colwise().homogeneous())).colwise().hnormalized();
}

Points2 normalized_to_pixels(const Eigen::Matrix3d& K, const Points2& xn) {
    return (K * xn.colwise().homogeneous()).colwise().hnormalized();
}

Eigen::Matrix<double, 2, 3> projection_jacobian(const Eigen::Matrix3d& K, const Eigen::Vector3d& Xc) {
    const double X = Xc.x(), Y = Xc.y(), Z = Xc.z();
    const double fx = K(0, 0), s = K(0, 1), fy = K(1, 1);
    Eigen::Matrix<double, 2, 3> J;                    // eq. (2.16)
    J << fx / Z, s / Z, -(fx * X + s * Y) / (Z * Z),  //
        0.0, fy / Z, -fy * Y / (Z * Z);
    return J;
}

double field_of_view(double size_px, double focal_px) { return 2.0 * std::atan(size_px / (2.0 * focal_px)); }

Points3 PinholeCamera::to_camera(const Points3& Xw) const { return (R * Xw).colwise() + t; }

Points2 PinholeCamera::project(const Points3& Xw) const {
    const Points3 Xc = to_camera(Xw);                         // step 1
    const Points2 xn = Xc.colwise().hnormalized();            // step 2
    return normalized_to_pixels(K, distort(xn, distortion));  // steps 3-4
}

Eigen::Array<bool, Eigen::Dynamic, 1> PinholeCamera::is_visible(const Points3& Xw) const {
    const Points3 Xc = to_camera(Xw);
    const Points2 px = project(Xw);
    Eigen::Array<bool, Eigen::Dynamic, 1> v(Xw.cols());
    for (Eigen::Index j = 0; j < Xw.cols(); ++j)
        v(j) = Xc(2, j) > 0.0 && px(0, j) >= 0.0 && px(0, j) < width && px(1, j) >= 0.0 && px(1, j) < height;
    return v;
}

Points3 PinholeCamera::rays(const Points2& px) const {
    const Points2 xn = undistort(pixels_to_normalized(K, px), distortion);
    Points3 d = R.transpose() * xn.colwise().homogeneous();  // eq. (2.15), direction R^T K^-1 x
    d.colwise().normalize();
    return d;
}

Points3 PinholeCamera::backproject(const Points2& px, const Eigen::VectorXd& depth) const {
    if (depth.size() != px.cols()) throw std::invalid_argument("backproject: one depth per pixel");
    const Points2 xn = undistort(pixels_to_normalized(K, px), distortion);
    Points3 Xc = xn.colwise().homogeneous();
    Xc.array().rowwise() *= depth.transpose().array();  // X_c = Z_c (x_n, y_n, 1)
    return R.transpose() * (Xc.colwise() - t);
}

Eigen::Vector3d PinholeCamera::center() const { return -R.transpose() * t; }

Matrix34 PinholeCamera::P() const { return projection_matrix(K, R, t); }

Eigen::Matrix4d PinholeCamera::pose() const { return make_transform(R.transpose(), center()); }

}  // namespace mvg
