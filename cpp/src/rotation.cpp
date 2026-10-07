#include "mvg/rotation.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mvg/projective.hpp"

namespace mvg {

Eigen::Matrix3d rot_x(double a) {
    const double c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3d R;
    R << 1, 0, 0, 0, c, -s, 0, s, c;
    return R;
}

Eigen::Matrix3d rot_y(double a) {
    const double c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3d R;
    R << c, 0, s, 0, 1, 0, -s, 0, c;
    return R;
}

Eigen::Matrix3d rot_z(double a) {
    const double c = std::cos(a), s = std::sin(a);
    Eigen::Matrix3d R;
    R << c, -s, 0, s, c, 0, 0, 0, 1;
    return R;
}

Eigen::Matrix3d rotation_from_euler(const std::string& axes, const Eigen::Vector3d& angles) {
    if (axes.size() != 3) throw std::invalid_argument("rotation_from_euler: axes must have three letters");
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i) {
        const char a = axes[i];
        if (i > 0 && a == axes[i - 1])
            throw std::invalid_argument("rotation_from_euler: consecutive axes must differ");
        if (a == 'X' || a == 'x')
            R = R * rot_x(angles(i));
        else if (a == 'Y' || a == 'y')
            R = R * rot_y(angles(i));
        else if (a == 'Z' || a == 'z')
            R = R * rot_z(angles(i));
        else
            throw std::invalid_argument("rotation_from_euler: axes must be X, Y or Z");
    }
    return R;
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
    const double theta2 = w.squaredNorm();
    const double theta = std::sqrt(theta2);
    double a, b;  // sin(t)/t and (1 - cos t)/t^2, eq. (2.2)
    if (theta < 1e-4) {
        a = 1.0 - theta2 / 6.0;
        b = 0.5 - theta2 / 24.0;
    } else {
        a = std::sin(theta) / theta;
        b = (1.0 - std::cos(theta)) / theta2;
    }
    const Eigen::Matrix3d W = skew(w);
    return Eigen::Matrix3d::Identity() + a * W + b * W * W;
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
    const double c = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
    const double theta = std::acos(c);
    const Eigen::Vector3d vee(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));  // 2 sin(t) n
    if (theta < 1e-4)
        return 0.5 * (1.0 + theta * theta / 6.0) * vee;  // theta / (2 sin theta) ~ 1/2 (1 + t^2/6)
    if (M_PI - theta > 1e-4) return theta / (2.0 * std::sin(theta)) * vee;  // eq. (2.3)
    // Near pi: sym(R) = cos(t) I + (1 - cos t) n n^T; take the best-conditioned column of n n^T.
    const Eigen::Matrix3d nnT = (0.5 * (R + R.transpose()) - c * Eigen::Matrix3d::Identity()) / (1.0 - c);
    Eigen::Index k = 0;
    nnT.diagonal().maxCoeff(&k);
    Eigen::Vector3d n = nnT.col(k) / std::sqrt(nnT(k, k));
    if (n.dot(vee) < 0.0) n = -n;
    return theta * n;
}

bool is_rotation(const Eigen::Matrix3d& R, double tol) {
    return (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <= tol &&
           std::abs(R.determinant() - 1.0) <= tol;
}

Eigen::Matrix3d nearest_rotation(const Eigen::Matrix3d& M) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0 ? -1.0 : 1.0;  // eq. (2.4)
    return svd.matrixU() * D * svd.matrixV().transpose();
}

Eigen::Matrix3d look_at(const Eigen::Vector3d& center, const Eigen::Vector3d& target,
                        const Eigen::Vector3d& up) {
    const Eigen::Vector3d r3 = (target - center).normalized();  // eq. (2.7)
    const Eigen::Vector3d x = r3.cross(up);
    if (x.norm() < 1e-12) throw std::invalid_argument("look_at: viewing direction is parallel to up");
    const Eigen::Vector3d r1 = x.normalized();
    const Eigen::Vector3d r2 = r3.cross(r1);
    Eigen::Matrix3d R;
    R.row(0) = r1.transpose();
    R.row(1) = r2.transpose();
    R.row(2) = r3.transpose();
    return R;
}

Eigen::Matrix4d make_transform(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = t;
    return T;
}

Eigen::Matrix4d invert_transform(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d Rt = T.topLeftCorner<3, 3>().transpose();
    return make_transform(Rt, -Rt * T.topRightCorner<3, 1>());
}

double rotation_angle_between(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb) {
    return so3_log(Ra.transpose() * Rb).norm();
}

}  // namespace mvg
