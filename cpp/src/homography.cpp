#include "mvg/homography.hpp"

#include <Eigen/Dense>
#include <stdexcept>

#include "mvg/normalization.hpp"

namespace mvg {

Eigen::Matrix3d estimate_homography_dlt(const Points2& x, const Points2& xp, bool normalize) {
    if (x.cols() != xp.cols())
        throw std::invalid_argument("estimate_homography_dlt: point sets differ in size");
    if (x.cols() < 4) throw std::invalid_argument("estimate_homography_dlt: need at least 4 correspondences");
    const Eigen::Index n = x.cols();
    const Eigen::Matrix3d T = normalize ? normalization_transform(x) : Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Tp = normalize ? normalization_transform(xp) : Eigen::Matrix3d::Identity();
    const Points3 xh = T * x.colwise().homogeneous();
    const Points3 xph = Tp * xp.colwise().homogeneous();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 9);
    for (Eigen::Index i = 0; i < n; ++i) {  // eq. (3.7)
        const Eigen::RowVector3d X = xh.col(i).transpose();
        const double u = xph(0, i), v = xph(1, i), w = xph(2, i);
        A.block<1, 3>(2 * i, 3) = -w * X;
        A.block<1, 3>(2 * i, 6) = v * X;
        A.block<1, 3>(2 * i + 1, 0) = w * X;
        A.block<1, 3>(2 * i + 1, 6) = -u * X;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd h = svd.matrixV().col(8);
    Eigen::Matrix3d Hn;
    Hn << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    const Eigen::Matrix3d H = Tp.inverse() * Hn * T;  // eq. (3.2)
    return H / H.norm();
}

}  // namespace mvg
