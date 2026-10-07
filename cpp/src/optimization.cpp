#include "mvg/optimization.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>

namespace mvg {

Eigen::MatrixXd numeric_jacobian(const ResidualFunction& f, const Eigen::VectorXd& x, double numeric_step) {
    const Eigen::VectorXd r0 = f(x);
    Eigen::MatrixXd J(r0.size(), x.size());
    Eigen::VectorXd xp = x, xm = x;
    for (Eigen::Index k = 0; k < x.size(); ++k) {
        const double h = numeric_step * std::max(1.0, std::abs(x(k)));
        xp(k) = x(k) + h;
        xm(k) = x(k) - h;
        J.col(k) = (f(xp) - f(xm)) / (2.0 * h);
        xp(k) = x(k);
        xm(k) = x(k);
    }
    return J;
}

LMResult levenberg_marquardt(const ResidualFunction& residual, const Eigen::VectorXd& x0,
                             const LMOptions& opt, const JacobianFunction& jacobian) {
    LMResult out;
    Eigen::VectorXd x = x0;
    Eigen::VectorXd r = residual(x);
    double cost = r.squaredNorm();
    out.initial_cost = cost;
    out.cost_history.push_back(cost);
    double lambda = opt.initial_lambda;
    out.stop_reason = "maximum number of iterations";

    for (int it = 0; it < opt.max_iterations; ++it) {
        const Eigen::MatrixXd J = jacobian ? jacobian(x) : numeric_jacobian(residual, x, opt.numeric_step);
        const Eigen::VectorXd g = J.transpose() * r;  // half the gradient of the cost
        if (g.cwiseAbs().maxCoeff() < opt.gradient_tolerance) {
            out.stop_reason = "gradient below tolerance";
            break;
        }
        const Eigen::MatrixXd JtJ = J.transpose() * J;
        const Eigen::VectorXd diag = JtJ.diagonal().cwiseMax(1e-12);

        bool accepted = false;
        bool small_step = false;
        while (!accepted) {
            Eigen::MatrixXd A = JtJ;
            A.diagonal() += lambda * diag;  // eq. (3.14)
            const Eigen::VectorXd delta = A.ldlt().solve(-g);
            if (delta.norm() < opt.step_tolerance * (x.norm() + opt.step_tolerance)) {
                small_step = true;
                break;
            }
            const Eigen::VectorXd x_new = x + delta;
            const Eigen::VectorXd r_new = residual(x_new);
            const double cost_new = r_new.squaredNorm();
            if (std::isfinite(cost_new) && cost_new < cost) {
                const double rel = (cost - cost_new) / std::max(cost, 1e-300);
                x = x_new;
                r = r_new;
                cost = cost_new;
                lambda = std::max(lambda / 10.0, 1e-12);
                accepted = true;
                out.cost_history.push_back(cost);
                ++out.iterations;
                if (rel < opt.cost_tolerance) {
                    small_step = true;  // converged: the cost no longer decreases
                }
            } else {
                lambda *= 10.0;
                if (lambda > 1e16) {
                    small_step = true;
                    break;
                }
            }
        }
        if (small_step) {
            out.stop_reason = "converged (step or cost change below tolerance)";
            break;
        }
    }
    out.x = x;
    out.final_cost = cost;
    return out;
}

}  // namespace mvg
