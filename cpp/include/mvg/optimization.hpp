#pragma once
/// @file optimization.hpp
/// Levenberg-Marquardt for non-linear least squares, min ||r(theta)||^2, with analytic or numeric Jacobians.
///
/// Theory: 1_theory/03_calibration.md, section 3.8, eq. (3.14).

#include <Eigen/Core>
#include <functional>
#include <string>
#include <vector>

namespace mvg {

/// r(theta): residual vector (size m) at the parameters theta (size n).
using ResidualFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
/// J(theta) = dr/dtheta, m x n.
using JacobianFunction = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

struct LMOptions {
    int max_iterations = 100;
    double initial_lambda = 1e-3;
    double gradient_tolerance = 1e-10;  ///< stop when ||J^T r||_inf < this
    double cost_tolerance = 1e-12;      ///< stop when the relative decrease of the cost < this
    double step_tolerance = 1e-12;      ///< stop when ||delta|| < this * (||theta|| + this)
    double numeric_step = 1e-6;         ///< relative step of the central differences
};

struct LMResult {
    Eigen::VectorXd x;                 ///< final parameters
    double initial_cost = 0.0;         ///< ||r||^2 at the start
    double final_cost = 0.0;           ///< ||r||^2 at the end
    int iterations = 0;                ///< accepted steps
    std::vector<double> cost_history;  ///< cost after each accepted step (first entry: initial cost)
    std::string stop_reason;
};

/// Central-difference Jacobian of f at x, step h_k = numeric_step * max(1, |x_k|). Section 3.8, step 2.
Eigen::MatrixXd numeric_jacobian(const ResidualFunction& f, const Eigen::VectorXd& x,
                                 double numeric_step = 1e-6);

/// Levenberg-Marquardt with Marquardt's diagonal scaling, eq. (3.14) and the algorithm of section 3.8.
/// If `jacobian` is empty the Jacobian is computed with numeric_jacobian().
LMResult levenberg_marquardt(const ResidualFunction& residual, const Eigen::VectorXd& x0,
                             const LMOptions& options = {}, const JacobianFunction& jacobian = {});

}  // namespace mvg
