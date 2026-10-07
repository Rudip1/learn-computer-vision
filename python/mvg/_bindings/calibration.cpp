// Chapter 3 bindings: normalization, DLT, decomposition, Zhang's method, Levenberg-Marquardt, calibration.

#include "mvg/calibration.hpp"

#include <pybind11/functional.h>

#include "common.hpp"
#include "mvg/homography.hpp"
#include "mvg/normalization.hpp"
#include "mvg/optimization.hpp"

namespace mvg::py_detail {

void bind_calibration(py::module_& m) {
    m.def(
        "normalization_transform", [](const RowPoints2& x) { return normalization_transform(cols(x)); },
        py::arg("x"), "3x3 Hartley normalization of N x 2 points (eq. 3.1).");
    m.def(
        "normalization_transform", [](const RowPoints3& X) { return normalization_transform(cols(X)); },
        py::arg("X"), "4x4 Hartley normalization of N x 3 points.");

    m.def(
        "estimate_projection_dlt",
        [](const RowPoints3& X, const RowPoints2& x, bool normalize) {
            return estimate_projection_dlt(cols(X), cols(x), normalize);
        },
        py::arg("X"), py::arg("x"), py::arg("normalize") = true, "Normalized DLT for P (section 3.3).");
    m.def(
        "estimate_projection_hall",
        [](const RowPoints3& X, const RowPoints2& x) { return estimate_projection_hall(cols(X), cols(x)); },
        py::arg("X"), py::arg("x"), "Hall's method, p34 = 1 (eq. 3.4).");
    m.def("rq_decomposition", &rq_decomposition, py::arg("M"), "M = K R, K upper triangular (eq. 3.5).");

    py::class_<ProjectionDecomposition>(m, "ProjectionDecomposition")
        .def_readonly("K", &ProjectionDecomposition::K)
        .def_readonly("R", &ProjectionDecomposition::R)
        .def_readonly("t", &ProjectionDecomposition::t)
        .def_readonly("center", &ProjectionDecomposition::center);
    m.def("decompose_projection", &decompose_projection, py::arg("P"), "P -> K, R, t (section 3.4).");

    m.def(
        "estimate_homography_dlt",
        [](const RowPoints2& x, const RowPoints2& xp, bool normalize) {
            return estimate_homography_dlt(cols(x), cols(xp), normalize);
        },
        py::arg("x"), py::arg("xp"), py::arg("normalize") = true, "DLT for H with x' ~ H x (eq. 3.7).");
    m.def("zhang_constraints", &zhang_constraints, py::arg("H"),
          "The 2 x 6 rows of eq. (3.10) for one view.");
    m.def("intrinsics_from_homographies", &intrinsics_from_homographies, py::arg("Hs"),
          py::arg("zero_skew") = false, "Zhang's closed form for K (eqs. 3.8-3.11).");
    m.def("pose_from_homography", &pose_from_homography, py::arg("K"), py::arg("H"),
          "(R, t) of a planar target from its homography (eq. 3.12).");

    py::class_<LMOptions>(m, "LMOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &LMOptions::max_iterations)
        .def_readwrite("initial_lambda", &LMOptions::initial_lambda)
        .def_readwrite("gradient_tolerance", &LMOptions::gradient_tolerance)
        .def_readwrite("cost_tolerance", &LMOptions::cost_tolerance)
        .def_readwrite("step_tolerance", &LMOptions::step_tolerance)
        .def_readwrite("numeric_step", &LMOptions::numeric_step);
    py::class_<LMResult>(m, "LMResult")
        .def_readonly("x", &LMResult::x)
        .def_readonly("initial_cost", &LMResult::initial_cost)
        .def_readonly("final_cost", &LMResult::final_cost)
        .def_readonly("iterations", &LMResult::iterations)
        .def_readonly("cost_history", &LMResult::cost_history)
        .def_readonly("stop_reason", &LMResult::stop_reason);
    m.def(
        "levenberg_marquardt",
        [](const ResidualFunction& f, const Eigen::VectorXd& x0, const LMOptions& opt,
           const std::optional<JacobianFunction>& jac) {
            py::gil_scoped_release release;  // callbacks re-acquire the GIL
            return levenberg_marquardt(f, x0, opt, jac ? *jac : JacobianFunction{});
        },
        py::arg("residual"), py::arg("x0"), py::arg("options") = LMOptions{},
        py::arg("jacobian") = py::none(),
        "Minimize ||residual(x)||^2 (section 3.8). residual and jacobian are Python callables.");
    m.def("numeric_jacobian", &numeric_jacobian, py::arg("f"), py::arg("x"), py::arg("numeric_step") = 1e-6);

    py::class_<CalibrationOptions>(m, "CalibrationOptions")
        .def(py::init<>())
        .def_readwrite("fix_skew", &CalibrationOptions::fix_skew)
        .def_readwrite("fix_principal_point", &CalibrationOptions::fix_principal_point)
        .def_readwrite("fix_k3", &CalibrationOptions::fix_k3)
        .def_readwrite("fix_tangential", &CalibrationOptions::fix_tangential)
        .def_readwrite("fix_distortion", &CalibrationOptions::fix_distortion)
        .def_readwrite("lm", &CalibrationOptions::lm);
    py::class_<CalibrationResult>(m, "CalibrationResult")
        .def_readonly("K", &CalibrationResult::K)
        .def_readonly("distortion", &CalibrationResult::distortion)
        .def_readonly("R", &CalibrationResult::R)
        .def_readonly("t", &CalibrationResult::t)
        .def_readonly("rms", &CalibrationResult::rms)
        .def_readonly("per_view_rms", &CalibrationResult::per_view_rms)
        .def_readonly("K_linear", &CalibrationResult::K_linear)
        .def_readonly("rms_linear", &CalibrationResult::rms_linear)
        .def_readonly("lm", &CalibrationResult::lm);
    m.def(
        "calibrate_planar",
        [](const RowPoints3& object_points, const std::vector<RowPoints2>& image_points,
           const CalibrationOptions& opt) {
            std::vector<Points2> obs;
            for (const auto& x : image_points) obs.push_back(cols(x));
            return calibrate_planar(cols(object_points), obs, opt);
        },
        py::arg("object_points"), py::arg("image_points"), py::arg("options") = CalibrationOptions{},
        "Zhang initialization + Levenberg-Marquardt refinement (sections 3.6-3.7). object_points: N x 3 on Z "
        "= 0; "
        "image_points: list of N x 2 arrays, one per view.");
    m.def(
        "reprojection_errors",
        [](const PinholeCamera& c, const RowPoints3& X, const RowPoints2& x) {
            return reprojection_errors(c, cols(X), cols(x));
        },
        py::arg("camera"), py::arg("X"), py::arg("x"), "||x_i - pi(X_i)|| for each point.");
}

}  // namespace mvg::py_detail
