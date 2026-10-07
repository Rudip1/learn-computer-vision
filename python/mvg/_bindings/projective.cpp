// Chapter 1 bindings: projective geometry of the plane.

#include "mvg/projective.hpp"

#include "common.hpp"

namespace mvg::py_detail {

void bind_projective(py::module_& m) {
    py::enum_<TransformClass>(m, "TransformClass", "Classes of the transformation hierarchy (section 1.4).")
        .value("TRANSLATION", TransformClass::Translation)
        .value("EUCLIDEAN", TransformClass::Euclidean)
        .value("SIMILARITY", TransformClass::Similarity)
        .value("AFFINE", TransformClass::Affine)
        .value("PROJECTIVE", TransformClass::Projective);

    m.def(
        "to_homogeneous",
        [](const Eigen::MatrixXd& x) -> Eigen::MatrixXd { return to_homogeneous(x.transpose()).transpose(); },
        py::arg("points"), "N x d points -> N x (d+1) homogeneous points (eq. 1.1).");
    m.def(
        "from_homogeneous",
        [](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
            return from_homogeneous(x.transpose()).transpose();
        },
        py::arg("points"), "N x (d+1) homogeneous points -> N x d points (eq. 1.2).");
    m.def("skew", &skew, py::arg("a"), "Skew-symmetric matrix [a]_x (eq. 1.6).");
    m.def("line_through", &line_through, py::arg("x"), py::arg("y"),
          "Line through two homogeneous points (eq. 1.4).");
    m.def("intersect_lines", &intersect_lines, py::arg("l"), py::arg("m"),
          "Intersection of two lines (eq. 1.5).");
    m.def("normalize_line", &normalize_line, py::arg("l"), "Scale a line so that a^2 + b^2 = 1.");
    m.def("normalize_projective", &normalize_projective, py::arg("x"),
          "Unit norm, largest-magnitude entry positive: a canonical representative of a projective point.");
    m.def("point_line_distance", &point_line_distance, py::arg("x"), py::arg("l"),
          "Distance from the point (u, v) to the line l (eq. 1.8).");
    m.def(
        "apply_homography",
        [](const Eigen::Matrix3d& H, const RowPoints2& x) { return rows(apply_homography(H, cols(x))); },
        py::arg("H"), py::arg("x"), "Map N x 2 points through H (eq. 1.9).");
    m.def("transform_line", &transform_line, py::arg("H"), py::arg("l"),
          "Image of a line: H^-T l (eq. 1.10).");
    m.def("classify_transform", &classify_transform, py::arg("H"), py::arg("tol") = 1e-9,
          "Smallest class of the hierarchy containing H (section 1.4).");
    m.def("make_euclidean", &make_euclidean, py::arg("theta"), py::arg("tx"), py::arg("ty"));
    m.def("make_similarity", &make_similarity, py::arg("s"), py::arg("theta"), py::arg("tx"), py::arg("ty"));
    m.def("make_affine", &make_affine, py::arg("A"), py::arg("t"));
    m.def(
        "fit_transform",
        [](TransformClass c, const RowPoints2& x, const RowPoints2& xp) {
            return fit_transform(c, cols(x), cols(xp));
        },
        py::arg("cls"), py::arg("x"), py::arg("xp"),
        "Least-squares transformation of the given class (translation .. affine) mapping x to xp (section "
        "1.7).");
    m.def(
        "rms_transfer_error",
        [](const Eigen::Matrix3d& H, const RowPoints2& x, const RowPoints2& xp) {
            return rms_transfer_error(H, cols(x), cols(xp));
        },
        py::arg("H"), py::arg("x"), py::arg("xp"), "sqrt(mean ||xp_i - H(x_i)||^2).");
    m.def(
        "cross_ratio", [](const RowPoints2& x) { return cross_ratio(cols(x)); }, py::arg("x4"),
        "Cross ratio of four collinear points given as a 4 x 2 array (eq. 1.12).");

    py::class_<ProjectiveDecomposition>(m, "ProjectiveDecomposition")
        .def_readonly("similarity", &ProjectiveDecomposition::similarity)
        .def_readonly("affine", &ProjectiveDecomposition::affine)
        .def_readonly("projective", &ProjectiveDecomposition::projective)
        .def_readonly("scale", &ProjectiveDecomposition::scale)
        .def_readonly("angle", &ProjectiveDecomposition::angle);
    m.def("decompose_projective", &decompose_projective, py::arg("H"), "H = H_S H_A H_P (eq. 1.13).");
    m.def("affine_rectification", &affine_rectification, py::arg("vanishing_line"),
          "Homography sending the vanishing line back to l_inf (eq. 1.15).");
}

}  // namespace mvg::py_detail
