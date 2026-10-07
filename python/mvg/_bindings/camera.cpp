// Chapter 2 bindings: rotations and the pinhole camera.

#include "mvg/camera.hpp"

#include "common.hpp"
#include "mvg/rotation.hpp"

namespace mvg::py_detail {

void bind_camera(py::module_& m) {
    m.def("rot_x", &rot_x, py::arg("angle"));
    m.def("rot_y", &rot_y, py::arg("angle"));
    m.def("rot_z", &rot_z, py::arg("angle"));
    m.def(
        "rotation_from_euler", &rotation_from_euler, py::arg("axes"), py::arg("angles"),
        "R = R_a1(angles[0]) R_a2(angles[1]) R_a3(angles[2]) for axes such as 'XYZ' or 'XYX' (section 2.1).");
    m.def("so3_exp", &so3_exp, py::arg("w"), "Rodrigues' formula (eq. 2.2).");
    m.def("so3_log", &so3_log, py::arg("R"), "Rotation vector of R (eq. 2.3).");
    m.def("is_rotation", &is_rotation, py::arg("R"), py::arg("tol") = 1e-9);
    m.def("nearest_rotation", &nearest_rotation, py::arg("M"),
          "Closest rotation in Frobenius norm (eq. 2.4).");
    m.def("look_at", &look_at, py::arg("center"), py::arg("target"), py::arg("up"),
          "World-to-camera rotation of a camera at center looking at target (eq. 2.7).");
    m.def("make_transform", &make_transform, py::arg("R"), py::arg("t"), "4x4 rigid transform [R t; 0 1].");
    m.def("invert_transform", &invert_transform, py::arg("T"));
    m.def("rotation_angle_between", &rotation_angle_between, py::arg("Ra"), py::arg("Rb"),
          "Angle of Ra^T Rb in radians.");

    py::class_<Distortion>(m, "Distortion", "Brown-Conrady coefficients (k1, k2, p1, p2, k3), eq. (2.12).")
        .def(py::init<>())
        .def(py::init([](double k1, double k2, double p1, double p2, double k3) {
                 return Distortion{k1, k2, p1, p2, k3};
             }),
             py::arg("k1") = 0.0, py::arg("k2") = 0.0, py::arg("p1") = 0.0, py::arg("p2") = 0.0,
             py::arg("k3") = 0.0)
        .def_readwrite("k1", &Distortion::k1)
        .def_readwrite("k2", &Distortion::k2)
        .def_readwrite("p1", &Distortion::p1)
        .def_readwrite("p2", &Distortion::p2)
        .def_readwrite("k3", &Distortion::k3)
        .def("as_vector", &Distortion::as_vector, "(k1, k2, p1, p2, k3), the order used by OpenCV.")
        .def_static("from_vector", &Distortion::from_vector, py::arg("v"))
        .def("__repr__", [](const Distortion& d) {
            return "Distortion(k1=" + std::to_string(d.k1) + ", k2=" + std::to_string(d.k2) +
                   ", p1=" + std::to_string(d.p1) + ", p2=" + std::to_string(d.p2) +
                   ", k3=" + std::to_string(d.k3) + ")";
        });

    m.def("make_intrinsics", &make_intrinsics, py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("skew") = 0.0, "K = [fx s cx; 0 fy cy; 0 0 1] (eq. 2.9).");
    m.def("projection_matrix", &projection_matrix, py::arg("K"), py::arg("R"), py::arg("t"),
          "P = K [R | t].");
    m.def("camera_center", &camera_center, py::arg("P"), "Right null vector of P, dehomogenized.");
    m.def(
        "project", [](const Matrix34& P, const RowPoints3& X) { return rows(project(P, cols(X))); },
        py::arg("P"), py::arg("X"), "Project N x 3 world points with P (eq. 2.10).");
    m.def(
        "distort", [](const RowPoints2& xn, const Distortion& d) { return rows(distort(cols(xn), d)); },
        py::arg("xn"), py::arg("d"), "Distort N x 2 normalized coordinates (eq. 2.12).");
    m.def("distortion_jacobian", &distortion_jacobian, py::arg("xn"), py::arg("d"), "Eq. (2.14).");
    m.def(
        "undistort",
        [](const RowPoints2& xd, const Distortion& d, int iters) {
            return rows(undistort(cols(xd), d, iters));
        },
        py::arg("xd"), py::arg("d"), py::arg("max_iterations") = 20,
        "Invert the distortion with Newton's method (eq. 2.13).");
    m.def(
        "pixels_to_normalized",
        [](const Eigen::Matrix3d& K, const RowPoints2& px) {
            return rows(pixels_to_normalized(K, cols(px)));
        },
        py::arg("K"), py::arg("px"));
    m.def(
        "normalized_to_pixels",
        [](const Eigen::Matrix3d& K, const RowPoints2& xn) {
            return rows(normalized_to_pixels(K, cols(xn)));
        },
        py::arg("K"), py::arg("xn"));
    m.def("projection_jacobian", &projection_jacobian, py::arg("K"), py::arg("Xc"),
          "d(u,v)/dX_c (eq. 2.16).");
    m.def("field_of_view", &field_of_view, py::arg("size_px"), py::arg("focal_px"), "Eq. (2.11), radians.");

    py::class_<PinholeCamera>(m, "PinholeCamera",
                              "Intrinsics, distortion, extrinsics (world -> camera), size.")
        .def(py::init([](const Eigen::Matrix3d& K, int width, int height, const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& t, const Distortion& d) {
                 PinholeCamera c;
                 c.K = K;
                 c.width = width;
                 c.height = height;
                 c.R = R;
                 c.t = t;
                 c.distortion = d;
                 return c;
             }),
             py::arg("K"), py::arg("width") = 640, py::arg("height") = 480,
             py::arg("R") = Eigen::Matrix3d::Identity(), py::arg("t") = Eigen::Vector3d::Zero(),
             py::arg("distortion") = Distortion{})
        .def_readwrite("K", &PinholeCamera::K)
        .def_readwrite("distortion", &PinholeCamera::distortion)
        .def_readwrite("R", &PinholeCamera::R)
        .def_readwrite("t", &PinholeCamera::t)
        .def_readwrite("width", &PinholeCamera::width)
        .def_readwrite("height", &PinholeCamera::height)
        .def(
            "to_camera",
            [](const PinholeCamera& c, const RowPoints3& X) { return rows(c.to_camera(cols(X))); },
            py::arg("X"))
        .def(
            "project", [](const PinholeCamera& c, const RowPoints3& X) { return rows(c.project(cols(X))); },
            py::arg("X"), "Full projection with distortion (section 2.7).")
        .def(
            "is_visible",
            [](const PinholeCamera& c, const RowPoints3& X) -> Eigen::Array<bool, Eigen::Dynamic, 1> {
                return c.is_visible(cols(X));
            },
            py::arg("X"), "In front of the camera and inside the image.")
        .def(
            "rays", [](const PinholeCamera& c, const RowPoints2& px) { return rows(c.rays(cols(px))); },
            py::arg("px"), "Unit world directions of pixels (eq. 2.15).")
        .def(
            "backproject",
            [](const PinholeCamera& c, const RowPoints2& px, const Eigen::VectorXd& depth) {
                return rows(c.backproject(cols(px), depth));
            },
            py::arg("px"), py::arg("depth"), "World points of pixels with known depth Z_c.")
        .def("center", &PinholeCamera::center)
        .def("P", &PinholeCamera::P)
        .def("pose", &PinholeCamera::pose, "T_wc, 4x4.")
        .def("__copy__", [](const PinholeCamera& c) { return PinholeCamera(c); })
        .def("__deepcopy__", [](const PinholeCamera& c, py::dict) { return PinholeCamera(c); });
}

}  // namespace mvg::py_detail
