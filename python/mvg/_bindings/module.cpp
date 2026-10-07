#include "common.hpp"

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of the mvg multi-view geometry module. Point sets are N x 2 / N x 3 arrays (rows).";
    mvg::py_detail::bind_projective(m);
    mvg::py_detail::bind_camera(m);
    mvg::py_detail::bind_calibration(m);
}
