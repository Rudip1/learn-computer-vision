#pragma once
// Conversions between the column-wise C++ point sets and the row-wise (N x d) NumPy arrays of the Python API.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mvg/types.hpp"

namespace mvg::py_detail {

namespace py = pybind11;

using RowPoints2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using RowPoints3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowPoints4 = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

inline Points2 cols(const RowPoints2& x) { return x.transpose(); }
inline Points3 cols(const RowPoints3& x) { return x.transpose(); }
inline RowPoints2 rows(const Points2& x) { return x.transpose(); }
inline RowPoints3 rows(const Points3& x) { return x.transpose(); }

// One registration function per chapter; each lives in its own file.
void bind_projective(py::module_& m);
void bind_camera(py::module_& m);
void bind_calibration(py::module_& m);

}  // namespace mvg::py_detail
