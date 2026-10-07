#pragma once
/// @file image.hpp
/// Minimal grey-level image support: a row-major float matrix (row = v, column = u), Gaussian smoothing,
/// gradients, bilinear sampling, pyramids and integral images. Used by features (chapter 4), warping (chapter 5),
/// stereo matching (chapter 7), marker detection (chapter 8) and event simulation (chapter 10).
///
/// Theory: 1_theory/04_features.md, section 4.2 (gradients and smoothing).

#include <Eigen/Core>
#include <utility>

namespace mvg {

/// Grey-level image, I(v, u) = img(row, col), values usually in [0, 1].
using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Separable Gaussian smoothing with standard deviation `sigma` (pixels); kernel radius ceil(3 sigma); borders
/// are replicated.
Image gaussian_blur(const Image& img, double sigma);

/// Image derivatives (dI/du, dI/dv) with the 3x3 Sobel operator divided by 8 (exact on linear ramps).
std::pair<Image, Image> image_gradients(const Image& img);

/// Bilinear interpolation at the continuous position (u, v); positions outside the image are clamped to the
/// border.
float sample_bilinear(const Image& img, double u, double v);

/// Smooth with sigma = 1 and keep every second pixel (one pyramid level).
Image downsample(const Image& img);

/// Integral image S with S(v, u) = sum of img over rows < v and columns < u; size (h+1) x (w+1).
Eigen::MatrixXd integral_image(const Image& img);

/// Mean over the (2r+1) x (2r+1) window centred on each pixel (window clipped at the borders).
Image box_mean(const Image& img, int r);

}  // namespace mvg
