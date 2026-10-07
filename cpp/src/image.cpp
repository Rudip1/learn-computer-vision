#include "mvg/image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mvg {

namespace {

std::vector<float> gaussian_kernel(double sigma) {
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<float> k(2 * r + 1);
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += (k[i + r] = static_cast<float>(std::exp(-0.5 * i * i / (sigma * sigma))));
    for (auto& v : k) v = static_cast<float>(v / sum);
    return k;
}

// Correlate rows (horizontal = true) or columns with a 1-D kernel, replicating the border.
Image filter_1d(const Image& img, const std::vector<float>& k, bool horizontal) {
    const int r = static_cast<int>(k.size() / 2);
    const int h = static_cast<int>(img.rows()), w = static_cast<int>(img.cols());
    Image out(h, w);
    for (int v = 0; v < h; ++v) {
        for (int u = 0; u < w; ++u) {
            float acc = 0.0f;
            for (int i = -r; i <= r; ++i) {
                const int uu = horizontal ? std::clamp(u + i, 0, w - 1) : u;
                const int vv = horizontal ? v : std::clamp(v + i, 0, h - 1);
                acc += k[i + r] * img(vv, uu);
            }
            out(v, u) = acc;
        }
    }
    return out;
}

}  // namespace

Image gaussian_blur(const Image& img, double sigma) {
    if (sigma <= 0.0) return img;
    const auto k = gaussian_kernel(sigma);
    return filter_1d(filter_1d(img, k, true), k, false);
}

std::pair<Image, Image> image_gradients(const Image& img) {
    // Sobel = smoothing [1 2 1]/4 in one direction and central difference [-1 0 1]/2 in the other.
    const std::vector<float> smooth{0.25f, 0.5f, 0.25f}, diff{-0.5f, 0.0f, 0.5f};
    Image gu = filter_1d(filter_1d(img, diff, true), smooth, false);
    Image gv = filter_1d(filter_1d(img, smooth, true), diff, false);
    return {gu, gv};
}

float sample_bilinear(const Image& img, double u, double v) {
    const int w = static_cast<int>(img.cols()), h = static_cast<int>(img.rows());
    u = std::clamp(u, 0.0, w - 1.0);
    v = std::clamp(v, 0.0, h - 1.0);
    const int u0 = std::min(static_cast<int>(u), w - 2 < 0 ? 0 : w - 2);
    const int v0 = std::min(static_cast<int>(v), h - 2 < 0 ? 0 : h - 2);
    const int u1 = std::min(u0 + 1, w - 1), v1 = std::min(v0 + 1, h - 1);
    const double a = u - u0, b = v - v0;
    return static_cast<float>((1 - a) * (1 - b) * img(v0, u0) + a * (1 - b) * img(v0, u1) +
                              (1 - a) * b * img(v1, u0) + a * b * img(v1, u1));
}

Image downsample(const Image& img) {
    const Image s = gaussian_blur(img, 1.0);
    const Eigen::Index h = (img.rows() + 1) / 2, w = (img.cols() + 1) / 2;
    Image out(h, w);
    for (Eigen::Index v = 0; v < h; ++v)
        for (Eigen::Index u = 0; u < w; ++u) out(v, u) = s(2 * v, 2 * u);
    return out;
}

Eigen::MatrixXd integral_image(const Image& img) {
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(img.rows() + 1, img.cols() + 1);
    for (Eigen::Index v = 0; v < img.rows(); ++v)
        for (Eigen::Index u = 0; u < img.cols(); ++u)
            S(v + 1, u + 1) = img(v, u) + S(v, u + 1) + S(v + 1, u) - S(v, u);
    return S;
}

Image box_mean(const Image& img, int r) {
    const Eigen::MatrixXd S = integral_image(img);
    const int h = static_cast<int>(img.rows()), w = static_cast<int>(img.cols());
    Image out(h, w);
    for (int v = 0; v < h; ++v) {
        const int v0 = std::max(0, v - r), v1 = std::min(h, v + r + 1);
        for (int u = 0; u < w; ++u) {
            const int u0 = std::max(0, u - r), u1 = std::min(w, u + r + 1);
            const double sum = S(v1, u1) - S(v0, u1) - S(v1, u0) + S(v0, u0);
            out(v, u) = static_cast<float>(sum / ((v1 - v0) * (u1 - u0)));
        }
    }
    return out;
}

}  // namespace mvg
