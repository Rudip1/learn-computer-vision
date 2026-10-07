// Places a camera with look-at, projects the corners of a cube with and without lens distortion, and
// back-projects the pixels at their true depth (chapter 2).
//
//   ./02_camera [k1]      k1: radial distortion coefficient (default -0.25)

#include <cstdio>
#include <cstdlib>

#include "mvg/camera.hpp"
#include "mvg/rotation.hpp"

int main(int argc, char** argv) {
    const double k1 = argc > 1 ? std::atof(argv[1]) : -0.25;

    mvg::PinholeCamera cam;
    cam.K = mvg::make_intrinsics(520.0, 520.0, 320.0, 240.0);
    const Eigen::Vector3d C(3.0, -2.5, 1.8);
    cam.R = mvg::look_at(C, Eigen::Vector3d(0, 0, 0.5), Eigen::Vector3d(0, 0, 1));
    cam.t = -cam.R * C;

    mvg::Points3 cube(3, 8);
    for (int i = 0; i < 8; ++i) cube.col(i) << (i & 1), ((i >> 1) & 1), ((i >> 2) & 1);

    const mvg::Points2 ideal = cam.project(cube);
    cam.distortion.k1 = k1;
    const mvg::Points2 real = cam.project(cube);
    const Eigen::VectorXd depth = cam.to_camera(cube).row(2).transpose();
    const mvg::Points3 back = cam.backproject(real, depth);

    std::printf("camera centre (%.2f, %.2f, %.2f), horizontal FOV %.1f deg\n", cam.center().x(),
                cam.center().y(), cam.center().z(),
                mvg::field_of_view(cam.width, cam.K(0, 0)) * 180.0 / M_PI);
    std::printf("corner   pinhole (u, v)        distorted (u, v)      back-projection error\n");
    for (int i = 0; i < 8; ++i)
        std::printf("%d        (%7.2f, %7.2f)    (%7.2f, %7.2f)    %.2e\n", i, ideal(0, i), ideal(1, i),
                    real(0, i), real(1, i), (back.col(i) - cube.col(i)).norm());
    return 0;
}
