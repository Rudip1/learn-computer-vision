# Chapter plan — learn-computer-vision   (`<pkg>` = `mvg`)

Multi-view geometry from the camera model up to visual odometry. Existing material to mine: calibration,
feature matching + homography/RANSAC, epipolar geometry and stereo, stereo VO, ArUco, event cameras (MATLAB and
reports; port to C++). OpenCV may be used for image I/O, feature *detection* and as a test reference; the geometry
is implemented here.

1. Projective geometry: homogeneous coordinates, points/lines in P², transformations hierarchy.
2. The pinhole camera: intrinsics, extrinsics, distortion; projection and back-projection.
3. Calibration: DLT, Zhang's method, nonlinear refinement (Levenberg–Marquardt written here).
4. Features and matching: detectors/descriptors (overview), ratio test, cross-check.
5. Homographies and robust estimation: normalized DLT, RANSAC, image registration and mosaics.
6. Epipolar geometry: F and E, 8-point algorithm, rank-2 enforcement, decomposition of E.
7. Triangulation and stereo: linear and midpoint triangulation, rectification, disparity to depth.
8. Pose from known points: PnP (EPnP or iterative), fiducial markers (ArUco-style decoding and pose).
9. Visual odometry: stereo VO pipeline, scale, drift; evaluation (ATE/RPE).
10. Event cameras: event generation model, event frames, simple feature tracking.
