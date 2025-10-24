#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <dictionary> <camera_calib.yaml> <marker_length_meters>\n";
        return -1;
    }

    string dictName = argv[1];
    string calibFile = argv[2];
    float markerLength = atof(argv[3]);

    // Select dictionary
    int dictId;
    if (dictName == "DICT_6X6_250") dictId = aruco::DICT_6X6_250;
    else if (dictName == "DICT_4X4_50") dictId = aruco::DICT_4X4_50;
    else if (dictName == "DICT_ARUCO_ORIGINAL") dictId = aruco::DICT_ARUCO_ORIGINAL;
    else {
        cerr << "Invalid dictionary name." << endl;
        return -1;
    }

    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>(aruco::getPredefinedDictionary(dictId));
    Ptr<aruco::DetectorParameters> detectorParams = makePtr<aruco::DetectorParameters>();

    // Load camera calibration
    Mat cameraMatrix, distCoeffs;
    FileStorage fs(calibFile, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open calibration file." << endl;
        return -1;
    }
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs.release();

    // Open camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Could not open camera." << endl;
        return -1;
    }

    cout << "Press 'q' to quit." << endl;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        vector<int> ids;
        vector<vector<Point2f>> corners;
        aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams);

        if (!ids.empty()) {
            vector<Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            for (size_t i = 0; i < ids.size(); ++i) {
                // Define cube points centered at the marker origin
                float l = markerLength / 2.0f;
                vector<Point3f> cubePoints = {
                    {-l, -l, 0}, {l, -l, 0}, {l, l, 0}, {-l, l, 0},
                    {-l, -l, -markerLength}, {l, -l, -markerLength}, {l, l, -markerLength}, {-l, l, -markerLength}
                };

                vector<Point2f> imagePoints;
                projectPoints(cubePoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints);

                // Draw cube edges
                for (int j = 0; j < 4; ++j) {
                    line(frame, imagePoints[j], imagePoints[(j+1)%4], Scalar(255, 0, 0), 2); // bottom
                    line(frame, imagePoints[j+4], imagePoints[(j+1)%4 + 4], Scalar(255, 0, 0), 2); // top
                    line(frame, imagePoints[j], imagePoints[j+4], Scalar(255, 0, 0), 2); // sides
                }
            }
        }

        imshow("Cube Overlay", frame);
        if (waitKey(10) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
