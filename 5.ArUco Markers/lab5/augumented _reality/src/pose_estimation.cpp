#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <dictionary> <marker_id> <marker_length_in_meters>\n";
        return -1;
    }

    string dictName = argv[1];
    int markerId = atoi(argv[2]);
    float markerLength = atof(argv[3]);

    // Dictionary selection
    int dictId;
    if (dictName == "DICT_6X6_250") dictId = aruco::DICT_6X6_250;
    else if (dictName == "DICT_4X4_50") dictId = aruco::DICT_4X4_50;
    else if (dictName == "DICT_ARUCO_ORIGINAL") dictId = aruco::DICT_ARUCO_ORIGINAL;
    else {
        cerr << " Invalid dictionary name" << endl;
        return -1;
    }

    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>(aruco::getPredefinedDictionary(dictId));
    Ptr<aruco::DetectorParameters> parameters = makePtr<aruco::DetectorParameters>();

    // Load camera calibration
    FileStorage fs("camera_calib.yaml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << " Failed to open camera_calib.yaml" << endl;
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs.release();

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << " Failed to open camera" << endl;
        return -1;
    }

    cout << " Starting marker detection... Press 'q' to quit.\n";

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        vector<int> ids;
        vector<vector<Point2f>> corners;
        vector<Vec3d> rvecs, tvecs;

        aruco::detectMarkers(frame, dictionary, corners, ids, parameters);

        if (!ids.empty()) {
            aruco::drawDetectedMarkers(frame, corners, ids);
            aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            for (size_t i = 0; i < ids.size(); ++i) {
                // Draw axis for every detected marker
                drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1f); // 10cm axis

                // Optional: Highlight your specific marker ID
                Scalar color = (ids[i] == markerId) ? Scalar(0,255,0) : Scalar(0,0,255);
                circle(frame, corners[i][0], 5, color, FILLED);

                // Text info
                stringstream ss;
                ss << "ID: " << ids[i] << " Pos: ["
                   << fixed << setprecision(2)
                   << tvecs[i][0] << ", " << tvecs[i][1] << ", " << tvecs[i][2] << "] m";
                putText(frame, ss.str(), corners[i][0] + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
            }
        }

        imshow("Pose Estimation", frame);
        if (waitKey(10) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
