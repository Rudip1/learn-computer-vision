#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 8) {
        cerr << "Usage: " << argv[0] << " <dictionary> <detector_params.yaml> <rows> <cols> <marker_length> <marker_separation> <output.yaml>\n";
        return -1;
    }

    string dictName = argv[1];
    string detectorParamsFile = argv[2];
    int rows = atoi(argv[3]);
    int cols = atoi(argv[4]);
    float markerLength = atof(argv[5]);
    float markerSeparation = atof(argv[6]);
    string outputFile = argv[7];

    // Select dictionary
    int dictId;
    if (dictName == "DICT_6X6_250") dictId = aruco::DICT_6X6_250;
    else if (dictName == "DICT_4X4_50") dictId = aruco::DICT_4X4_50;
    else if (dictName == "DICT_ARUCO_ORIGINAL") dictId = aruco::DICT_ARUCO_ORIGINAL;
    else {
        cerr << "Invalid dictionary name\n";
        return -1;
    }

    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>(aruco::getPredefinedDictionary(dictId));
    Ptr<aruco::GridBoard> board = makePtr<aruco::GridBoard>(Size(cols, rows), markerLength, markerSeparation, *dictionary);

    // Load detection parameters
    Ptr<aruco::DetectorParameters> detectorParams = makePtr<aruco::DetectorParameters>();
    FileStorage fs(detectorParamsFile, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << " Failed to open detector parameters file" << endl;
        return -1;
    }
    fs["adaptiveThreshWinSizeMin"] >> detectorParams->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> detectorParams->adaptiveThreshWinSizeMax;
    fs.release();

    // Capture from camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << " Cannot open camera" << endl;
        return -1;
    }

    Size imageSize;
    vector<vector<vector<Point2f>>> allCorners;
    vector<vector<int>> allIds;
    vector<int> markerCounterPerFrame;

    cout << "Press 'c' to capture, ESC to calibrate.\n";

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        imageSize = frame.size();

        vector<int> ids;
        vector<vector<Point2f>> corners, rejected;
        aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams, rejected);
        aruco::drawDetectedMarkers(frame, corners, ids);

        char key = (char)waitKey(10);
        if (key == 'c' && !ids.empty()) {
            allCorners.push_back(corners);
            allIds.push_back(ids);
            markerCounterPerFrame.push_back((int)ids.size());
            cout << "Captured " << ids.size() << " markers.\n";
        } else if (key == 27) {
            cout << " ESC pressed. Starting calibration...\n";
            break;
        }

        imshow("Calibration View", frame);
    }

    cap.release();
    destroyAllWindows();

    if (allCorners.size() < 20) {
        cerr << " Not enough captures. Need at least 20.\n";
        return -1;
    }

    // Flatten data
    vector<vector<Point2f>> flatCorners;
    vector<int> flatIds;
    for (size_t i = 0; i < allCorners.size(); ++i) {
        flatCorners.insert(flatCorners.end(), allCorners[i].begin(), allCorners[i].end());
        flatIds.insert(flatIds.end(), allIds[i].begin(), allIds[i].end());
    }

    // Calibrate
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    double repError = aruco::calibrateCameraAruco(
        flatCorners,
        flatIds,
        markerCounterPerFrame,
        board,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs
    );

    cout << "Calibration complete! Reprojection error: " << repError << endl;

    FileStorage fsOut(outputFile, FileStorage::WRITE);
    fsOut << "cameraMatrix" << cameraMatrix;
    fsOut << "distCoeffs" << distCoeffs;
    fsOut.release();

    cout << "Calibration saved to: " << outputFile << endl;
    return 0;
}
