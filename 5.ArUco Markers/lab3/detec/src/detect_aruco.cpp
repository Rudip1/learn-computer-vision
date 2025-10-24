#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <dictionary>\n";
        return -1;
    }

    // Select dictionary
    int dictEnum;
    string dictName = argv[1];
    if (dictName == "DICT_6X6_250") dictEnum = cv::aruco::DICT_6X6_250;
    else if (dictName == "DICT_4X4_50") dictEnum = cv::aruco::DICT_4X4_50;
    else if (dictName == "DICT_ARUCO_ORIGINAL") dictEnum = cv::aruco::DICT_ARUCO_ORIGINAL;
    else {
        cerr << " Invalid dictionary name\n";
        return -1;
    }

    // FIX: Correct dictionary initialization
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(dictEnum));

    // Open video capture (default camera: 0)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera\n";
        return -1;
    }

    // Load detection parameters
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::makePtr<cv::aruco::DetectorParameters>();

    cout << "ArUco Marker Detection Started. Press 'q' to exit.\n";

    while (true) {
        Mat frame;
        cap >> frame;  // Capture frame
        if (frame.empty()) {
            cerr << " Error: Captured empty frame. Check your camera connection.\n";
            break;
        }

        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedCandidates;

        // Detect markers
        cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            // Draw marker ID and red dot at upper-left corner
            for (size_t i = 0; i < markerCorners.size(); i++) {
                putText(frame, "ID: " + to_string(markerIds[i]), markerCorners[i][0], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                circle(frame, markerCorners[i][0], 5, Scalar(0, 0, 255), -1);  // Red dot
            }

            cout << " Detected " << markerIds.size() << " markers: ";
            for (int id : markerIds) {
                cout << id << " ";
            }
            cout << endl;
        } else {
            cout << " No markers detected. Try adjusting parameters or lighting.\n";
        }

        imshow("ArUco Marker Detection", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
