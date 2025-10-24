#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <dictionary> <id> <size> <filename>\n";
        return -1;
    }

    // Parse arguments
    string dictName = argv[1];
    int markerId = stoi(argv[2]);
    int markerSize = stoi(argv[3]);
    string filename = argv[4];

    // Select dictionary based on user input
    cv::aruco::PredefinedDictionaryType dictEnum;
    if (dictName == "DICT_6X6_250") dictEnum = cv::aruco::DICT_6X6_250;
    else if (dictName == "DICT_4X4_50") dictEnum = cv::aruco::DICT_4X4_50;
    else if (dictName == "DICT_ARUCO_ORIGINAL") dictEnum = cv::aruco::DICT_ARUCO_ORIGINAL;
    else {
        cerr << "Invalid dictionary name\n";
        return -1;
    }

    // Load dictionary
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictEnum);

    // Create marker image with white border
    int borderSize = markerSize / 4; // Border size
    int totalSize = markerSize + 2 * borderSize;
    Mat markerImage(markerSize, markerSize, CV_8UC1, Scalar(255));
    Mat markerWithBorder(totalSize, totalSize, CV_8UC1, Scalar(255)); // White background

    // **Use `generateImageMarker` instead of `drawMarker`**
    cv::aruco::generateImageMarker(dictionary, markerId, markerSize, markerImage);

    // Place marker inside white border
    markerImage.copyTo(markerWithBorder(Rect(borderSize, borderSize, markerSize, markerSize)));

    // Save marker image
    imwrite(filename, markerWithBorder);

    cout << "Marker saved as " << filename << " with white border!\n";

    return 0;
}
