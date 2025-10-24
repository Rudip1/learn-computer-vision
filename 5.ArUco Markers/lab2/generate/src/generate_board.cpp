#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <rows> <cols> <dictionary> <marker_size> <separation> <filename>\n";
        return -1;
    }

    // Parse arguments
    int rows = stoi(argv[1]);              // Number of rows
    int cols = stoi(argv[2]);              // Number of columns
    string dictName = argv[3];             // Dictionary name
    int markerSize = stoi(argv[4]);        // Marker size in pixels
    int separation = stoi(argv[5]);        // Separation between markers
    string filename = argv[6];             // Output filename

    // Select dictionary based on user input
    cv::aruco::PredefinedDictionaryType dictEnum;
    if (dictName == "DICT_6X6_250") dictEnum = cv::aruco::DICT_6X6_250;
    else if (dictName == "DICT_4X4_50") dictEnum = cv::aruco::DICT_4X4_50;
    else if (dictName == "DICT_ARUCO_ORIGINAL") dictEnum = cv::aruco::DICT_ARUCO_ORIGINAL;
    else {
        cerr << "Invalid dictionary name\n";
        return -1;
    }

    // Load the dictionary
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictEnum);

    // Use constructor instead of `create()`
    cv::Ptr<cv::aruco::GridBoard> board = cv::makePtr<cv::aruco::GridBoard>(Size(cols, rows), float(markerSize), float(separation), dictionary);

    // Board Size Calculation
    int boardWidth = (cols * markerSize) + ((cols + 1) * separation);
    int boardHeight = (rows * markerSize) + ((rows + 1) * separation);

    // Create blank image for board
    Mat boardImage(boardHeight, boardWidth, CV_8UC1, Scalar(255));  // White background

    // Ensure the board fits within the image size
    cv::aruco::drawPlanarBoard(board, boardImage.size(), boardImage, 10, 1);

    // Save the board image
    imwrite(filename, boardImage);

    cout << "Board saved as " << filename << endl;

    return 0;
}
