cmake_minimum_required(VERSION 3.1)
project(CameraCalibration)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Define source files
set(calibrate_source src/calibrate_camera.cpp)

# Create calibration executable
add_executable(calibrate_camera ${calibrate_source})
target_link_libraries(calibrate_camera ${OpenCV_LIBS})

# Compiler optimizations
target_compile_options(calibrate_camera PRIVATE -O3 -std=c++11)
