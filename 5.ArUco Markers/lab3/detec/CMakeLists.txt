cmake_minimum_required(VERSION 3.1)
project(ArUcoMarkerDetection)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Define source files
set(detect_source src/detect_aruco.cpp)

# Create detection executable
add_executable(detect_aruco ${detect_source})
target_link_libraries(detect_aruco ${OpenCV_LIBS})

# Compiler optimizations
target_compile_options(detect_aruco PRIVATE -O3 -std=c++11)
