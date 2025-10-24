cmake_minimum_required(VERSION 3.1)
project(GenerateMarkerBoard)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Define source files
set(marker_source src/generate_marker.cpp)
set(board_source src/generate_board.cpp)

# Create marker executable
add_executable(generate_marker ${marker_source})
target_link_libraries(generate_marker ${OpenCV_LIBS})

# Create board executable
add_executable(generate_board ${board_source})
target_link_libraries(generate_board ${OpenCV_LIBS})

# Compiler optimizations
target_compile_options(generate_marker PRIVATE -O3 -std=c++11)
target_compile_options(generate_board PRIVATE -O3 -std=c++11)
