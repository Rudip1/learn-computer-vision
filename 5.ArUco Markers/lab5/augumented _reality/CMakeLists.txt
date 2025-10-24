cmake_minimum_required(VERSION 3.1)
project(AugmentedReality)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Define source files
set(pose_source src/pose_estimation.cpp)
set(cube_source src/draw_cube.cpp)

# Create pose estimation executable
add_executable(pose_estimation ${pose_source})
target_link_libraries(pose_estimation ${OpenCV_LIBS})

# Create draw cube executable
add_executable(draw_cube ${cube_source})
target_link_libraries(draw_cube ${OpenCV_LIBS})

# Compiler optimizations
target_compile_options(pose_estimation PRIVATE -O3 -std=c++11)
target_compile_options(draw_cube PRIVATE -O3 -std=c++11)
