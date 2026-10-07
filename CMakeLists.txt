cmake_minimum_required(VERSION 3.20)
project(mvg VERSION 0.1.0 LANGUAGES CXX)

# Multi-view geometry library: C++ core, Catch2 tests, command-line examples and pybind11 bindings.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MVG_BUILD_TESTS "Build the Catch2 test suite" ON)
option(MVG_BUILD_EXAMPLES "Build the command-line examples" ON)
option(MVG_BUILD_PYTHON "Build the pybind11 Python module" OFF)

include(FetchContent)

# Eigen 3.4: use the system package when present, otherwise fetch it.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 not found on the system, fetching 3.4.0")
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

add_library(mvg STATIC
  cpp/src/projective.cpp
  cpp/src/rotation.cpp
  cpp/src/camera.cpp
  cpp/src/normalization.cpp
  cpp/src/optimization.cpp
  cpp/src/homography.cpp
  cpp/src/calibration.cpp
)
target_include_directories(mvg PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>)
target_link_libraries(mvg PUBLIC Eigen3::Eigen)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(mvg PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MVG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()

if(MVG_BUILD_EXAMPLES)
  add_subdirectory(cpp/examples)
endif()

if(MVG_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  file(GLOB MVG_BINDING_SOURCES CONFIGURE_DEPENDS python/mvg/_bindings/*.cpp)
  pybind11_add_module(_core MODULE ${MVG_BINDING_SOURCES})
  target_link_libraries(_core PRIVATE mvg)
  install(TARGETS _core LIBRARY DESTINATION mvg)
endif()
